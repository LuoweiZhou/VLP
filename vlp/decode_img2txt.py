"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import json
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import pickle

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.seq2seq_loader as seq2seq_loader
from vlp.lang_utils import language_eval
from misc.data_parallel import DataParallelImbalance

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# SPECIAL_TOKEN = ["[UNK]", "[PAD]", "[CLS]", "[MASK]"]


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased")
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument('--max_position_embeddings', type=int, default=512,
                        help="max position embeddings")

    # For decoding
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--max_tgt_length', type=int, default=20,
                        help="maximum length of target sequence")

    # Others for VLP
    parser.add_argument("--src_file", default='/mnt/dat/COCO/annotations/dataset_coco.json', type=str,		
                        help="The input data file name.")		
    parser.add_argument('--dataset', default='coco', type=str,
                        help='coco | flickr30k | cc')
    parser.add_argument('--len_vis_input', type=int, default=100)
    # parser.add_argument('--resnet_model', type=str, default='imagenet_weights/resnet101.pth')
    parser.add_argument('--image_root', type=str, default='/mnt/dat/COCO/images')		
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--enable_butd', action='store_true',
                        help='set to take in region features')
    parser.add_argument('--region_bbox_file', default='coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5', type=str)
    parser.add_argument('--region_det_file_prefix', default='feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval', type=str)
    parser.add_argument('--file_valid_jpgs', default='', type=str)

    args = parser.parse_args()

    if args.enable_butd:
        assert(args.len_vis_input == 100)
        args.region_bbox_file = os.path.join(args.image_root, args.region_bbox_file)
        args.region_det_file_prefix = os.path.join(args.image_root, args.region_det_file_prefix) if args.dataset in ('cc', 'coco') and args.region_det_file_prefix != '' else ''

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    args.max_seq_length = args.max_tgt_length + args.len_vis_input + 3 # +3 for 2x[SEP] and [CLS]
    tokenizer.max_len = args.max_seq_length

    bi_uni_pipeline = []
    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(
        tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
        max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids,
        mode='s2s', len_vis_input=args.len_vis_input, enable_butd=args.enable_butd,
        region_bbox_file=args.region_bbox_file, region_det_file_prefix=args.region_det_file_prefix))

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(
        ["[MASK]", "[SEP]"])
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))
    print(args.model_recover_path)
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model,
            max_position_embeddings=args.max_position_embeddings, config_path=args.config_path,
            state_dict=model_recover, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id,
            search_beam_size=args.beam_size, length_penalty=args.length_penalty,
            eos_id=eos_word_ids, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
            forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size, min_len=args.min_len,
            enable_butd=args.enable_butd, len_vis_input=args.len_vis_input)
        del model_recover

        # from vlp.resnet import resnet		
        # cnn = resnet(args.resnet_model, _num_layers=101, _fixed_block=4, pretrained=True) # no finetuning

        if args.fp16:
            model.half()
            # cnn.half()
        model.to(device)
        # cnn.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
            # cnn = torch.nn.DataParallel(cnn)

        torch.cuda.empty_cache()
        model.eval()
        # cnn.eval()

        eval_lst = []
        with open(args.src_file, "r", encoding='utf-8') as f_src:
            img_dat = json.load(f_src)['images']
            img_idx = 0
            valid_jpgs = None if (args.file_valid_jpgs == '' or args.dataset in \
                ('coco', 'flickr30k')) else json.load(open(args.file_valid_jpgs))
            for src in img_dat:
                if src['split'] == args.split and (valid_jpgs is None or src['filename'] in valid_jpgs):
                    if args.enable_butd:
                        src_tk = os.path.join(args.image_root, src.get('filepath', 'trainval'), src['filename'][:-4]+'.npy')
                    else:
                        src_tk = os.path.join(args.image_root, src.get('filepath', 'trainval'), src['filename'])
                    if args.dataset == 'coco':
                        imgid = int(src['filename'].split('_')[2][:-4])
                    elif args.dataset == 'cc':
                        imgid = int(src['imgid'])
                    elif args.dataset == 'flickr30k':
                        imgid = int(src['filename'].split('.')[0])
                    eval_lst.append((img_idx, imgid, src_tk)) # id and path for COCO
                    img_idx += 1
        input_lines = eval_lst

        next_i = 0
        output_lines = [""] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        print('start the caption evaluation...')
        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[2] for x in _chunk]
                next_i += args.batch_size
                instances = []
                for instance in [(x, args.len_vis_input) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = batch_list_to_batch_tensors(
                        instances)
                    batch = [t.to(device) for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, task_idx, img, vis_pe = batch

                    if args.fp16:
                        img = img.half()
                        vis_pe = vis_pe.half()

                    if args.enable_butd:
                        conv_feats = img.data # Bx100x2048
                        vis_pe = vis_pe.data
                    else:
                        conv_feats, _ = cnn(img.data) # Bx2048x7x7
                        conv_feats = conv_feats.view(conv_feats.size(0), conv_feats.size(1),
                            -1).permute(0,2,1).contiguous()

                    traces = model(conv_feats, vis_pe, input_ids, token_type_ids,
                                   position_ids, input_mask, task_idx=task_idx)
                    if args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces[0].tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]"):
                                break
                            output_tokens.append(t)
                        output_sequence = ' '.join(detokenize(output_tokens))
                        output_lines[buf_id[i]] = output_sequence

                pbar.update(1)

        predictions = [{'image_id': tup[1], 'caption': output_lines[img_idx]} for img_idx, tup in enumerate(input_lines)]

        lang_stats = language_eval(args.dataset, predictions, args.model_recover_path.split('/')[-2]+'-'+args.split+'-'+args.model_recover_path.split('/')[-1].split('.')[-2], args.split)


if __name__ == "__main__":
    main()
