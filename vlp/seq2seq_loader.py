from random import randint, shuffle, choices
from random import random as rand
import pickle
import math
import json
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from vlp.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline

import torchvision.transforms as transforms
from PIL import Image
# https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import imghdr
import numpy as np
import h5py


def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


class Img2txtDataset(torch.utils.data.Dataset):
    """ Load image-sentence pairs """

    def __init__(self, file_src, image_root, split, batch_size, tokenizer, max_len, file_valid_jpgs='tmp.json', use_num_imgs=-1, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[], s2s_prob=1, bi_prob=0, l2r_prob=0, enable_butd=False, tasks='img2txt'):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.s2s_prob = s2s_prob
        self.bi_prob = bi_prob
        self.l2r_prob = l2r_prob
        print('Sample seq2seq {}, bidirectional {}, and left2right {}'.format(self.s2s_prob, self.bi_prob, self.l2r_prob))
        assert(self.s2s_prob + self.bi_prob + self.l2r_prob == 1)

        # read the file into memory
        self.ex_list = []

        if tasks == 'img2txt':
            assert(len(file_src) == 1)
            with open(file_src[0], "r", encoding='utf-8') as f_src:
                # raw inputs are given
                img_dat = json.load(f_src)['images']
                counter = 0

                if not os.path.isfile(file_valid_jpgs):
                    valid_img = {}
                    for src in img_dat:
                        if src['split'] in split:
                            if use_num_imgs == -1 or counter < use_num_imgs:
                                if enable_butd:
                                    src_tk = os.path.join(image_root, src.get('filepath', 'trainval'),
                                        src['filename'][:-4]+'.npy')
                                    for sent in src['sentences']:
                                        tgt_tk = tokenizer.tokenize(sent['raw'])
                                        assert len(tgt_tk) > 0
                                        self.ex_list.append((src_tk, tgt_tk, {'answers': ['dummy']}))
                                        if counter%10000 == 0:
                                            print(src_tk, tgt_tk)
                                    counter += 1
                                else:
                                    src_tk = os.path.join(image_root, src.get('filepath', ''),
                                        src['filename'])
                                    # check if the image is valid
                                    if os.stat(src_tk).st_size > 0 and imghdr.what(src_tk) == 'jpeg':
                                        try:
                                            Image.open(src_tk)
                                            for sent in src['sentences']:
                                                tgt_tk = tokenizer.tokenize(sent['raw'])
                                                assert len(tgt_tk) > 0
                                                self.ex_list.append((src_tk, tgt_tk, {'answers': ['dummy']}))
                                            valid_img[src['filename']] = src['filename']
                                            counter += 1
                                        except:
                                            pass
                    json.dump(valid_img, open(file_valid_jpgs, 'w'))
                    print('Saving {0} valid JPG IDs'.format(len(valid_img)))
                else:
                    valid_jpgs = json.load(open(file_valid_jpgs))
                    print('Loading {0} valid JPG IDs!'.format(len(valid_jpgs)))
                    for src in img_dat:
                        if src['split'] in split:
                            if use_num_imgs == -1 or counter < use_num_imgs:
                                if enable_butd:
                                    src_tk = os.path.join(image_root, src.get('filepath', 'trainval'),
                                        src['filename'][:-4]+'.npy')
                                else:
                                    src_tk = os.path.join(image_root, src.get('filepath', ''),
                                        src['filename'])
                                # check if the image is valid
                                if src['filename'] in valid_jpgs:
                                    for sent in src['sentences']:
                                        tgt_tk = tokenizer.tokenize(sent['raw'])
                                        assert len(tgt_tk) > 0
                                        self.ex_list.append((src_tk, tgt_tk, {'answers': ['dummy']}))
                                        if counter%10000 == 0:
                                            print(src_tk, tgt_tk)
                                    counter += 1
        elif tasks == 'vqa2':
            counter = 0
            for file_s in file_src:
                img_dat = np.load(file_s, allow_pickle=True)
                assert(img_dat[0]['has_answer'] == True)
                for i in range(1, img_dat.shape[0]):
                    if use_num_imgs == -1 or counter < use_num_imgs:
                        if enable_butd:
                            src_tk = os.path.join(image_root, img_dat[i]['image_name'].split('_')[1],
                                img_dat[i]['feature_path'])
                        else:
                            raise NotImplementedError
                        tgt_tk = tokenizer.tokenize(img_dat[i]['question_str'])
                        ans_tk = {'answers': img_dat[i]['answers']}
                        self.ex_list.append((src_tk, tgt_tk, ans_tk))
                        counter += 1

        print('Load {0} documents'.format(len(self.ex_list)))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choices(self.bi_uni_pipeline, weights=[self.s2s_prob, self.bi_prob, self.l2r_prob])[0]
        instance = proc(instance)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, block_mask=False, new_segment_ids=False, truncate_config={}, mask_image_regions=False, mode="s2s", len_vis_input=49, vis_mask_prob=0.25, enable_butd=False, region_bbox_file='', region_det_file_prefix='', local_rank=-1, load_vqa_ann=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.mask_image_regions = mask_image_regions
        assert mode in ("s2s", "l2r", "bi")
        self.mode = mode
        self.region_bbox_file = region_bbox_file
        self.region_det_file_prefix = region_det_file_prefix

        if mode == 's2s':
            self.task_idx = 3   # relax projection layer for different tasks
        elif mode == 'bi':
            self.task_idx = 0
        elif mode == 'l2r':
            self.task_idx = 1

        self.len_vis_input = len_vis_input
        self.vis_mask_prob = vis_mask_prob

        # for images
        self.enable_butd = enable_butd
        if not enable_butd:
            self.Resize = transforms.Resize((255, 255))
            self.RandomCrop = transforms.RandomCrop((224, 224))
            self.ToTensor = transforms.ToTensor()
            self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            if load_vqa_ann:
                # import packages from pythia
                import pythia.tasks.processors as pythia_proc # VQAAnswerProcessor
                from pythia.utils.configuration import ConfigNode
                args = {'vocab_file': 'pythia/data/vocabs/answers_vqa.txt', 'num_answers':10, 'preprocessor':{'type':'simple_word', 'params':{}}}
                args = ConfigNode(args)
                self.ans_proc = pythia_proc.registry.get_processor_class('vqa_answer')(args)
            else:
                self.ans_proc = None


    def __call__(self, instance):
        img_path, tokens_b = instance[:2]
        tokens_a = ['[UNK]'] * self.len_vis_input

        num_truncated_a, _ = truncate_tokens_pair(tokens_a, tokens_b,
            self.max_len - 3, max_len_a=self.max_len_a, max_len_b=self.max_len_b,
            trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

        if self.new_segment_ids:
            if self.mode == 's2s':
                segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
            elif self.mode == 'bi':
                segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)
            elif self.mode == 'l2r':
                segment_ids = [2] * (len(tokens))
        else:
            segment_ids = [0] * (len(tokens_a)+2) + [1] * (len(tokens_b)+1)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(tokens_b)
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)

        masked_pos = cand_pos[:n_pred]

        if self.mask_image_regions:
            vis_masked_pos = np.random.choice(self.len_vis_input,
                int(self.len_vis_input*self.vis_mask_prob), replace=False)+1 # +1 for [CLS], always of the same length, no need to pad
        else:
            vis_masked_pos = []

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        # self-attention mask
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        second_st, second_end = len(tokens_a)+2, len(tokens_a)+len(tokens_b)+3

        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
        elif self.mode == 'bi':
            input_mask = torch.tensor([1]*len(tokens)+[0]*n_pad, dtype=torch.long) \
                .unsqueeze(0).expand(self.max_len, self.max_len).clone()
        elif self.mode == 'l2r':
            st, end = 0, len(tokens_a) + len(tokens_b) + 3
            input_mask[st:end, st:end].copy_(self._tril_matrix[:end, :end])

        if self.mask_image_regions:
            input_mask[:, vis_masked_pos].fill_(0) # block the masked visual feature

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0]*n_pad)
            masked_pos.extend([0]*n_pad)
            masked_weights.extend([0]*n_pad)

        if not self.enable_butd:
            # loading images
            img = Image.open(img_path).convert('RGB')
            img = self.Resize(img)
            img = self.RandomCrop(img)
            img = self.ToTensor(img)
            img = self.res_Normalize(img)
        else:
            # loading pre-processed features
            img_id = img_path.split('/')[-1].split('.')[0]
            if self.region_det_file_prefix != '':
                # read data from h5 files
                with h5py.File(self.region_det_file_prefix+'_feat'+img_id[-3:] +'.h5', 'r') as region_feat_f, \
                        h5py.File(self.region_det_file_prefix+'_cls'+img_id[-3:] +'.h5', 'r') as region_cls_f, \
                        h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    img = torch.from_numpy(region_feat_f[img_id][:]).float()
                    cls_label = torch.from_numpy(region_cls_f[img_id][:]).float()
                    vis_pe = torch.from_numpy(region_bbox_f[img_id][:])
            else:
                # legacy, for some datasets, read data from numpy files
                img = torch.from_numpy(np.load(img_path))
                cls_label = torch.from_numpy(np.load(img_path.replace('.npy', '_cls_prob.npy')))
                with h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    vis_pe = torch.from_numpy(region_bbox_f[img_id][:])

            # lazy normalization of the coordinates...
            w_est = torch.max(vis_pe[:, [0, 2]])*1.+1e-5
            h_est = torch.max(vis_pe[:, [1, 3]])*1.+1e-5
            vis_pe[:, [0, 2]] /= w_est
            vis_pe[:, [1, 3]] /= h_est
            assert h_est > 0, 'should greater than 0! {}'.format(h_est)
            assert w_est > 0, 'should greater than 0! {}'.format(w_est)
            rel_area = (vis_pe[:, 3]-vis_pe[:, 1])*(vis_pe[:, 2]-vis_pe[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1) # confident score
            normalized_coord = F.normalize(vis_pe.data[:, :5]-0.5, dim=-1)
            vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
                F.layer_norm(cls_label, [1601])), dim=-1) # 1601 hard coded...

            # process answer
            if self.ans_proc:
                ans_tk = self.ans_proc(instance[2])['answers_scores']
            else:
                ans_tk = img.new(1)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, -1, self.task_idx, img, vis_masked_pos, vis_pe, ans_tk)


class Preprocess4Seq2seqDecoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, new_segment_ids=False, mode="s2s", enable_butd=False, len_vis_input=49, region_bbox_file='', region_det_file_prefix=''):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        self.mode = mode
        if self.mode not in ["s2s", "l2r"]:
            raise ValueError("Invalid mode for seq2seq decode: %s" % self.mode)
        self.max_tgt_length = max_tgt_length
        self.len_vis_input = len_vis_input
        self.region_bbox_file = region_bbox_file
        self.region_det_file_prefix = region_det_file_prefix

        # for images
        self.enable_butd = enable_butd
        if not enable_butd:
            self.Resize = transforms.Resize((255, 255))
            self.CenterCrop = transforms.CenterCrop((224, 224))
            self.ToTensor = transforms.ToTensor()
            self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, instance):
        img_path, max_a_len = instance[:2]
        tokens_a = ['[UNK]'] * self.len_vis_input

        # Add Special Tokens
        padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_tgt_length +
                               max_a_len + 2, self.max_len)
        tokens = padded_tokens_a
        if self.new_segment_ids:
            if self.mode == "s2s":
                segment_ids = [4]*(len(padded_tokens_a)) + \
                    [5]*(max_len_in_batch - len(padded_tokens_a))
            else:
                segment_ids = [2]*max_len_in_batch
        else:
            segment_ids = [0]*(len(padded_tokens_a)) \
                + [1]*(max_len_in_batch - len(padded_tokens_a))

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
        else:
            st, end = 0, len(tokens_a) + 2
            input_mask[st:end, st:end].copy_(
                self._tril_matrix[:end, :end])
            input_mask[end:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        if not self.enable_butd:
            try:
                img = Image.open(img_path).convert('RGB')
                img = self.Resize(img)
                img = self.CenterCrop(img)
                img = self.ToTensor(img)
                img = self.res_Normalize(img)
            except:
                print('Unable to load image {}! Loading mean image instead...'.format(img_path))
                img = torch.Tensor(self.res_Normalize.mean).view(-1, 1, 1).expand(
                    (3, self.CenterCrop.size[0], self.CenterCrop.size[1]))
        else:
            img_id = img_path.split('/')[-1].split('.')[0]
            if self.region_det_file_prefix != '':
                # read data from h5 files
                with h5py.File(self.region_det_file_prefix+'_feat'+img_id[-3:] +'.h5', 'r') as region_feat_f, \
                        h5py.File(self.region_det_file_prefix+'_cls'+img_id[-3:] +'.h5', 'r') as region_cls_f, \
                        h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    img = torch.from_numpy(region_feat_f[img_id][:]).float()
                    cls_label = torch.from_numpy(region_cls_f[img_id][:]).float()
                    vis_pe = torch.from_numpy(region_bbox_f[img_id][:])
            else:
                # legacy, for some datasets, read data from numpy files
                img = torch.from_numpy(np.load(img_path))
                cls_label = torch.from_numpy(np.load(img_path.replace('.npy', '_cls_prob.npy')))
                with h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    vis_pe = torch.from_numpy(region_bbox_f[img_id][:])

            # lazy normalization of the coordinates...
            w_est = torch.max(vis_pe[:, [0, 2]])*1.+1e-5
            h_est = torch.max(vis_pe[:, [1, 3]])*1.+1e-5
            vis_pe[:, [0, 2]] /= w_est
            vis_pe[:, [1, 3]] /= h_est
            rel_area = (vis_pe[:, 3]-vis_pe[:, 1])*(vis_pe[:, 2]-vis_pe[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1) # confident score
            normalized_coord = F.normalize(vis_pe.data[:, :5]-0.5, dim=-1)
            vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
                F.layer_norm(cls_label, [1601])), dim=-1) # 1601 hard coded...

        return (input_ids, segment_ids, position_ids, input_mask, self.task_idx, img, vis_pe)
