
# from https://github.com/jiasenlu/NeuralBabyTalk/blob/master/misc/utils.py

import os
import json

def language_eval(dataset, preds, model_id, split):
    import sys
    sys.path.append("coco-caption")
    if dataset == 'coco':
        annFile = 'coco-caption/annotations/captions_val2014.json'
    elif dataset == 'flickr30k':
        annFile = 'coco-caption/annotations/caption_flickr30k.json'
    elif dataset == 'cc':
        annFile = 'coco-caption/annotations/caption_cc_val.json'

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')

    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    coco = COCO(annFile)
    valids = coco.getImgIds()
    # valids = json.load(open('/mnt/dat/CC/annotations/cc_valid_jpgs.json'))
    # valids = {int(i[:-4]):int(i[:-4]) for i,j in valids.items()}

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    # print(preds_filt)
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    imgToEval = cocoEval.imgToEval

    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out
