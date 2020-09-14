# Based on https://github.com/ruotianluo/self-critical.pytorch

import sys
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np

sys.path.append("coco-caption")
from pycocoevalcap.cider.cider import Cider

CiderD_scorer = Cider(df='corpus')


def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)


def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def get_self_critical_reward(greedy_res, gt_ids, gen_result, batch_size):

    greedy_res = greedy_res.data.cpu().numpy()
    gen_result = gen_result.data.cpu().numpy()
    gt_ids = gt_ids.data.cpu().numpy()
    res = OrderedDict()

    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(batch_size):
        gts[i] = [array_to_str(gt_ids[i])]
    for i in range(batch_size):
        gts[batch_size + i] = [array_to_str(gt_ids[i])]

    cider_reward_weight = 1
    # print(gts, res)
    _, cider_scores = CiderD_scorer.compute_score(gts, res)
    # print('Cider scores:', _)

    scores = cider_reward_weight * cider_scores
    scores = scores[:batch_size] - scores[batch_size:]
    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output
