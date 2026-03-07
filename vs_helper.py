from itertools import groupby
from operator import itemgetter
from typing import Tuple, Iterable, List
import torch

def simple_knapsack(values, weights, capacity):
    n = len(values)
    if n == 0 or capacity <= 0:
        return []
    values = list(values)
    weights = list(weights)
    capacity = int(capacity)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    packed_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            packed_items.append(i - 1)
            w -= weights[i - 1]
    packed_items.reverse()
    return packed_items

def seq2bbox(sequence):
    sequence = sequence.bool()
    selected_indices = torch.nonzero(sequence).squeeze().to(sequence.device)
    bboxes_lr = []
    for k, g in groupby(enumerate(selected_indices), lambda x: x[0] - x[1]):
        segment = list(map(itemgetter(1), g))
        start_frame, end_frame = segment[0], segment[-1] + 1
        bboxes_lr.append([start_frame, end_frame])
    bboxes_lr = torch.tensor(bboxes_lr, dtype=torch.int32).to(sequence.device)
    return bboxes_lr

def iou_lr(anchor_bbox, target_bbox):
    anchor_left, anchor_right = anchor_bbox[:, 0], anchor_bbox[:, 1]
    target_left, target_right = target_bbox[:, 0], target_bbox[:, 1]
    inter_left = torch.max(anchor_left, target_left)
    inter_right = torch.min(anchor_right, target_right)
    union_left = torch.min(anchor_left, target_left)
    union_right = torch.max(anchor_right, target_right)
    intersect = inter_right - inter_left
    intersect[intersect < 0] = 0
    union = union_right - union_left
    union[union <= 0] = 1e-6
    iou = intersect / union
    return iou

def nms(scores, bboxes, thresh):
    valid_idx = bboxes[:, 0] < bboxes[:, 1]
    scores = scores[valid_idx]
    bboxes = bboxes[valid_idx]
    arg_desc = torch.argsort(scores, descending=True)
    scores_remain = scores[arg_desc]
    bboxes_remain = bboxes[arg_desc]
    keep_bboxes = []
    keep_scores = []
    while bboxes_remain.size(0) > 0:
        bbox = bboxes_remain[0]
        score = scores_remain[0]
        keep_bboxes.append(bbox)
        keep_scores.append(score)
        iou = iou_lr(bboxes_remain, bbox.unsqueeze(0))
        keep_indices = (iou < thresh)
        bboxes_remain = bboxes_remain[keep_indices]
        scores_remain = scores_remain[keep_indices]
    keep_bboxes = torch.stack(keep_bboxes)
    keep_scores = torch.stack(keep_scores)
    return keep_scores, keep_bboxes

def get_loc_label(target):
    seq_len, = target.shape
    bboxes = seq2bbox(target)
    offsets = bbox2offset(bboxes, seq_len)
    return offsets

def get_ctr_label(target, offset, eps=1e-8):
    target = target.bool()
    ctr_label = torch.zeros(target.shape, dtype=torch.float32).to(target.device)
    offset_left, offset_right = offset[target, 0], offset[target, 1]
    ctr_label[target] = torch.minimum(offset_left, offset_right) / (torch.maximum(offset_left, offset_right) + eps)
    return ctr_label

def bbox2offset(bboxes, seq_len):
    pos_idx = torch.arange(seq_len, dtype=torch.float32).to(bboxes.device)
    offsets = torch.zeros((seq_len, 2), dtype=torch.float32).to(bboxes.device)
    for lo, hi in bboxes:
        bbox_pos = pos_idx[lo:hi]
        offsets[lo:hi] = torch.stack((bbox_pos - lo, hi - 1 - bbox_pos), dim=1)
    return offsets

def offset2bbox(offsets):
    offset_left, offset_right = offsets[:, 0], offsets[:, 1]
    seq_len, _ = offsets.shape
    indices = torch.arange(seq_len).to(offset_left.device)
    bbox_left = indices - offset_left
    bbox_right = indices + offset_right + 1
    bboxes = torch.stack((bbox_left, bbox_right), dim=1)
    return bboxes

def f1_score(pred, test):
    assert pred.shape == test.shape
    pred = pred.bool()
    test = test.bool().to(pred.device)
    overlap = (pred & test).sum()
    if overlap == 0:
        return 0.0
    precision = overlap.float() / pred.sum().float()
    recall = overlap.float() / test.sum().float()
    f1 = 2 * precision * recall / (precision + recall)
    return float(f1)

def knapsack(values, weights, capacity):
    values = list(values)
    weights = list(weights)
    capacity = int(capacity)
    return simple_knapsack(values, weights, capacity)

def downsample_summ(summ):
    return summ[::15]

def get_keyshot_summ(pred, cps, n_frames, nfps, picks, proportion=0.15):
    assert pred.shape == picks.shape
    frame_scores = torch.zeros(n_frames, dtype=torch.float32).to(pred.device)
    for i in range(len(picks)):
        pos_lo = picks[i]
        pos_hi = picks[i + 1] if i + 1 < len(picks) else n_frames
        frame_scores[pos_lo:pos_hi] = pred[i]
    seg_scores = torch.zeros(len(cps), dtype=torch.int32).to(pred.device)
    for seg_idx, (first, last) in enumerate(cps):
        scores = frame_scores[first:last + 1]
        seg_scores[seg_idx] = int(1000 * scores.mean())
    limits = int(n_frames * proportion)
    seg_scores_list = seg_scores.cpu().tolist()
    nfps_list = nfps.cpu().tolist() if isinstance(nfps, torch.Tensor) else list(nfps)
    packed = knapsack(seg_scores_list, nfps_list, limits)
    summary = torch.zeros(n_frames, dtype=torch.bool).to(pred.device)
    for seg_idx in packed:
        first, last = cps[seg_idx]
        summary[first:last + 1] = True
    return summary, frame_scores

def bbox2summary(seq_len, pred_cls, pred_bboxes, change_points, n_frames, nfps, picks):
    score = torch.zeros(seq_len, dtype=torch.float32).to(pred_cls.device)
    for bbox_idx in range(len(pred_bboxes)):
        lo, hi = pred_bboxes[bbox_idx, 0], pred_bboxes[bbox_idx, 1]
        score[lo:hi] = torch.maximum(score[lo:hi], pred_cls[bbox_idx])
    pred_summ, pred_score_upsampled = get_keyshot_summ(score, change_points, n_frames, nfps, picks)
    return pred_summ, pred_score_upsampled

def get_summ_diversity(pred_summ, features):
    assert len(pred_summ) == len(features)
    pred_summ = pred_summ.bool()
    pos_features = features[pred_summ]
    if len(pos_features) < 2:
        return 0.0
    diversity = 0.0
    for feat in pos_features:
        diversity += (feat * pos_features).sum() - (feat * feat).sum()
    diversity /= len(pos_features) * (len(pos_features) - 1)
    return diversity

def get_summ_f1score(pred_summ, test_summ, eval_metric='avg'):
    pred_summ = pred_summ.bool()
    test_summ = test_summ.bool()
    _, n_frames = test_summ.shape
    if pred_summ.size(0) > n_frames:
        pred_summ = pred_summ[:n_frames]
    elif pred_summ.size(0) < n_frames:
        pred_summ = torch.nn.functional.pad(pred_summ, (0, n_frames - pred_summ.size(0)))
    f1s = [f1_score(user_summ, pred_summ) for user_summ in test_summ]
    if eval_metric == 'avg':
        final_f1 = torch.mean(torch.tensor(f1s))
    elif eval_metric == 'max':
        final_f1 = torch.max(torch.tensor(f1s))
    else:
        raise ValueError(f'Invalid eval metric {eval_metric}')
    return float(final_f1)