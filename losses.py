import torch
from torch.nn import functional as F


def calc_cls_loss(pred: torch.Tensor, target: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """Classification loss with label smoothing"""
    target = target.type(torch.long)
    num_pos = target.sum()
    
    # Label smoothing
    smooth_target = target.float() * (1 - smoothing) + 0.5 * smoothing
    
    pred = pred.unsqueeze(-1)
    pred = torch.cat([1 - pred, pred], dim=-1)
    
    loss = focal_loss(pred, target, smooth_target)
    loss = loss / (num_pos + 1e-8)
    return loss


def iou_offset(offset_a: torch.Tensor, offset_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    left_a, right_a = offset_a[:, 0], offset_a[:, 1]
    left_b, right_b = offset_b[:, 0], offset_b[:, 1]

    length_a = left_a + right_a
    length_b = left_b + right_b

    intersect = torch.min(left_a, left_b) + torch.min(right_a, right_b)
    intersect[intersect < 0] = 0
    union = length_a + length_b - intersect
    union[union <= 0] = eps

    iou = intersect / union
    return iou


def calc_loc_loss(pred_loc: torch.Tensor, test_loc: torch.Tensor, cls_label: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cls_label = cls_label.type(torch.bool)
    pred_loc = pred_loc[cls_label]
    test_loc = test_loc[cls_label]
    
    if pred_loc.shape[0] == 0:
        return torch.tensor(0.0, device=pred_loc.device)
    
    iou = iou_offset(pred_loc, test_loc)
    loss = -torch.log(iou + eps).mean()
    return loss


def calc_ctr_loss(pred, target, pos_mask):
    pos_mask = pos_mask.type(torch.bool)
    pred = pred[pos_mask]
    target = target[pos_mask]
    
    if pred.shape[0] == 0:
        return torch.tensor(0.0, device=pred.device)
    
    loss = F.binary_cross_entropy(pred, target)
    return loss


def one_hot_embedding(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    eye = torch.eye(num_classes, device=labels.device)
    return eye[labels]


def focal_loss(x: torch.Tensor, y: torch.Tensor, smooth_target: torch.Tensor = None, 
               alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    _, num_classes = x.shape
    t = one_hot_embedding(y, num_classes)
    
    # Use smooth target if provided
    if smooth_target is not None:
        t_smooth = torch.stack([1 - smooth_target, smooth_target], dim=-1)
    else:
        t_smooth = t
    
    p_t = x * t + (1 - x) * (1 - t)
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    
    fl = -alpha_t * (1 - p_t).pow(gamma) * (p_t + 1e-8).log()
    
    # Weight by smooth target
    if smooth_target is not None:
        fl = fl * t_smooth
    
    fl = fl.sum()
    return fl


def reconstruction_loss(reconstructed_feature, original_feature):
    """Reconstruction loss with normalization"""
    # Normalize features before computing loss
    recon_norm = F.normalize(reconstructed_feature, p=2, dim=-1)
    orig_norm = F.normalize(original_feature, p=2, dim=-1)
    loss = F.mse_loss(recon_norm, orig_norm)
    return loss


def diversity_loss(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Encourage diverse frame selection"""
    # Normalize features
    features_norm = F.normalize(features, p=2, dim=-1)
    # Compute similarity matrix
    sim_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))
    # Exclude diagonal
    mask = 1 - torch.eye(sim_matrix.shape[1], device=features.device)
    sim_matrix = sim_matrix * mask.unsqueeze(0)
    # Penalize high similarity
    loss = sim_matrix.mean()
    return loss


def temporal_smoothness_loss(pred_scores: torch.Tensor) -> torch.Tensor:
    """Encourage temporally smooth predictions"""
    if pred_scores.dim() == 1:
        pred_scores = pred_scores.unsqueeze(0)
    diff = pred_scores[:, 1:] - pred_scores[:, :-1]
    loss = (diff ** 2).mean()
    return loss
