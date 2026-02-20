# modules/siou.py
import torch
import torch.nn as nn
import math

def siou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    SIoU: https://arxiv.org/abs/2205.12740
    pred_boxes, target_boxes: [N,4] in xyxy 或 xywh 统一格式
    返回：每个框的 SIoU 相似度
    """
    # 以 xywh 格式计算
    px, py, pw, ph = pred_boxes.unbind(-1)
    tx, ty, tw, th = target_boxes.unbind(-1)

    # 中心距离项
    dx = tx - px
    dy = ty - py
    rho2 = dx.pow(2) + dy.pow(2)
    # 对角线长度
    cw = pw / 2 + tw / 2
    ch = ph / 2 + th / 2
    # 角度转向
    atan_dx = torch.atan(dx / (dy + eps))
    atan_dy = torch.atan(dy / (dx + eps))
    v = 4 / (math.pi ** 2) * (atan_dx - atan_dy).pow(2)
    with torch.no_grad():
        alpha = v / (1 - v + eps)

    # IoU 基础项
    inter_w = (pw / 2 + tw / 2 - torch.abs(dx)).clamp(min=0)
    inter_h = (ph / 2 + th / 2 - torch.abs(dy)).clamp(min=0)
    inter = inter_w * inter_h
    union = pw * ph + tw * th - inter + eps
    base_iou = inter / union

    ciou_term = base_iou - (rho2 / (cw.pow(2) + ch.pow(2) + eps) + alpha * v)
    return ciou_term.clamp(min=-1.0, max=1.0)
