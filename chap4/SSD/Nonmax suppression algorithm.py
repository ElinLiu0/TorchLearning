#Author: Elin.Liu
# Date: 2022-11-17 20:34:03
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-17 20:34:03
import torch.nn as nn
import torch
# 初始化非极大值抑制函数


def box_nms(bboxes, scores, threshold=0.5):
    # 获得所有的锚点框的x1,y1,x2,y2坐标点列表
    x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # 计算所有的锚点框的面积，即Iou的分母
    areas = (x2-x1+1)*(y2-y1+1)
    # 获得排序后的索引
    _, order = scores.sort(0, descending=True)

    # 初始化保留的锚点框索引列表
    keep = []
    # 循环遍历所有的锚点框
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        # 如果只有一个锚点框，则证明在获取锚点框时，与原始特征图像完全重合，因此不需要进行非极大值抑制
        if order.numel() == 1:
            break
        # 计算当前锚点框与其他锚点框的交集
        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        # 计算锚点选中图像区域的宽度和高度
        # 其中clamp函数的作用是将最小值和最大值限制为指定的范围
        w = (xx2-xx1+1).clamp(min=0)
        h = (yy2-yy1+1).clamp(min=0)
        # 计算交集的面积
        inter = w*h

        # 计算交集与并集的比值
        ovr = inter/(areas[i]+areas[order[1:]]-inter)
        # 保留交集与并集比值小于阈值的锚点框索引
        ids = (ovr <= threshold).nonzero().squeeze()

        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
