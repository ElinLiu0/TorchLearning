#Author: Elin.Liu
# Date: 2022-11-17 18:48:22
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-17 18:48:22
import torch.nn as nn
import torch
import torch.nn.functional as F

# 初始化SSDLoss损失函数类


class SSDLoss(nn.Module):
    def __init__(self, num_classes) -> None:
        super(SSDLoss, self).__init__()
        # 初始化类别数量
        self.num_classes = num_classes

    # 定义困难负样本挖掘函数
    def _hard_negative_mining(self, cls_loss, pos):

        # 计算cls_loss中正样本的数量
        cls_loss = cls_loss * (pos.float() - 1)

        # 损失函数排序
        _, idx = cls_loss.sort(1)
        # 对排序后的损失函数进行降序排列
        _, rank = idx.sort(1)
        # 计算负样本的数量
        num_pos = 3 * pos.sum(1)
        # 计算负样本的数量
        neg = rank < num_pos[:, None]
        return neg

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):

        # 获取正样本的数量
        pos = cls_targets > 0

        # 计算迷你批次大小
        batch_size = pos.size(0)

        # 计算正样本的数量
        num_pos = pos.sum().item()

        # 计算掩码，用于过滤掉负样本
        mask = pos.unsqueeze(2).expand_as(loc_preds)

        # 计算锚点框的位置损失
        loc_loss = F.smooth_l1_loss(
            loc_preds[mask], loc_targets[mask], reduction='sum')
        # 计算锚点框选定的图像类别损失
        cls_loss = F.cross_entropy(
            cls_preds.view(-1, self.num_classes), cls_targets.view(-1), reduction='none')

        # 对cls_loss创建view
        cls_loss = cls_loss.view(batch_size, -1)

        # 设置忽略的损失函数为0
        cls_loss[cls_targets < 0] = 0
        # 获取困难负样本
        neg = self._hard_negative_mining(cls_loss, pos)
        # 计算负样本的损失函数
        cls_loss = cls_loss[pos | neg].sum()

        # 计算平均损失函数
        loss = (loc_loss + cls_loss) / num_pos

        return loss
