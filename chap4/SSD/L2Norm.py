#Author: Elin.Liu
# Date: 2022-11-14 20:59:18
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-14 20:59:18

import torch.nn as nn
import torch.nn.functional as F
import torch


class L2Norm(nn.Module):

    def __init__(self, in_features, scale) -> None:
        super(L2Norm, self).__init__()

        # 初始化L2的权重
        self.weight = nn.Parameter(torch.Tensor(in_features))
        # 重置权重
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        # 以scale为标准，初始化为权重，创建连续张量
        nn.init.constant_(self.weight, scale)

    def forward(self, x):
        # 计算x的L2范数
        x = F.normalize(x, p=2, dim=1)
        # 计算权重
        out = self.weight.unsqueeze(0).unsqueeze(
            2).unsqueeze(3).expand_as(x) * x
        return out
