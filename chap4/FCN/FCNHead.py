#Author: Elin.Liu
# Date: 2022-11-19 17:39:16
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-19 17:39:16

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNHead(nn.Sequential()):
    def __init__(self, in_channels, channels) -> None:
        # 初始化输入通道数
        inter_channels = in_channels
        # 初始化层
        layers = [
            # 3x3卷积
            nn.Conv2d(in_channels, inter_channels,
                      kernel_size=3, padding=1, bias=False),
            # BN层
            nn.BatchNorm2d(inter_channels),
            # ReLU激活函数
            nn.ReLU(),
            # 设置dropout
            nn.Dropout(0.1),
            # 1x1卷积
            nn.Conv2d(inter_channels, channels, 1)
        ]
        super(FCNHead, self).__init__()
