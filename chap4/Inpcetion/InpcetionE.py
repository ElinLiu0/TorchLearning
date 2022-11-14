#Author: Elin.Liu
# Date: 2022-11-14 14:30:03
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-14 14:30:03
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionE(nn.Module):
    def __init__(self, in_channels) -> None:
        super(InceptionE, self).__init__()

        # 初始化第一个1*1的卷积层，输出特征通道数为320
        self.branch1x1 = nn.Conv2d(in_channels, 320, kernel_size=1)

        # 初始化第二个1*1的卷积层，输入特征通道数为in_channels，输出特征通道数为384
        self.branch3x3_1 = nn.Conv2d(in_channels, 384, kernel_size=1)

        # 初始化第一个1*3的卷积层，输入特征通道数为384，输出特征通道数为384，填充为1
        self.branch3x3_2a = nn.Conv2d(
            384, 384, kernel_size=(1, 3), padding=(0, 1))

        