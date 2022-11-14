#Author: Elin.Liu
# Date: 2022-11-12 14:10:41
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-12 14:10:41
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionB(nn.Module):
    def __init__(self, in_channels):
        # 继承来自inceptionB的父类方法
        super(InceptionB, self).__init__()
        # 初始化一个3*3，步长为2的卷积层，输出特征通道数为384
        self.branch3x3 = nn.Conv2d(in_channels, 384, kernel_size=3, stride=2)
        # 初始化第二个1*1的卷积层，输出特征通道数为64
        self.branch3x3db1_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        # 初始化第二个3*3的卷积层，输出特征通道数为96
        self.branch3x3db1_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        # 初始化第二个3*3的卷积层，输出特征通道数为96
        self.branch3x3db1_3 = nn.Conv2d(96, 96, kernel_size=3, stride=2)

    # 定义前向传播函数
    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3db1 = self.branch3x3db1_1(x)
        branch3x3db1 = self.branch3x3db1_2(branch3x3db1)
        branch3x3db1 = self.branch3x3db1_3(branch3x3db1)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3db1, branch_pool]
        return torch.cat(outputs, 1)
