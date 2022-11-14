#Author: Elin.Liu
# Date: 2022-11-14 14:12:36
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-14 14:12:36
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionC(nn.Module):
    def __init__(self, in_channels) -> None:
        super(InceptionC, self).__init__()

        # 初始化第一个1*1的卷积层，输出特征通道数为192
        self.branch1x1 = nn.Conv2d(in_channels, 192, kernel_size=1)

        # 初始化一个3*3的卷积层，输入特征通道数为192，输出特征通道数为320
        self.branch3x3_1 = nn.Conv2d(192, 320, kernel_size=3, stride=2)

        # 初始化第二个1*1的卷积层，输入特征通道数为in_channels，输出特征通道数为192
        self.branch3x3_2 = nn.Conv2d(in_channels, 192, kernel_size=1)

        # 初始化第一个1*7的卷积层，输入特征通道数为192，输出特征通道数为192，填充为3
        self.branch3x3_3 = nn.Conv2d(
            192, 192, kernel_size=(1, 7), padding=(0, 3))
        # 初始化第二个7*1的卷积层，输入特征通道数为192，输出特征通道数为192，填充为3
        self.branch3x3_4 = nn.Conv2d(
            192, 192, kernel_size=(7, 1), padding=(3, 0))

    def forward(self, x):
        # 第一个分支
        branch1x1 = self.branch1x1(x)

        # 第二个分支
        branch3x3 = F.relu(self.branch3x3_1(branch1x1))

        # 第三个分支
        branch3x3stack = F.relu(self.branch3x3_2(x))
        branch3x3stack = F.relu(self.branch3x3_3(branch3x3stack))
        branch3x3stack = F.relu(self.branch3x3_4(branch3x3stack))

        # 第四个分支
        branchpool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch1x1, branch3x3, branch3x3stack, branchpool]
        return torch.cat(outputs, dim=1)
