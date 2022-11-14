#Author: Elin.Liu
# Date: 2022-11-14 14:04:32
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-14 14:04:32
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7) -> None:
        super(InceptionC, self).__init__()

        # 初始化第一个1*1的卷积层，输出特征通道数为192
        self.branch1x1 = nn.Conv2d(in_channels, 192, kernel_size=1)
        # 初始化7x7的通道
        c7 = channels_7x7
        # 初始化第二个1*1的卷积层，输出特征通道数为c7
        self.branch7x7_1 = nn.Conv2d(in_channels, c7, kernel_size=1)
        # 初始化第一个1x7的卷积层，输出特征通道数为c7，填充为3
        self.branch7x7_2 = nn.Conv2d(
            c7, c7, kernel_size=(1, 7), padding=(0, 3))
        # 初始化第二个7x1的卷积层，输出特征通道数为192，填充为3
        self.branch7x7_3 = nn.Conv2d(
            c7, 192, kernel_size=(7, 1), padding=(3, 0))
        # 初始化第三个1*1的卷积层，输出特征通道数为192
        self.branch7x7db1_1 = nn.Conv2d(in_channels, c7, kernel_size=1)
        # 初始化第三个7x1的卷积层，输出特征通道数为c7，填充为3
        self.branch7x7db1_2 = nn.Conv2d(
            c7, c7, kernel_size=(7, 1), padding=(3, 0))
        # 初始化第三个1x7的卷积层，输出特征通道数为c7，填充为3
        self.branch7x7db1_3 = nn.Conv2d(
            c7, c7, kernel_size=(1, 7), padding=(0, 3))
        # 初始化第三个7x1的卷积层，输出特征通道数为192，填充为3
        self.branch7x7db1_4 = nn.Conv2d(
            c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7db1_5 = nn.Conv2d(
            c7, 192, kernel_size=(1, 7), padding=(0, 3))
        # 初始化分支池化层，输出特征通道数为192
        self.branch_pool = nn.Conv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7db1 = self.branch7x7db1_1(x)
        branch7x7db1 = self.branch7x7db1_2(branch7x7db1)
        branch7x7db1 = self.branch7x7db1_3(branch7x7db1)
        branch7x7db1 = self.branch7x7db1_4(branch7x7db1)
        branch7x7db1 = self.branch7x7db1_5(branch7x7db1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7db1, branch_pool]
        return torch.cat(outputs, dim=1)
