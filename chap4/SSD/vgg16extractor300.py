#Author: Elin.Liu
# Date: 2022-11-14 20:49:19
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-14 20:49:19
import torch.nn as nn
import torch.nn.functional as F
import torch
from vgg16 import VGG16
from L2Norm import L2Norm
# 定义VGG16的特征提取网络


class VGG16Extractor300(nn.Module):
    def __init__(self) -> None:
        super(VGG16Extractor300, self).__init__()
        # 初始化特征为来自vgg16的网络的输出
        self.features = VGG16()
        # 初始化L2正则化层
        self.L2Norm = L2Norm(512, 20)
        # 初始化卷积层
        self.conv5_1 = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, dilation=1)
        # 叠加两层卷积层
        self.conv5_2 = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(
            512, 512, kernel_size=3, padding=1, dilation=1)

        # 叠加第四层卷积层，输出通道数为1024，输入通道数为512，卷积核大小为3，填充为6，dilation为6
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        # 叠加第五层卷积层，输出通道数为1024，输入通道数为1024，卷积核大小为1，
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        # 叠加第六层卷积层，输出通道数为256，输入通道数为1024，卷积核大小为1，
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        # 叠加第七层卷积层，输出通道数为512，输入通道数为256，卷积核大小为3，填充为1
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        # 叠加第八层卷积层，输出通道数为128，输入通道数为512，卷积核大小为1，
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        # 叠加第九层卷积层，输出通道数为256，输入通道数为128，卷积核大小为3，填充为1
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # 叠加第十层卷积层，输出通道数为128，输入通道数为256，卷积核大小为1，
        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        # 叠加第十一层卷积层，输出通道数为256，输入通道数为128，卷积核大小为3，填充为1
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)
        # 叠加第十二层卷积层，输出通道数为128，输入通道数为256，卷积核大小为1，
        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        # 叠加第十三层卷积层，输出通道数为256，输入通道数为128，卷积核大小为3，填充为1
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

    def forward(self, x):
        hs = []
        # 获取vgg16的特征
        h = self.features(x)
        # 获取第一个卷积层的特征
        hs.append(self.L2Norm(h))
        # 获取第二个卷积层的特征
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))

        h = F.max_pool2d(h, kernel_size=3, stride=1, padding=1, ceil_mode=True)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)

        return hs
