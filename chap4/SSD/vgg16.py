#Author: Elin.Liu
# Date: 2022-11-14 20:45:40
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-14 20:45:40
import torch.nn as nn
import torch.nn.functional as F
import torch

# 定义vgg16网络


class VGG16(nn.Module):
    def __init__(self) -> None:
        super(VGG16, self).__init__()
        self.layers = self._make_layers()

    def forward(self, x):
        out = self.layers(x)
        return out

    def _make_layers(self):
        # 初始化VGG16的配置文件
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
               'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        layers = []
        # 初始化输入通道数为3
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,
                                        stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # 返回vgg16的网络结构
        return nn.Sequential(*layers)
        
