#Author: Elin.Liu
# Date: 2022-11-19 21:49:10
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-19 21:49:10
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms as T
import torch

# 初始化UNet类


class UNet(nn.Module):
    # 初始化函数，设置输入通道数为3、输出通道数为1、基础特征通道数为64
    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(UNet, self).__init__()
        
        # 初始化特征通道数
        features = init_features
        # 定义编码器
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
    
        # 初始化第一个池化层，大小为2x2，步长为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义第二个编码器
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        # 初始化第二个池化层，大小为2x2，步长为2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义第三个编码器
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        # 初始化第三个池化层，大小为2x2，步长为2
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义第四个编码器
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        # 初始化第四个池化层，大小为2x2，步长为2
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 设置bottleneck层，输入通道数为features * 8，输出通道数为features * 16
        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        # 初始化第一个上采样层，大小为2x2，步长为2
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )

        # 定义第一个解码器
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")

        # 初始化第二个上采样层，大小为2x2，步长为2
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )

        # 定义第二个解码器
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")

        # 初始化第三个上采样层，大小为2x2，步长为2
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )

        # 定义第三个解码器

        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")

        # 初始化第四个上采样层，大小为2x2，步长为2
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )

        # 定义第四个解码器
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        # 定义输出层
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def foward(self,x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1", nn.Conv2d(in_channels, features, kernel_size=3, padding=1)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2", nn.Conv2d(features, features, kernel_size=3, padding=1)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
