#Author: Elin.Liu
# Date: 2022-11-19 17:37:45
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-19 17:37:45
import torch
import torch.nn as nn
import torch.nn.functional as F
from FeatureExtractor import _segm_resnet
from FCNHead import FCNHead
from collections import OrderedDict
import torchvision.transforms as T


class _SimpleSegmentationModel(nn.Module):
    # 定义初始化函数
    def __init__(self, backbone, classifier, aux_classifier):
        super(_SimpleSegmentationModel, self).__init__()
        # 初始化backbone
        self.backbone = backbone
        # 初始化分类器
        self.classifier = classifier
        # 初始化辅助分类器
        self.aux_classifier = aux_classifier

    # 定义前向函数
    def forward(self, x):
        # 获取输入的尺寸
        input_size = x.shape[-2:]
        # 获取来自ResNet50中层的输出作为特征
        features = self.backbone(x)

        # 使用OrderedDict来保存来自ResNet50中层的输出
        result = OrderedDict()
        # 获得特征的输出
        x = features["out"]
        # 将特征的输出进行分类
        x = self.classifier(x)
        # 将分类后的结果进行双线性插值上采样
        x = F.interpolate(x, size=input_size,
                          mode="bilinear", align_corners=False)
        # 获取输出
        result["out"] = x

        # 检测是否设置aux_classifier
        if self.aux_classifier is not None:
            # 获取aux_classifier的输出
            x = features["aux"]
            # 将aux_classifier的输出进行分类
            x = self.aux_classifier(x)
            # 将分类后的结果进行双线性插值上采样
            x = F.interpolate(x, size=input_size,
                              mode="bilinear", align_corners=False)
            # 获取输出
            result["aux"] = x


class FCN(_SimpleSegmentationModel):
    pass


# 定义获取特征变化函数
def get_transform(train):
    # 设置基础图像尺寸
    base_size = 520
    # 设置遮罩图像尺寸
    crop_size = 480

    # 设置最小最大尺寸，确保两个值（即HW）比为0.5:2
    min_size = int(0.5 * base_size)
    max_size = int(2.0 if train else 1.0 * base_size)

    # 初始化TransForms列表
    transforms = []
    # 使用RandomResize()函数对图像进行随机尺寸缩放
    transforms.append(T.RandomResize(min_size, max_size))
    # 检测训练模式
    if train:
        # 使用RandomHorizontalFlip()函数对图像进行随机水平翻转
        transforms.append(T.RandomHorizontalFlip(0.5))
        # 使用RandomCrop()函数对图像进行随机裁剪
        transforms.append(T.RandomCrop(crop_size))
    # 使用ToTensor()函数将图像转换为Tensor
    transforms.append(T.ToTensor())
    # 使用Normalize()函数对图像进行归一化
    transforms.append(T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # 将transforms列表转换为Compose()函数
    transforms = T.Compose(transforms)
    # 返回transforms
    return transforms
