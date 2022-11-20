#Author: Elin.Liu
# Date: 2022-11-16 17:45:49
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-16 17:45:49
import torch.nn as nn
import torch.nn.functional as F
import torch
from vgg16extractor300 import VGG16Extractor300
# 初始化SSD300目标检测模型


class SSD300(nn.Module):
    # 初始化步长列表
    strides = (8, 16, 32, 64, 100, 300)
    # 初始化锚点框大小列表
    box_sizes = (30, 60, 111, 162, 213, 264, 315)
    # 初始化长宽比列表
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    # 初始化特征图大小列表
    feature_maps = (38, 19, 10, 5, 3, 1)

    def __init__(self, num_classes):
        super(SSD300, self).__init__()
        # 初始化类别
        self.num_classes = num_classes
        # 初始化锚点框数量
        self.num_anchors = (4, 6, 6, 6, 4, 4)
        # 初始化输入通道数
        self.in_channels = (512, 1024, 512, 256, 256, 256)

        # 初始化特征提取器
        self.extractor = VGG16Extractor300()
        # 初始化预测器
        self.loc_layer = nn.ModuleList()
        # 初始化分类器
        self.cls_layer = nn.ModuleList()

        # 初始化预测器和分类器
        for i in range(len(self.in_channels)):
            # 添加预测器
            self.loc_layer += [nn.Conv2d(self.in_channels[i],
                                         self.num_anchors[i] * 4, kernel_size=3, padding=1)]
            # 添加分类器
            self.cls_layer += [nn.Conv2d(self.in_channels[i],
                                         self.num_anchors[i] * self.num_classes, kernel_size=3, padding=1)]

    def forward(self, x):
        loc_preds = []
        cls_preds = []

        # 获取特征提取器的输出
        xs = self.extractor(x)
        # 遍历特征提取器的输出
        for i, x in enumerate(xs):
            loc_pred = self.loc_layer[i](x)
            # 使用permute函数将输出的维度变换为(batch_size, num_anchors, H, W, 4)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            # 使用view函数将输出的维度变换为(batch_size, -1, 4)
            loc_pred = loc_pred.view(loc_pred.size(0), -1, 4)
            # 将预测结果添加到loc_preds列表中
            loc_preds.append(loc_pred)

            cls_pred = self.cls_layer[i](x)
            # 使用permute函数将输出的维度变换为(batch_size, num_anchors, H, W, num_classes)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            # 使用view函数将输出的维度变换为(batch_size, -1, num_classes)
            cls_pred = cls_pred.view(cls_pred.size(0), -1, self.num_classes)
            # 将预测结果添加到cls_preds列表中
            cls_preds.append(cls_pred)

        # 将loc_preds和cls_preds列表中的预测结果进行拼接
        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)

        return loc_preds, cls_preds
