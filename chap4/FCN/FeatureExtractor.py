#Author: Elin.Liu
# Date: 2022-11-17 21:03:42
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-17 21:03:42
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models._utils import IntermediateLayerGetter

resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)


def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True):

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone, replaced_stride_with_dilation=[False, True, True])

    return_layers = {'layer4': 'out'}

    if aux:
        return_layers['layer3'] = 'aux'
    # 获取来自ResNet的模型骨架
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None

    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)
    # 初始化模型字典
    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN)
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model
