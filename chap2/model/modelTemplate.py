import torch.nn as nn
# 创建一个模型类，继承自nn.Module


class Model(nn.Module):
    def __init__(self,kwargs):  # 定义类的初始化函数，...是用户的传入参数
        super(Model, self).__init__()
        ...  # 根据传入的参数来定义子模块

    def forward(self, x):
        # 定义前向计算传入的参数，...一般是张量或者其他的参数
        ret = ...  # 根据传入的张量和子模块计算返回张量
        return ret
