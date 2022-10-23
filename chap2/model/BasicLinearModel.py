import torch
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
class LinearModel(nn.Module):
    # 初始化类的参数
    def __init__(self, ndim):
        super(LinearModel, self).__init__()
        # 定义一个线性层，输入维度为ndim，输出维度为1
        self.ndim = ndim
        # 初始化权重和偏置，这里因为我们是以参数的形式传入，所以我们需要使用nn.Parameter来定义
        self.weight = nn.Parameter(torch.randn(ndim, 1, device='cuda'))
        self.bias = nn.Parameter(torch.randn(1, device='cuda'))

    # 定义前向计算函数
    def forward(self, x):
        # y = Wx + b
        return x.mm(self.weight) + self.bias


# 初始化模型，定义线性模型的维度为5
lm = LinearModel(5)
# 初始化随机输入为一个5维的向量
x = torch.randn(4, 5, device='cuda')
print(lm(x))
# 生成模型的计算图
g = make_dot(lm(x), params=dict(lm.named_parameters()))
# 保存模型的计算图
g.view()