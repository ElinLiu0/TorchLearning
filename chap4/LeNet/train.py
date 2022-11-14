import torch
import torch.nn as nn
from model import LeNet
from tqdm import trange
from data import data_train_loader
# 初始化LeNet模型
net = LeNet()
# 将网络调整为训练模式
net.train()
# 设置学习率
Learning_rate = 0.001
# 设置损失函数
criterion = nn.CrossEntropyLoss()
# 设置优化器，本次实验中使用SGD优化器，设置动量为0.9，权重衰减为0.0001
optimizer = torch.optim.SGD(params=net.parameters(
), lr=Learning_rate, momentum=0.9, weight_decay=5e-4)

# 初始化训练集和测试集的损失值
train_loss = 0
test_loss = 0
total = 0

# 开始训练
for idx, (input, target) in enumerate(data_train_loader):
    # 将当前迭代的优化器梯度清零
    optimizer.zero_grad()
    # 获得模型的输出值
    output = net(input)
    # 计算损失值
    loss = criterion(output, target)
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()

    # 计算训练集的损失值
    train_loss += loss.item()
    # 获取预测值
    _, predicted = output.max(1)
    # 计算预测正确的个数
    total += target.size(0)
    # 计算训练集的准确率
    corrcet = predicted.eq(target).sum().item()

    print("Epoch: %d, Iter: %d, Loss: %.3f, Acc: %.3f" %
          (idx, idx, train_loss/(idx+1), corrcet/total))
