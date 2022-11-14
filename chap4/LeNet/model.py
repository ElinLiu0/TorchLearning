import torch
import torch.nn as nn

# 初始化LeNet模型


class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)  # 第一个卷积层，输入通道为1，输出特征通道为6，卷积核为3
        self.pool1 = nn.MaxPool2d(2, 2)  # 第一个池化层，池化核为2，步长为2
        self.conv2 = nn.Conv2d(6, 16, 3)  # 第二个卷积层，输入通道为6，输出特征通道为16，卷积核为3
        self.pool2 = nn.MaxPool2d(2, 2)  # 第二个池化层，池化核为2，步长为2
        self.fc1 = nn.Linear(16*6*6, 120)  # 第一个全连接层，输入为16*5*5，输出为120
        self.fc2 = nn.Linear(120, 84)  # 第二个全连接层，输入为120，输出为84
        self.fc3 = nn.Linear(84, 10)  # 最后将输出的84维特征转换为10维，对应10个类别

    def forward(self, x):
        # 使用ReLU作为激活函数初始化第一层卷积，并对应进行第一个次池化
        x = self.pool1(torch.relu(self.conv1(x)))
        # 使用ReLU作为激活函数初始化第二层卷积，并进行第二次池化
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 将x的张量进行展开
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
