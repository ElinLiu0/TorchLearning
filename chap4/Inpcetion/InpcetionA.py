#Author: Elin.Liu
# Date: 2022-11-12 13:43:36
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-11-12 13:43:36
import torch
import torch.nn as nn
import torch.nn.functional as F
class InceptionA(nn.Module):
    def __init__(self,in_channels,pool_features) -> None:
        super(InceptionA).__init__()
        self.branch1x1 = nn.Conv2d(in_channels,64,kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels,48,kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(48,64,kernel_size=1)

        self.branch3x3db1_1 = nn.Conv2d(in_channels,64,kernel_size=1)
        self.branch3x3db1_2 = nn.Conv2d(64,96,kernel_size=3,padding=1)
        self.branch3x3db1_3 = nn.Conv2d(96,96,kernel_size=3,padding=1)

        self.branch_pool = nn.Conv2d(in_channels,pool_features,kernel_size=1)
    def forward(self,x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3db1 = self.branch3x3db1_1(x)
        branch3x3db1 = self.branch3x3db1_2(branch3x3db1)
        branch3x3db1 = self.branch3x3db1_3(branch3x3db1)

        branch_pool = F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1,branch5x5,branch3x3db1,branch_pool]

        return torch.cat(outputs,1)
