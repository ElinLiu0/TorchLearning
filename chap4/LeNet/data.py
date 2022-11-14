# 导入MNIST数据集
from torchvision.datasets import MNIST
# 导入transformer变换类
from torchvision.transforms import transforms
# 导入DataLoader
from torch.utils.data import DataLoader

# 定义训练集和测试集数据
data_train = MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
]))

data_test = MNIST('./data', train=False, download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()
                  ]))
# 使用DataLoader初始化训练集和测试集数据
data_train_loader = DataLoader(
    dataset=data_train,
    batch_size=256,
    shuffle=True,
    num_workers=8
)
data_test_loader = DataLoader(
    dataset=data_test,
    batch_size=1024,
    num_workers=8
)
