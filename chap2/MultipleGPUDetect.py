import torch
a = torch.cuda.device_count()
if a < 2:
    print("醒醒吧穷逼，你只有一张显卡")
