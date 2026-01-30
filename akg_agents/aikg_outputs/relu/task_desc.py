import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x)


def get_inputs():
    # 使用合理的默认值：batch_size=16, feature_dim=1024
    return [torch.randn(16, 1024)]


def get_init_inputs():
    # ReLU算子不需要初始化参数
    return []