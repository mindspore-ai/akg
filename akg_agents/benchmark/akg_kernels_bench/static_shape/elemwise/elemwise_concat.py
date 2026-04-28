import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Slice and concat operation: 对多个输入张量进行切片后在指定维度上拼接
    输入7个张量，对最后一维分别切片[128, 32, 48, 48, 48, 48, 48]后在dim=2上拼接
    """
    def __init__(self, dim=2):
        super(Model, self).__init__()
        self.dim = dim
        # 每个输入张量的切片大小
        self.slice_sizes = (128, 32, 48, 48, 48, 48, 48)

    def forward(self, *xs):
        # 对每个输入张量进行切片
        # x[..., :sz] 表示在最后一维取前sz个元素
        # 输出形状: (128, 50, 400) where 400 = 128+32+48+48+48+48+48
        slices = [x[..., :sz] for x, sz in zip(xs, self.slice_sizes)]
        
        # 在指定维度上拼接所有切片
        output = torch.cat(slices, dim=self.dim)
        
        return output


def get_inputs():
    # 7个输入张量，每个形状为 (128, 50, 128)
    # 使用float16类型生成随机数据
    x1 = torch.randn(128, 50, 128, dtype=torch.float16)
    x2 = torch.randn(128, 50, 128, dtype=torch.float16)
    x3 = torch.randn(128, 50, 128, dtype=torch.float16)
    x4 = torch.randn(128, 50, 128, dtype=torch.float16)
    x5 = torch.randn(128, 50, 128, dtype=torch.float16)
    x6 = torch.randn(128, 50, 128, dtype=torch.float16)
    x7 = torch.randn(128, 50, 128, dtype=torch.float16)
    
    return [x1, x2, x3, x4, x5, x6, x7]


def get_init_inputs():
    # 在dim=2维度上进行拼接
    dim = 2
    return [dim]