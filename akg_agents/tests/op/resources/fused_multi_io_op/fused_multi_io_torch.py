import torch
import torch.nn as nn


class Model(nn.Module):
    """多输入多输出 + in-place 操作的测试算子。

    forward 中对 x 做 in-place add_(bias)，验证参考数据保存时
    能正确 clone 原始输入，避免 in-place 污染。
    """

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, y, bias):
        # in-place：修改 x（如果保存 inputs 不在 forward 前 clone，会被污染）
        x.add_(bias)

        # output 1: simplified layernorm on modified x
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        out_norm = (x - mean) / torch.sqrt(var + self.eps) * self.weight

        # output 2: gated product
        out_gated = x * torch.sigmoid(y)

        return out_norm, out_gated


batch_size = 4
hidden_size = 32


def get_inputs():
    x = torch.randn(batch_size, hidden_size)
    y = torch.randn(batch_size, hidden_size)
    bias = torch.randn(hidden_size)
    return [x, y, bias]


def get_init_inputs():
    return [hidden_size]
