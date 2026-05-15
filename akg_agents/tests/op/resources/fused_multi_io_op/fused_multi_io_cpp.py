import torch
import torch.nn as nn


class ModelNew(nn.Module):
    """与 Model 等价的纯 torch 实现（无 in-place 操作）。

    使用 out-of-place 的 x + bias 替代 x.add_(bias)，
    功能和数值结果完全一致。
    """

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, y, bias):
        x = x + bias

        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        out_norm = (x - mean) / torch.sqrt(var + self.eps) * self.weight

        out_gated = x * torch.sigmoid(y)

        return out_norm, out_gated
