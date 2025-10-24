import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise equal comparison with 1D horizontal broadcasting.
    Small scale: e1
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 1D equal comparison, horizontal broadcast
        return torch.eq(input1, input2).to(torch.float16)


def get_inputs():
    # Small scale: 16 ≈ e1

    input1 = torch.randn(16, dtype=torch.float16)
    input2 = torch.randn(1, dtype=torch.float16)
    return [input1, input2]


def get_init_inputs():
    return []
