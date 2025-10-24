import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise equality comparison with broadcast (2D).
    Medium scale: e5
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 2D equal comparison, second dimension broadcast
        return torch.eq(input1, input2)


def get_inputs():
    # Medium scale: 512 * 512 ≈ e5

    input1 = torch.randint(-128, 127, (512, 512), dtype=torch.int64)
    input2 = torch.randint(-128, 127, (1, 512), dtype=torch.int64)
    return [input1, input2]


def get_init_inputs():
    return []

