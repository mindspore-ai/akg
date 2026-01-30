import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # torch.lt(input, other, *, out=None)
        # Computes input < other elementwise.
        # This operation is commonly used in neural networks for:
        # - Implementing comparison operations
        # - Creating masks for conditional operations
        # - Implementing certain activation functions or gating mechanisms
        return torch.lt(input1, input2)


def get_inputs_dyn_list():
    # Case 1: Small (batch=256, hidden=512)
    input1_1 = torch.randn(256, 512, dtype=torch.float32)
    input1_2 = torch.randn(256, 512, dtype=torch.float32)

    # Case 2: Middle (batch=1024, hidden=4096)
    input2_1 = torch.randn(1024, 4096, dtype=torch.float32)
    input2_2 = torch.randn(1024, 4096, dtype=torch.float32)

    # Case 3: Large (batch=2048, hidden=4096)
    input3_1 = torch.randn(2048, 4096, dtype=torch.float32)
    input3_2 = torch.randn(2048, 4096, dtype=torch.float32)

    # Case 4: Non-aligned (batch=768, hidden=2688)
    input4_1 = torch.randn(768, 2688, dtype=torch.float32)
    input4_2 = torch.randn(768, 2688, dtype=torch.float32)

    return [
        [input1_1, input1_2],
        [input2_1, input2_2],
        [input3_1, input3_2],
        [input4_1, input4_2]
    ]


def get_init_inputs():
    # No parameters needed for less
    return []