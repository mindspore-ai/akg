import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.logical_not(input, *, out=None)
        # Computes the element-wise logical NOT of the given input tensor.
        # Zeros are treated as False and nonzeros are treated as True.
        # This operation is commonly used in neural networks for:
        # - Inverting boolean masks
        # - Implementing negation of conditions
        # - Creating complementary masks
        return torch.logical_not(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randint(0, 2, (1024, 4096), dtype=torch.bool)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for logical_not
    return []