import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var, var_scale, indices):
        # torch.hstack(tensors, *, out=None)
        # Stack tensors in sequence horizontally (column wise).
        # This is equivalent to concatenation along the first axis for 1-D tensors,
        # and along the second axis for 2-D tensors.
        # Horizontal stacking is commonly used in neural networks for:
        # - Combining feature vectors
        # - Creating wider layers
        tensors = [var, var_scale, indices]
        return torch.hstack(tensors)


def get_inputs_dyn_list():
    # Small shape case
    tensor1_1 = torch.randn(128, 1024, dtype=torch.float32)
    tensor2_1 = torch.randn(128, 1024, dtype=torch.float32)
    tensor3_1 = torch.randn(128, 1024, dtype=torch.float32)

    # Middle shape case
    tensor1_2 = torch.randn(512, 2048, dtype=torch.float32)
    tensor2_2 = torch.randn(512, 2048, dtype=torch.float32)
    tensor3_2 = torch.randn(512, 2048, dtype=torch.float32)

    # Large shape case
    tensor1_3 = torch.randn(1024, 4096, dtype=torch.float32)
    tensor2_3 = torch.randn(1024, 4096, dtype=torch.float32)
    tensor3_3 = torch.randn(1024, 4096, dtype=torch.float32)

    # Noaligned shape case
    tensor1_4 = torch.randn(513, 3000, dtype=torch.float32)
    tensor2_4 = torch.randn(513, 3000, dtype=torch.float32)
    tensor3_4 = torch.randn(513, 3000, dtype=torch.float32)

    return [
        [tensor1_1, tensor2_1, tensor3_1],
        [tensor1_2, tensor2_2, tensor3_2],
        [tensor1_3, tensor2_3, tensor3_3],
        [tensor1_4, tensor2_4, tensor3_4]
    ]


def get_init_inputs():
    # No parameters needed for hstack
    # Extract params
    return []