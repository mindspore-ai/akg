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


def get_inputs_dyn_list():
    # Case 1: Small (batch=256, hidden=512)
    input_tensor1 = torch.randint(0, 2, (256, 512), dtype=torch.bool)

    # Case 2: Middle (batch=1024, hidden=4096)
    input_tensor2 = torch.randint(0, 2, (1024, 4096), dtype=torch.bool)

    # Case 3: Large (batch=2048, hidden=4096)
    input_tensor3 = torch.randint(0, 2, (2048, 4096), dtype=torch.bool)

    # Case 4: Non-aligned (batch=768, hidden=2688)
    input_tensor4 = torch.randint(0, 2, (768, 2688), dtype=torch.bool)

    return [
        [input_tensor1],
        [input_tensor2],
        [input_tensor3],
        [input_tensor4]
    ]


def get_init_inputs():
    # No parameters needed for logical_not
    return []