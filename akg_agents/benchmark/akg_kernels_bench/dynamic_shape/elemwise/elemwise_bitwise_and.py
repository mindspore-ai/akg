import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # torch.bitwise_and(input, other, *, out=None)
        # Computes the element-wise bitwise AND of the given input tensors.
        # Zeros are treated as False and nonzeros are treated as True.
        # This operation is commonly used in neural networks for:
        # - Implementing bit manipulation operations
        # - Creating bit masks
        # - Low-level data processing
        return torch.bitwise_and(input1, input2)


def get_inputs_dyn_list():
    # Case 1: Small (batch=256, hidden=512)
    input1_1 = torch.randint(0, 256, (256, 512), dtype=torch.int32)
    input1_2 = torch.randint(0, 256, (256, 512), dtype=torch.int32)

    # Case 2: Middle (batch=1024, hidden=4096)
    input2_1 = torch.randint(0, 256, (1024, 4096), dtype=torch.int32)
    input2_2 = torch.randint(0, 256, (1024, 4096), dtype=torch.int32)

    # Case 3: Large (batch=2048, hidden=4096)
    input3_1 = torch.randint(0, 256, (2048, 4096), dtype=torch.int32)
    input3_2 = torch.randint(0, 256, (2048, 4096), dtype=torch.int32)

    # Case 4: Non-aligned (batch=768, hidden=2688)
    input4_1 = torch.randint(0, 256, (768, 2688), dtype=torch.int32)
    input4_2 = torch.randint(0, 256, (768, 2688), dtype=torch.int32)

    return [
        [input1_1, input1_2],
        [input2_1, input2_2],
        [input3_1, input3_2],
        [input4_1, input4_2]
    ]


def get_init_inputs():
    # No parameters needed for bitwise_and
    return []