import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, alpha=1.0):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, input_tensor, other):
        # torch.add(input, other, *, alpha=1, out=None)
        # Adds other to input tensor element-wise.
        # If alpha is specified, other is multiplied by alpha before addition.
        # This operation is commonly used in neural networks for:
        # - Implementing residual connections
        # - Adding bias terms
        # - Combining feature maps
        return torch.add(input_tensor, other, alpha=self.alpha)


def get_inputs_dyn_list():
    # Case 1: Small/Medium shapes
    # Shape (256, 1024) represents a batch of 256 samples with 1024 features each
    inp1_1 = torch.randn(256, 1024, dtype=torch.float32)
    inp2_1 = torch.randn(256, 1024, dtype=torch.float32)

    # Case 2: Standard large model shapes
    # Shape (1024, 4096) represents a batch of 1024 samples with 4096 features each
    inp1_2 = torch.randn(1024, 4096, dtype=torch.float32)
    inp2_2 = torch.randn(1024, 4096, dtype=torch.float32)

    # Case 3: Large shapes
    # Shape (2048, 8192) represents a batch of 2048 samples with 8192 features each
    inp1_3 = torch.randn(2048, 8192, dtype=torch.float32)
    inp2_3 = torch.randn(2048, 8192, dtype=torch.float32)

    # Case 4: Non-16-aligned shapes
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inp1_4 = torch.randn(125, 5120, dtype=torch.float32)
    inp2_4 = torch.randn(125, 5120, dtype=torch.float32)

    return [
        [inp1_1, inp2_1],
        [inp1_2, inp2_2],
        [inp1_3, inp2_3],
        [inp1_4, inp2_4]
    ]


def get_init_inputs():
    # Fixed parameters for add with alpha=1.0
    alpha = 1.0  # Multiplier for the second operand
    return [alpha]
