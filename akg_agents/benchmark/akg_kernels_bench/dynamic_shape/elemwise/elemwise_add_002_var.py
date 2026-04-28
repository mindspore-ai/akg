import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, alpha=0.5):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, input_tensor, other):
        # torch.add(input, other, *, alpha=1, out=None)
        # Adds other to input tensor element-wise.
        # If alpha is specified, other is multiplied by alpha before addition.
        # This is an element-wise addition with alpha=0.5.
        # Element-wise addition is commonly used in neural networks for:
        # - Implementing residual connections
        # - Adding bias terms
        # - Combining feature maps
        return torch.add(input_tensor, other, alpha=self.alpha)


def get_inputs_dyn_list():
    # Case 1: 16-aligned batch, 16-aligned hidden
    # Shape (256, 4096) represents a batch of 256 samples with 4096 features each
    inp1_1 = torch.randn(256, 4096, dtype=torch.float32)
    inp2_1 = torch.randn(256, 4096, dtype=torch.float32)
    
    # Case 2: Non-16-aligned batch, 16-aligned hidden
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inp1_2 = torch.randn(125, 5120, dtype=torch.float32)
    inp2_2 = torch.randn(125, 5120, dtype=torch.float32)
    
    # Case 3: 16-aligned batch, non-16-aligned hidden
    # Shape (512, 6144) represents a batch of 512 samples with 6144 features each
    inp1_3 = torch.randn(512, 6144, dtype=torch.float32)
    inp2_3 = torch.randn(512, 6144, dtype=torch.float32)
    
    # Case 4: Large batch size
    # Shape (1024, 8192) represents a batch of 1024 samples with 8192 features each
    inp1_4 = torch.randn(1024, 8192, dtype=torch.float32)
    inp2_4 = torch.randn(1024, 8192, dtype=torch.float32)
    
    return [
        [inp1_1, inp2_1],
        [inp1_2, inp2_2],
        [inp1_3, inp2_3],
        [inp1_4, inp2_4]
    ]


def get_init_inputs():
    # Fixed parameters for add with alpha=0.5
    alpha = 0.5  # Multiplier for the second operand
    return [alpha]