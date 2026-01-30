import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, alpha=1.0):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, input_tensor, other):
        # torch.sub(input, other, *, alpha=1, out=None)
        # Subtracts other from input tensor element-wise.
        # If alpha is specified, other is multiplied by alpha before subtraction.
        # This operation is commonly used in neural networks for:
        # - Computing residuals in ResNet architectures
        # - Implementing certain loss functions
        # - Mathematical transformations in specialized layers
        return torch.sub(input_tensor, other, alpha=self.alpha)


def get_inputs_dyn_list():
    # Element-wise subtraction variation cases with both aligned and non-aligned shapes
    
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
    # Fixed parameters for sub
    alpha = 2.0  # Multiplier for the second operand
    return [alpha]