import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, alpha=1.0):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, input1, input2):
        # torch.sub(input, other, *, alpha=1, out=None)
        # Subtracts other from input tensor element-wise.
        # If alpha is specified, other is multiplied by alpha before subtraction.
        # This operation is commonly used in neural networks for:
        # - Computing residuals in ResNet architectures
        # - Implementing certain loss functions
        # - Mathematical transformations in specialized layers
        return torch.sub(input1, input2, alpha=self.alpha)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input1 = torch.randn(1024, 4096, dtype=torch.float32)
    input2 = torch.randn(1024, 4096, dtype=torch.float32)
    return [input1, input2]


def get_init_inputs():
    # Parameters for sub
    alpha = 2.0
    return [alpha]