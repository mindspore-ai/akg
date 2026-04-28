import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Batch Normalization on 4D input (images).
    Maintains running mean and variance for inference, uses batch statistics during training.
    Includes learnable affine parameters (weight and bias).
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm2d layer.

        Args:
            num_features (int): Number of features (channels) in the input tensor.
        """
        super(Model, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, height, width).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)

batch_size = 16
num_features = 64
height = 256
width = 256

def get_inputs():
    x = torch.randn(batch_size, num_features, height, width)
    return [x]

def get_init_inputs():
    return [num_features]
