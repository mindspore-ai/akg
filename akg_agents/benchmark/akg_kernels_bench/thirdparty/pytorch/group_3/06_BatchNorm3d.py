import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Batch Normalization on 5D input (volumetric/video data).
    Maintains running mean and variance, with learnable affine parameters.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm3d layer.

        Args:
            num_features (int): Number of features (channels) in the input tensor.
        """
        super(Model, self).__init__()
        self.bn = nn.BatchNorm3d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the 5D input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)

batch_size = 4
num_features = 64
depth = 8
height = 32
width = 32

def get_inputs():
    x = torch.randn(batch_size, num_features, depth, height, width)
    return [x]

def get_init_inputs():
    return [num_features]
