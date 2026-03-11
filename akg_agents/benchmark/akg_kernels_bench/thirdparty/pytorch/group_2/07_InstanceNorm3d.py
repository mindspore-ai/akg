import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Instance Normalization on 5D input (volumetric/video data).
    Normalizes each instance in each channel independently across the spatial dimensions.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm3d layer.

        Args:
            num_features (int): Number of features (channels) in the input tensor.
        """
        super(Model, self).__init__()
        self.inorm = nn.InstanceNorm3d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the 5D input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, depth, height, width).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.inorm(x)

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
