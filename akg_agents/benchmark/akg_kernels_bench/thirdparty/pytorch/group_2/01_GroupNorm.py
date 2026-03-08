import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Group Normalization.
    Divides channels into groups and normalizes within each group.
    """
    def __init__(self, num_groups: int, num_channels: int):
        """
        Initializes the GroupNorm layer.

        Args:
            num_groups (int): Number of groups to divide the channels into.
            num_channels (int): Number of channels in the input tensor.
        """
        super(Model, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Group Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Group Normalization applied, same shape as input.
        """
        return self.gn(x)

batch_size = 16
num_channels = 64
num_groups = 8
height = 256
width = 256

def get_inputs():
    x = torch.randn(batch_size, num_channels, height, width)
    return [x]

def get_init_inputs():
    return [num_groups, num_channels]
