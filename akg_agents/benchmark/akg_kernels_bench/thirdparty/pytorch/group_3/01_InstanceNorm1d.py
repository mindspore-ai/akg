import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Instance Normalization on 3D input (sequences).
    Normalizes each instance in each channel independently.
    Commonly used in style transfer and sequence modeling.
    """
    def __init__(self, num_features: int):
        """
        Initializes the InstanceNorm1d layer.

        Args:
            num_features (int): Number of features (channels) in the input tensor.
        """
        super(Model, self).__init__()
        self.inorm = nn.InstanceNorm1d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Instance Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, length).

        Returns:
            torch.Tensor: Output tensor with Instance Normalization applied, same shape as input.
        """
        return self.inorm(x)

batch_size = 32
num_features = 128
length = 4096

def get_inputs():
    x = torch.randn(batch_size, num_features, length)
    return [x]

def get_init_inputs():
    return [num_features]
