import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Batch Normalization on 2D or 3D input.
    Accepts both (N, C) and (N, C, L) shaped tensors.
    Maintains running statistics and learnable affine parameters.
    """
    def __init__(self, num_features: int):
        """
        Initializes the BatchNorm1d layer.

        Args:
            num_features (int): Number of features (channels) in the input tensor.
        """
        super(Model, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Batch Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, seq_len).

        Returns:
            torch.Tensor: Output tensor with Batch Normalization applied, same shape as input.
        """
        return self.bn(x)

batch_size = 32
num_features = 256
seq_len = 4096

def get_inputs():
    x = torch.randn(batch_size, num_features, seq_len)
    return [x]

def get_init_inputs():
    return [num_features]
