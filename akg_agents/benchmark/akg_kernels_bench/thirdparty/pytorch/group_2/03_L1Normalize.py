import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that performs L1 normalization using F.normalize.
    Scales input so that L1 norm along the specified dimension equals 1.
    """
    def __init__(self):
        """
        Initializes the L1Normalize layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies L1 normalization to the input tensor along dimension 1.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: L1-normalized output tensor, same shape as input.
        """
        return F.normalize(x, p=1, dim=1)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
