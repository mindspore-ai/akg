import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Local Response Normalization (LRN).
    Normalizes over local input regions across channels using a sliding window.
    Originally proposed in AlexNet.
    """
    def __init__(self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0):
        """
        Initializes the LocalResponseNorm layer.

        Args:
            size (int): Amount of neighbouring channels used for normalization.
            alpha (float): Multiplicative factor. Default: 1e-4.
            beta (float): Exponent. Default: 0.75.
            k (float): Additive factor. Default: 1.0.
        """
        super(Model, self).__init__()
        self.lrn = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Local Response Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with LRN applied, same shape as input.
        """
        return self.lrn(x)

batch_size = 32
channels = 128
height = 64
width = 64
size = 5

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [size]
