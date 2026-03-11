import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Layer Normalization over multiple dimensions.
    Unlike standard LayerNorm which normalizes over the last dim only,
    this normalizes over the last two dimensions (seq_len, hidden_size),
    computing mean and variance across a much larger reduction domain.
    """
    def __init__(self, normalized_shape: tuple):
        """
        Initializes the LayerNorm layer with a multi-dimensional normalized_shape.

        Args:
            normalized_shape (tuple): Shape of the dimensions to normalize over.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization over the last two dimensions.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)

batch_size = 32
channels = 128
height = 64
width = 64

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [(channels, height, width)]
