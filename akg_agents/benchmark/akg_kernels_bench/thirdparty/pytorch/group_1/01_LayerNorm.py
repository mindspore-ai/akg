import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Layer Normalization over the last dimension.
    """
    def __init__(self, normalized_shape: int):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (int): Size of the last dimension to normalize over.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        return self.ln(x)

batch_size = 32
seq_len = 512
hidden_size = 768

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [hidden_size]
