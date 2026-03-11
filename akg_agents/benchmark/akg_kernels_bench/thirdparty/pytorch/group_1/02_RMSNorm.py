import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    Normalizes by the root mean square of the input, without subtracting the mean.
    Used in Transformer variants like LLaMA.
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            hidden_size (int): Size of the last dimension to normalize over.
            eps (float, optional): A small value added to the denominator for numerical stability.
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)

batch_size = 32
seq_len = 512
hidden_size = 768

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    return [x]

def get_init_inputs():
    return [hidden_size]
