import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Weight Normalization on a Linear layer.
    Weight normalization reparameterizes the weight vector w as:
        w = g * (v / ||v||)
    where g is a learnable scalar and v is the unnormalized weight vector.
    Based on torch.nn.utils.weight_norm (Salimans & Kingma, 2016).
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Initializes a weight-normalized Linear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
        """
        super(Model, self).__init__()
        self.linear = nn.utils.weight_norm(nn.Linear(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the weight-normalized linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        return self.linear(x)

batch_size = 64
in_features = 4096
out_features = 4096

def get_inputs():
    x = torch.randn(batch_size, in_features)
    return [x]

def get_init_inputs():
    return [in_features, out_features]
