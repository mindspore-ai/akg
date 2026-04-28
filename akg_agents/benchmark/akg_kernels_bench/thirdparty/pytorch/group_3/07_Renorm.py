import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Renormalization (torch.renorm).
    Clips slices of the tensor along the given dimension so that
    the Lp norm of each slice does not exceed maxnorm.
    """
    def __init__(self, p: float, dim: int, maxnorm: float):
        """
        Initializes the Renorm layer.

        Args:
            p (float): The exponent value in the norm formulation.
            dim (int): The dimension to slice over.
            maxnorm (float): The maximum norm value.
        """
        super(Model, self).__init__()
        self.p = p
        self.dim = dim
        self.maxnorm = maxnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies renormalization to the input tensor along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features).

        Returns:
            torch.Tensor: Output tensor with slices renormalized, same shape as input.
        """
        return torch.renorm(x, self.p, self.dim, self.maxnorm)

batch_size = 256
features = 16384
p = 2.0
dim = 0
maxnorm = 1.0

def get_inputs():
    x = torch.randn(batch_size, features)
    return [x]

def get_init_inputs():
    return [p, dim, maxnorm]
