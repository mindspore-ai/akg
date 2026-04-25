import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that computes the pairwise distance between two input tensors
    using the Lp norm (default: L2 / Euclidean distance).
    """
    def __init__(self, p: float = 2.0):
        """
        Initializes the PairwiseDistance layer.

        Args:
            p (float): The norm degree. Default: 2.0 (Euclidean distance).
        """
        super(Model, self).__init__()
        self.pdist = nn.PairwiseDistance(p=p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes the pairwise distance between two tensors.

        Args:
            x1 (torch.Tensor): First input tensor of shape (batch_size, dim).
            x2 (torch.Tensor): Second input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Pairwise distances of shape (batch_size,).
        """
        return self.pdist(x1, x2)

batch_size = 256
dim = 65536

def get_inputs():
    x1 = torch.randn(batch_size, dim)
    x2 = torch.randn(batch_size, dim)
    return [x1, x2]

def get_init_inputs():
    return [2.0]
