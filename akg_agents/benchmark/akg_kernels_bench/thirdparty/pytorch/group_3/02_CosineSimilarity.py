import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that computes Cosine Similarity between two tensors.
    Uses L2 norms internally: cos_sim(x1, x2) = (x1 · x2) / (||x1|| * ||x2||).
    """
    def __init__(self):
        """
        Initializes the CosineSimilarity layer.
        """
        super(Model, self).__init__()
        self.cosine_sim = nn.CosineSimilarity(dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Computes Cosine Similarity between two tensors.

        Args:
            x1 (torch.Tensor): First input tensor of shape (batch_size, dim).
            x2 (torch.Tensor): Second input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Cosine similarity values of shape (batch_size,).
        """
        return self.cosine_sim(x1, x2)

batch_size = 256
dim = 65536

def get_inputs():
    x1 = torch.randn(batch_size, dim)
    x2 = torch.randn(batch_size, dim)
    return [x1, x2]

def get_init_inputs():
    return []
