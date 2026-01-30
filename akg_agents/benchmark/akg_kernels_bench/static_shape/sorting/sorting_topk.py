import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, k, dim=-1):
        super(Model, self).__init__()
        self.k = k
        self.dim = dim

    def forward(self, input_tensor):
        # torch.topk(input, k, dim=-1, largest=True, sorted=True, *, out=None)
        # Returns the k largest elements of the given input tensor along a given dimension.
        # Returns a namedtuple (values, indices) where values is the k largest elements
        # and indices is the indices of the k largest elements in the original tensor.
        # Top-k operations are commonly used in neural networks for:
        # - Implementing attention mechanisms
        # - Selecting the most probable tokens in language models
        # - Non-maximum suppression in object detection
        values, indices = torch.topk(input_tensor, self.k, dim=self.dim)
        return values, indices


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn((1024, 4096), dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for topk
    k = 5      # Number of top elements to return
    dim = -1   # Dimension along which to find the topk elements (last dimension)
    return [k, dim]