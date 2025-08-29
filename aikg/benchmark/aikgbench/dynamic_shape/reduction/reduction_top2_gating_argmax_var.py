import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var):
        # torch.max(input, dim, keepdim=False)
        # Returns a namedtuple (values, indices) where values is the maximum value of each row
        # of the input tensor in the given dimension dim, and indices is the index location of
        # each maximum value found.
        # This operation is commonly used in neural networks for:
        # - Max pooling in convolutional networks
        # - Finding the most activated neuron in a layer
        # - Attention mechanisms in transformers
        # - Gating mechanisms in mixture-of-experts models
        return torch.max(var, dim=self.dim)


def get_inputs_dyn_list():
    # Top-2 gating argmax along dimension 1 variation cases with both aligned and non-aligned shapes

    # Case 1
    inputs1 = torch.randn(8, 2048, 8, dtype=torch.float32)

    # Case 2
    inputs2 = torch.randn(16, 1024, 8, dtype=torch.float32)

    # Case 3
    inputs3 = torch.randn(1, 150, 8, dtype=torch.float32)

    # Case 4
    inputs4 = torch.randn(2, 4096, 8, dtype=torch.float32)

    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4],
    ]


def get_init_inputs():
    # Fixed parameters for top-2 gating argmax along dimension 1
    dim = 1  # Reduce along second dimension (features dimension)
    return [dim]
