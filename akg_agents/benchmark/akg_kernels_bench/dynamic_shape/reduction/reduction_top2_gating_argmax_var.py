import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var):
        # Top-2 gating argmax operation
        # Returns the top-2 values and their indices along the specified dimension
        # This operation is commonly used in neural networks for:
        # - Top-2 selection in mixture-of-experts models
        # - Gating mechanisms that select top-2 experts
        # - Sparse attention mechanisms
        # - Hierarchical routing in neural networks
        topk_values, topk_indices = torch.topk(var, k=2, dim=self.dim)
        return topk_values, topk_indices


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
