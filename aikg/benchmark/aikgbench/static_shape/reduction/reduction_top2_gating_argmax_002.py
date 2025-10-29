import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # Top-2 gating argmax operation
        # Returns the top-2 values and their indices along the specified dimension
        # This operation is commonly used in neural networks for:
        # - Top-2 selection in mixture-of-experts models
        # - Gating mechanisms that select top-2 experts
        # - Sparse attention mechanisms
        # - Hierarchical routing in neural networks
        topk_values, topk_indices = torch.topk(input_tensor, k=2, dim=self.dim)
        return topk_values, topk_indices


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific dim value for reduction
    # Reduce along first dimension (batch dimension)
    dim = 0
    return [dim]