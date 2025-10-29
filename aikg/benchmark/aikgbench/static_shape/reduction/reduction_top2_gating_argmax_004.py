import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # Top-2 gating argmax operation
        # Returns the top-2 values and their indices along the specified dimension
        # This operation is commonly used in neural networks for:
        # - Top-2 selection in mixture-of-experts models
        # - Gating mechanisms that select top-2 experts
        # - Sparse attention mechanisms
        # - Hierarchical routing in neural networks
        topk_values, topk_indices = torch.topk(input_tensor, k=2)
        return topk_values, topk_indices


def get_inputs():
    # Sequence length: 16384
    input_tensor = torch.randn(16384, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    return []