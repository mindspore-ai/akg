import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Softmax operation that normalizes the input tensor along a specified dimension.
    This operation is commonly used in neural networks for:
    - Converting logits to probabilities in classification tasks
    - Used in attention mechanisms to compute attention weights
    - Normalizing outputs to form probability distributions
    
    Formula: output_i = exp(input_i) / sum(exp(input_j)) for j in dimension
    """
    def __init__(self, dim=-1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var):
        # Softmax operation on var along the specified dimension
        result = torch.softmax(var, dim=self.dim)
        return result


def get_inputs_dyn_list():
    # Softmax along dimension -1 variation cases with both aligned and non-aligned shapes

    # Case 1
    inputs1 = torch.randn(1, 1024, 50257, dtype=torch.float32)

    # Case 2
    inputs2 = torch.randn(4, 1024, 50257, dtype=torch.float32)

    # Case 3
    inputs3 = torch.randn(4, 2048, 32000, dtype=torch.float32)

    # Case 4
    inputs4 = torch.randn(1, 4096, 128256, dtype=torch.float32)

    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4],
    ]

def get_init_inputs():
    # Fixed parameters for softmax along dimension -1
    dim = -1  # Dimension along which to apply softmax
    return [dim]
