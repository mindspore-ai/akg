import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # torch.mean(input, dim, keepdim=False, dtype=None)
        # Returns the mean value of all elements in the input tensor or along the specified dimension.
        # This operation is commonly used in neural networks for:
        # - Computing loss functions (e.g., mean squared error)
        # - Normalizing activations across batch dimensions
        # - Pooling operations in convolutional networks
        return torch.mean(input_tensor, dim=self.dim)


def get_inputs():
    # Batch size: 256
    # Hidden dimension: 1024
    # Sequence length: 2048
    input_tensor = torch.randn(256, 1024, 2048, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Reduce along second dimension
    dim = [1]
    return [dim]