import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, shape=None):
        super(Model, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        # torch.broadcast_to(input, shape)
        # Broadcasts input to the shape shape.
        # Broadcasting is the process of making tensors with different shapes have compatible shapes for element-wise operations.
        # This operation is commonly used in neural networks for:
        # - Expanding tensors to match dimensions for broadcasting
        # - Creating patterned matrices
        # - Implementing certain attention mechanisms
        return torch.broadcast_to(input_tensor, self.shape)


def get_inputs():
    # Shape (1, 4096) represents a tensor that can be broadcasted to (1024, 4096)
    input_tensor = torch.randn(1, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific shape for broadcasting
    # Broadcast to (1024, 4096)
    shape = (1024, 4096)
    return [shape]