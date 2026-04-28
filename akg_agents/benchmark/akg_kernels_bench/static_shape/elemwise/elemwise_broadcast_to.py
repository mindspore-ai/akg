import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.broadcast_to(input, shape)
        # Broadcasts input to the shape shape.
        # This operation is commonly used in neural networks for:
        # - Broadcasting tensors to match dimensions for operations
        # - Expanding tensors for compatibility with other operations
        # - Implementing certain mathematical transformations
        return torch.broadcast_to(input_tensor, (1024, 4096))


def get_inputs():
    # Shape (1, 4096) represents a tensor that can be broadcasted to (1024, 4096)
    input_tensor = torch.randn(1, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for broadcast_to
    return []