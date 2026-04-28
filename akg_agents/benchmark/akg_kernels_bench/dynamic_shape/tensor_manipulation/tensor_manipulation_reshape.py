import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, shape=(64, 64, 1024)):
        super(Model, self).__init__()
        self.shape = shape

    def forward(self, input_tensor):
        # torch.reshape(input, shape)
        # Returns a tensor with the same data and number of elements as input, but with the specified shape.
        # This operation is commonly used in neural networks for:
        # - Changing tensor shapes for specific operations
        # - Implementing certain matrix operations
        # - Reshaping tensors for compatibility with other operations
        return torch.reshape(input_tensor, self.shape)


def get_inputs_dyn_list():
    # All inputs will be reshaped to 2D tensors with same total elements
    # Small shape case: (128, 1024) -> (64, 2048) = 131,072 elements
    input1 = torch.randn(128, 1024, dtype=torch.float32)

    # Middle shape case: (512, 3072) -> (64, 24576) = 1,572,864 elements  
    input2 = torch.randn(512, 3072, dtype=torch.float32)

    # Large shape case: (1024, 4096) -> (64, 65536) = 4,194,304 elements
    input3 = torch.randn(1024, 4096, dtype=torch.float32)

    # Non-aligned shape case: (513, 5120) -> (64, 41040) = 2,626,560 elements
    input4 = torch.randn(513, 5120, dtype=torch.float32)

    return [[input1], [input2], [input3], [input4]]


def get_init_inputs():
    # Parameters needed for reshape - flatten to 2D with first dim=64
    # This creates a consistent pattern: (64, remaining_elements)
    shape = (64, -1)  # -1 means infer the second dimension
    return [shape]