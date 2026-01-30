import torch
import torch.nn as nn


class Model(nn.Module):
    """
    RMS Normalization (std path) that normalizes the input tensor using
    standard deviation instead of reciprocal standard deviation.
    Formula: output = (x / sqrt(mean(x^2) + epsilon)) * gamma
    """
    def __init__(self, epsilon=1e-5):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, input_tensor, gamma):
        # Compute standard deviation path
        variance = input_tensor.pow(2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.epsilon)
        result = input_tensor / std
        result = result * gamma
        return result


def get_inputs():
    # Use the same shapes as the rstd variant for apples-to-apples comparison
    input_tensor = torch.randn(32, 1024, 4096, dtype=torch.float32)
    gamma = torch.randn(4096, dtype=torch.float32)
    return [input_tensor, gamma]


def get_init_inputs():
    epsilon = 1e-5
    return [epsilon]


