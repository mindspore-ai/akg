import torch
import torch.nn as nn


class Model(nn.Module):
    """
    RMS Normalization (std path) for small shape variant.
    Formula: output = (x / sqrt(mean(x^2) + epsilon)) * gamma
    """
    def __init__(self, epsilon=1e-5):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, input_tensor, gamma):
        variance = input_tensor.pow(2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.epsilon)
        result = input_tensor / std
        result = result * gamma
        return result


def get_inputs():
    # Match shapes of the small rstd variant for comparison
    input_tensor = torch.randn(16, 512, 2688, dtype=torch.float32)
    gamma = torch.randn(2688, dtype=torch.float32)
    return [input_tensor, gamma]


def get_init_inputs():
    epsilon = 1e-5
    return [epsilon]


