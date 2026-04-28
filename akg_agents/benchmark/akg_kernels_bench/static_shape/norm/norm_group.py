import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_groups):
        super(Model, self).__init__()
        self.num_groups = num_groups

    def forward(self, input_tensor):
        # torch.nn.functional.group_norm(input, num_groups, weight=None, bias=None, eps=1e-05)
        # Applies Group Normalization over a mini-batch of inputs.
        # Group normalization is commonly used in neural networks for:
        # - Normalizing activations in convolutional networks
        # - Providing an alternative to batch normalization
        # - Improving training stability
        
        # Create weight and bias tensors with the correct shape
        # The shape should match the channel dimension (input_tensor.shape[1])
        weight = torch.ones(input_tensor.shape[1], dtype=input_tensor.dtype, device=input_tensor.device)
        bias = torch.zeros(input_tensor.shape[1], dtype=input_tensor.dtype, device=input_tensor.device)
        
        return torch.nn.functional.group_norm(input_tensor, self.num_groups, weight, bias)


def get_inputs():
    # Batch size: 16
    # Channels: 16
    # Spatial locations: 4096
    input_tensor = torch.randn((16, 16, 4096), dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for group_norm
    num_groups = 8
    return [num_groups]