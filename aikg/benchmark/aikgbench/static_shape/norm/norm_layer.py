import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, normalized_shape):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(self, input_tensor):
        # torch.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05, cudnn_enable=True)
        # Applies Layer Normalization over a mini-batch of inputs.
        # Layer normalization is commonly used in neural networks for:
        # - Stabilizing training dynamics
        # - Reducing internal covariate shift
        # - Normalizing activations across feature dimensions
        
        # Create weight and bias tensors with the correct shape
        weight = torch.ones(self.normalized_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        bias = torch.zeros(self.normalized_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        return torch.layer_norm(input_tensor, self.normalized_shape, weight, bias)


def get_inputs():
    # Batch size: 16
    # Sequences: 16
    # Hidden features: 4096
    input_tensor = torch.randn((16, 16, 4096), dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for layer_norm
    normalized_shape = (4096,)
    return [normalized_shape]