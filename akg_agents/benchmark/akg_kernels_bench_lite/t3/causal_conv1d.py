import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Causal Conv1D for Mamba architecture (PyTorch Reference Implementation).
    
    Performs causal 1D convolution with optional SiLU activation.
    Used in Mamba SSM (State Space Model) for efficient sequence modeling.
    
    Formula:
        x_padded = concat(conv_state, x)
        out = conv1d(x_padded, weight, bias) + (optional: silu activation)
        conv_state = x_padded[:, :, -(width-1):]  # update state
    """

    def __init__(self, activation="silu"):
        super(Model, self).__init__()
        self.activation = activation

    def forward(self, x, conv_state, weight, bias, conv_state_indices):
        """
        Args:
            x: (batch, dim) - input tensor
            conv_state: (batch, dim, width-1) - convolution state buffer
            weight: (dim, width) - convolution weights
            bias: (dim,) - convolution bias
            conv_state_indices: (batch,) - indices for state selection (ignored in ref)
        
        Returns:
            out: (batch, dim) - output after causal conv1d + activation
        """
        batch, dim = x.shape
        width = weight.shape[1]
        
        # Expand to 3D: (batch, dim, 1)
        x_3d = x.unsqueeze(-1)
        
        # Concatenate conv_state and x
        x_padded = torch.cat([conv_state, x_3d], dim=-1).to(weight.dtype)
        
        # Grouped convolution
        out = F.conv1d(x_padded, weight.unsqueeze(1), bias, padding=0, groups=dim)
        out = out.squeeze(-1)
        
        # SiLU activation
        if self.activation == "silu":
            out = F.silu(out)
        
        # Update conv_state (in-place)
        conv_state.copy_(x_padded[:, :, -(width-1):])
        
        return out.to(x.dtype)


def get_inputs():
    batch = 32
    dim = 2048
    width = 4
    
    x = torch.randn(batch, dim, dtype=torch.float16)
    conv_state = torch.randn(batch, dim, width - 1, dtype=torch.float16)
    weight = torch.randn(dim, width, dtype=torch.float16)
    bias = torch.randn(dim, dtype=torch.float16)
    conv_state_indices = torch.arange(batch, dtype=torch.int32)
    
    return [x, conv_state, weight, bias, conv_state_indices]


def get_init_inputs():
    return ["silu"]
