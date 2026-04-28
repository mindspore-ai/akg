import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Rotary embedding (RoPE) operation.
    This operation is commonly used in neural networks for:
    - Adding positional information to transformer attention
    - Enabling relative position awareness via complex plane rotation

    Formula (even/odd pairing):
    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, state, cos, sin):
        _, _, dim = state.shape
        state_x = state[:, :, 0:dim:2]
        state_y = state[:, :, 1:dim:2]
        out_x = state_x * cos - state_y * sin
        out_y = state_x * sin + state_y * cos
        out = torch.empty_like(state)
        out[:, :, 0:dim:2] = out_x
        out[:, :, 1:dim:2] = out_y
        return out

def get_inputs():
    # Input shapes aligned with reference torch implementation
    tokens_num = 32
    num_heads = 48
    head_dim = 128
    state = torch.randn((tokens_num, num_heads, head_dim), dtype=torch.float32)
    cos = torch.randn((tokens_num, 1, head_dim // 2), dtype=torch.float32)
    sin = torch.randn((tokens_num, 1, head_dim // 2), dtype=torch.float32)
    return [state, cos, sin]

def get_init_inputs():
    # No parameters needed for rotary embedding
    return []