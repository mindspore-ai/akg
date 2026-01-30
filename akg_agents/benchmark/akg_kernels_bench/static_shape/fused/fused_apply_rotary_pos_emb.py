import torch
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self, layout=1):
        super(Model, self).__init__()
        self.layout = layout
    
    def forward(self, query, key, cos, sin):
        # Apply rotary position embedding (even/odd pairing on last dim):
        # out_even = x_even * cos - x_odd * sin
        # out_odd  = x_even * sin + x_odd * cos
        # Shapes:
        #   query/key: [B, S, H, D]
        #   cos/sin:   [B, S, H, D/2] or broadcast-compatible
        dim = query.shape[-1]
        q_even = query[..., 0:dim:2]
        q_odd  = query[..., 1:dim:2]
        k_even = key[..., 0:dim:2]
        k_odd  = key[..., 1:dim:2]
        q_out_even = q_even * cos - q_odd * sin
        q_out_odd  = q_even * sin + q_odd * cos
        k_out_even = k_even * cos - k_odd * sin
        k_out_odd  = k_even * sin + k_odd * cos
        q_out = torch.empty_like(query)
        k_out = torch.empty_like(key)
        q_out[..., 0:dim:2] = q_out_even
        q_out[..., 1:dim:2] = q_out_odd
        k_out[..., 0:dim:2] = k_out_even
        k_out[..., 1:dim:2] = k_out_odd
        return q_out, k_out

def get_inputs():
    # Shapes aligned to even/odd pairing convention
    # query/key: [B, S, H, D], cos/sin: [B, S, H, D/2]
    B, S, H, D = 16, 10, 1, 4096
    query = torch.randn(B, S, H, D, dtype=torch.float32)
    key = torch.randn(B, S, H, D, dtype=torch.float32)
    cos = torch.randn(B, S, H, D // 2, dtype=torch.float32)
    sin = torch.randn(B, S, H, D // 2, dtype=torch.float32)
    return [query, key, cos, sin]

def get_init_inputs():
    # No parameters needed for fused apply rotary embedding
    return []