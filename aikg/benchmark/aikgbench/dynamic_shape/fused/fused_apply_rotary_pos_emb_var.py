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

def get_inputs_dyn_list():
    # Case 1: Small batch, small seq (non-aligned batch)
    B1, S1, H1, D1 = 15, 127, 1, 1344
    query1 = torch.randn(B1, S1, H1, D1, dtype=torch.float32)
    key1 = torch.randn(B1, S1, H1, D1, dtype=torch.float32)
    cos1 = torch.randn(B1, S1, H1, D1 // 2, dtype=torch.float32)
    sin1 = torch.randn(B1, S1, H1, D1 // 2, dtype=torch.float32)
    
    # Case 2: Small batch, medium seq (aligned batch)
    B2, S2, H2, D2 = 16, 512, 1, 2688
    query2 = torch.randn(B2, S2, H2, D2, dtype=torch.float32)
    key2 = torch.randn(B2, S2, H2, D2, dtype=torch.float32)
    cos2 = torch.randn(B2, S2, H2, D2 // 2, dtype=torch.float32)
    sin2 = torch.randn(B2, S2, H2, D2 // 2, dtype=torch.float32)
    
    # Case 3: Medium batch, large seq (non-aligned batch)
    B3, S3, H3, D3 = 63, 2047, 1, 4096
    query3 = torch.randn(B3, S3, H3, D3, dtype=torch.float32)
    key3 = torch.randn(B3, S3, H3, D3, dtype=torch.float32)
    cos3 = torch.randn(B3, S3, H3, D3 // 2, dtype=torch.float32)
    sin3 = torch.randn(B3, S3, H3, D3 // 2, dtype=torch.float32)
    
    # Case 4: Large batch, large seq (aligned batch)
    B4, S4, H4, D4 = 64, 2048, 1, 5120
    query4 = torch.randn(B4, S4, H4, D4, dtype=torch.float32)
    key4 = torch.randn(B4, S4, H4, D4, dtype=torch.float32)
    cos4 = torch.randn(B4, S4, H4, D4 // 2, dtype=torch.float32)
    sin4 = torch.randn(B4, S4, H4, D4 // 2, dtype=torch.float32)
    
    # Case 5: Very large batch, very large seq (non-aligned batch)
    B5, S5, H5, D5 = 255, 4095, 1, 8192
    query5 = torch.randn(B5, S5, H5, D5, dtype=torch.float32)
    key5 = torch.randn(B5, S5, H5, D5, dtype=torch.float32)
    cos5 = torch.randn(B5, S5, H5, D5 // 2, dtype=torch.float32)
    sin5 = torch.randn(B5, S5, H5, D5 // 2, dtype=torch.float32)
    
    return [
        [query1, key1, cos1, sin1],
        [query2, key2, cos2, sin2],
        [query3, key3, cos3, sin3],
        [query4, key4, cos4, sin4],
        [query5, key5, cos5, sin5]
    ]

def get_init_inputs():
    # No parameters needed for fused apply rotary embedding
    return []