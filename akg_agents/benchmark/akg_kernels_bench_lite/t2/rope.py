import torch
import torch.nn as nn
import torch_npu


class Model(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Formula:
        x1, x2 = chunk(x, 2, dim=-1)
        x_new = concat(-x2, x1)
        output = cos * x + sin * x_new

    On NPU, use torch_npu.npu_rotary_mul(x, cos, sin).
    
    Constraints:
        - x, cos, sin must be 4D tensors
        - Last dimension must be multiple of 64
        - cos and sin last dim must match x last dim
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, cos, sin):
        return torch_npu.npu_rotary_mul(x, cos, sin)


def get_inputs():
    batch = 16  # B*N should be <= 1024 for gradient computation
    num_heads = 48
    seq_len = 1000
    head_dim = 128  # Must be multiple of 2 and < 896
    x = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float16)
    # cos and sin must have same last dim as x, shape: 11SD
    cos = torch.randn(1, 1, seq_len, head_dim, dtype=torch.float16).clamp(-1, 1)
    sin = torch.randn(1, 1, seq_len, head_dim, dtype=torch.float16).clamp(-1, 1)
    return [x, cos, sin]


def get_init_inputs():
    return []
