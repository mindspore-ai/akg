import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, sm_scale):
        super().__init__()
        self.sm_scale = sm_scale

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:        
        # This is a flash attention implementation.
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value,
            scale=self.sm_scale
        )


def get_inputs():
    # Using shapes that are representative of large model computations in transformer models
    Z, H, N_CTX, HEAD_DIM = 4, 32, 1024, 64
    dtype = torch.bfloat16
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype).normal_(mean=0.0, std=0.5).requires_grad_())
    return [q, k, v]


def get_init_inputs():
    sm_scale = 0.5
    return [sm_scale]