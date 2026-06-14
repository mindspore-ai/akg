"""
FlashAttention - PyTorch Reference Implementation
===================================================
Based on: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
(Dao et al., NeurIPS 2022)
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

NEG_INF = -1e10
EPSILON = 1e-10


class Model(nn.Module):
    """
    FlashAttention forward pass with tiling and online softmax rescaling.

    Args:
        block_size (int): Tile size for KV blocks. Defaults to 256.
    """
    def __init__(self, block_size: int = 256):
        super(Model, self).__init__()
        self.block_size = block_size

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            Q: (batch, heads, seq_len, head_dim)
            K: (batch, heads, seq_len, head_dim)
            V: (batch, heads, seq_len, head_dim)
            mask: (batch, seq_len) — 1 for valid, 0 for masked

        Returns:
            O: (batch, heads, seq_len, head_dim)
        """
        B, H, N, D = Q.shape
        scale = 1.0 / np.sqrt(D)

        O = torch.zeros_like(Q)
        l = torch.zeros(B, H, N, 1, device=Q.device, dtype=Q.dtype)
        m = torch.full((B, H, N, 1), NEG_INF, device=Q.device, dtype=Q.dtype)

        Q_BLOCK_SIZE = min(self.block_size, D)
        KV_BLOCK_SIZE = self.block_size

        Q_blocks = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_blocks = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_blocks = torch.split(V, KV_BLOCK_SIZE, dim=2)

        if mask is not None:
            mask_blocks = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))
        else:
            mask_blocks = [None] * len(K_blocks)

        Tc = len(K_blocks)
        Tr = len(Q_blocks)

        O_blocks = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_blocks = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_blocks = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j in range(Tc):
            Kj, Vj, maskj = K_blocks[j], V_blocks[j], mask_blocks[j]

            for i in range(Tr):
                Qi, Oi, li, mi = Q_blocks[i], O_blocks[i], l_blocks[i], m_blocks[i]

                S_ij = torch.einsum('b h i d, b h j d -> b h i j', Qi * scale, Kj)

                if maskj is not None:
                    maskj_expanded = rearrange(maskj, 'b j -> b 1 1 j')
                    S_ij = torch.where(maskj_expanded > 0, S_ij, NEG_INF)

                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)
                P_ij = torch.exp(S_ij - m_block_ij)

                if maskj is not None:
                    P_ij = torch.where(maskj_expanded > 0, P_ij, 0.0)

                l_block_ij = torch.sum(P_ij, dim=-1, keepdim=True) + EPSILON
                P_ij_Vj = torch.einsum('b h i j, b h j d -> b h i d', P_ij, Vj)

                mi_new = torch.maximum(m_block_ij, mi)
                li_new = (torch.exp(mi - mi_new) * li
                          + torch.exp(m_block_ij - mi_new) * l_block_ij)

                O_blocks[i] = ((li / li_new) * torch.exp(mi - mi_new) * Oi
                               + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj)
                l_blocks[i] = li_new
                m_blocks[i] = mi_new

        return torch.cat(O_blocks, dim=2)


# Test code
batch_size = 2
num_heads = 4
seq_len = 128
head_dim = 64
block_size = 64


def get_inputs():
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim)
    mask = (torch.rand(batch_size, seq_len) > 0.2).float()
    return [Q, K, V, mask]


def get_init_inputs():
    return [block_size]