"""
PagedAttention - PyTorch Reference Implementation
===================================================
Based on: "Efficient Memory Management for Large Language Model Serving
with PagedAttention" (Kwon et al., SOSP 2023 / vLLM)
"""

import torch
import torch.nn as nn
import numpy as np

NEG_INF = -1e10


class host_impl:
    """
    Internal implementation details for PagedAttention.
    PagedKVCache simulates paged physical memory for KV storage.
    """

    class PagedKVCache:
        def __init__(self, num_blocks, page_size, num_heads, head_dim, device, dtype):
            self.page_size    = page_size
            self.k_pool       = torch.zeros(num_blocks, page_size, num_heads, head_dim,
                                            device=device, dtype=dtype)
            self.v_pool       = torch.zeros_like(self.k_pool)
            self.free_blocks  = list(range(num_blocks))
            self.block_tables = {}
            self.seq_lengths  = {}

        def allocate_block(self):
            if not self.free_blocks:
                raise RuntimeError("No free blocks available!")
            return self.free_blocks.pop(0)

        def init_sequence(self, seq_id):
            self.block_tables[seq_id] = []
            self.seq_lengths[seq_id]  = 0

        def append_kv(self, seq_id, k_token, v_token):
            seq_len       = self.seq_lengths[seq_id]
            logical_block = seq_len // self.page_size
            offset        = seq_len %  self.page_size
            if logical_block >= len(self.block_tables[seq_id]):
                self.block_tables[seq_id].append(self.allocate_block())
            phys = self.block_tables[seq_id][logical_block]
            self.k_pool[phys, offset] = k_token
            self.v_pool[phys, offset] = v_token
            self.seq_lengths[seq_id] += 1

        def write_kv_sequence(self, seq_id, K_seq, V_seq):
            self.init_sequence(seq_id)
            for t in range(K_seq.shape[0]):
                self.append_kv(seq_id, K_seq[t], V_seq[t])

        def read_kv_sequence(self, seq_id):
            seq_len = self.seq_lengths[seq_id]
            table   = self.block_tables[seq_id]
            H, D    = self.k_pool.shape[2], self.k_pool.shape[3]
            K_out   = torch.zeros(seq_len, H, D,
                                  device=self.k_pool.device, dtype=self.k_pool.dtype)
            V_out   = torch.zeros_like(K_out)
            for t in range(seq_len):
                phys     = table[t // self.page_size]
                K_out[t] = self.k_pool[phys, t % self.page_size]
                V_out[t] = self.v_pool[phys, t % self.page_size]
            return K_out, V_out

        def free_sequence(self, seq_id):
            for phys in self.block_tables.get(seq_id, []):
                self.free_blocks.append(phys)
            self.block_tables.pop(seq_id, None)
            self.seq_lengths.pop(seq_id, None)


class Model(nn.Module):
    """
    PagedAttention: standard attention over a paged KV cache.

    Args:
        page_size (int): Number of tokens per KV cache page. Defaults to 16.
    """
    def __init__(self, page_size: int = 16):
        super(Model, self).__init__()
        self.page_size = page_size

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
        O     = torch.zeros_like(Q)

        for b in range(B):
            n_blocks = (N + self.page_size - 1) // self.page_size
            cache = host_impl.PagedKVCache(
                num_blocks=n_blocks * 2,
                page_size=self.page_size,
                num_heads=H, head_dim=D,
                device=Q.device, dtype=Q.dtype,
            )

            dummy = [cache.allocate_block() for _ in range(n_blocks // 2)]
            cache.write_kv_sequence(0, K[b].permute(1, 0, 2), V[b].permute(1, 0, 2))
            cache.free_blocks.extend(dummy)

            K_g, V_g = cache.read_kv_sequence(0)
            K_g = K_g.permute(1, 0, 2)  # (H, N, D)
            V_g = V_g.permute(1, 0, 2)

            S = torch.einsum('h i d, h j d -> h i j', Q[b] * scale, K_g)
            if mask is not None:
                mask_exp = mask[b].unsqueeze(0).unsqueeze(1)  # (1, 1, N)
                S = torch.where(mask_exp > 0, S, NEG_INF)

            attn = torch.softmax(S, dim=-1)
            O[b] = torch.einsum('h i j, h j d -> h i d', attn, V_g)
            cache.free_sequence(0)

        return O

B, H, N, D = 2, 4, 128, 64

def get_inputs():
    Q    = torch.randn(B, H, N, D)
    K    = torch.randn(B, H, N, D)
    V    = torch.randn(B, H, N, D)
    mask = (torch.rand(B, N) > 0.2).float()
    return [Q, K, V, mask]


def get_init_inputs():
    return [16]  # page_size