"""
RadixAttention - PyTorch Reference Implementation
===================================================
Based on: "SGLang: Efficient Execution of Structured Language Model Programs"
(Zheng et al., NeurIPS 2024 / LMSYS)
"""

import torch
import torch.nn as nn
import numpy as np

NEG_INF = -1e10


class host_impl:
    """
    Internal implementation details for RadixAttention.
    RadixTreeNode and RadixTree manage prefix-based KV cache with LRU eviction.
    """

    class RadixTreeNode:
        def __init__(self):
            self.children         = {}
            self.kv_cache         = None
            self.last_access_time = 0
            self.ref_count        = 0
            self.token_prefix     = ()

    class RadixTree:
        def __init__(self, max_cached_entries: int = 64):
            self.root = host_impl.RadixTreeNode()
            self.root.token_prefix  = ()
            self.max_cached_entries = max_cached_entries
            self.current_entries    = 0
            self.time_counter       = 0
            self.all_nodes          = [self.root]

        def _tick(self):
            self.time_counter += 1
            return self.time_counter

        def match_prefix(self, tokens):
            node, pos = self.root, 0
            while pos < len(tokens):
                token = tokens[pos]
                if token not in node.children:
                    break
                edge_tokens, child = node.children[token]
                remaining = tokens[pos: pos + len(edge_tokens)]
                if len(remaining) < len(edge_tokens):
                    match_count = 0
                    for a, b in zip(remaining, edge_tokens):
                        if a != b:
                            break
                        match_count += 1
                    pos += match_count
                    break
                if remaining == edge_tokens:
                    pos += len(edge_tokens)
                    node = child
                    node.last_access_time = self._tick()
                else:
                    match_count = 0
                    for a, b in zip(remaining, edge_tokens):
                        if a != b:
                            break
                        match_count += 1
                    pos += match_count
                    break
            return pos, node

        def insert(self, tokens, kv_cache):
            while self.current_entries >= self.max_cached_entries:
                if not self._evict_lru():
                    break
            matched_len, node = self.match_prefix(tokens)
            if matched_len == len(tokens):
                node.kv_cache         = kv_cache
                node.last_access_time = self._tick()
                return
            remaining = tokens[matched_len:]
            new_node  = host_impl.RadixTreeNode()
            new_node.token_prefix     = tokens
            new_node.kv_cache         = kv_cache
            new_node.last_access_time = self._tick()
            node.children[remaining[0]] = (remaining, new_node)
            self.all_nodes.append(new_node)
            self.current_entries += 1

        def _evict_lru(self):
            leaves = [n for n in self.all_nodes
                      if n is not self.root and not n.children and n.ref_count == 0]
            if not leaves:
                return False
            lru = min(leaves, key=lambda n: n.last_access_time)
            for node in self.all_nodes:
                for key, (_, child) in list(node.children.items()):
                    if child is lru:
                        del node.children[key]
            self.all_nodes.remove(lru)
            self.current_entries -= 1
            return True

        def get_cached_kv(self, tokens):
            matched_len, node = self.match_prefix(tokens)
            if matched_len > 0 and node.kv_cache is not None:
                node.last_access_time = self._tick()
                K_c, V_c   = node.kv_cache
                actual_len = min(matched_len, K_c.shape[0])
                return actual_len, K_c[:actual_len], V_c[:actual_len]
            return 0, None, None


class Model(nn.Module):
    """
    RadixAttention: attention with automatic KV prefix reuse via a radix tree.

    Args:
        max_cached_entries (int): Max entries before LRU eviction. Defaults to 64.
    """
    def __init__(self, max_cached_entries: int = 64):
        super(Model, self).__init__()
        self.max_cached_entries = max_cached_entries
        self.radix_tree = host_impl.RadixTree(max_cached_entries)

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

        prefix_tokens = [tuple(range(N)) for _ in range(B)]

        for b in range(B):
            tokens = prefix_tokens[b]
            self.radix_tree.get_cached_kv(tokens)

            S = torch.einsum('h i d, h j d -> h i j', Q[b] * scale, K[b])
            if mask is not None:
                mask_exp = mask[b].unsqueeze(0).unsqueeze(1)  # (1, 1, N)
                S = torch.where(mask_exp > 0, S, NEG_INF)

            attn = torch.softmax(S, dim=-1)
            O[b] = torch.einsum('h i j, h j d -> h i d', attn, V[b])

            K_store = K[b].permute(1, 0, 2).contiguous()  # (N, H, D)
            V_store = V[b].permute(1, 0, 2).contiguous()
            self.radix_tree.insert(tokens, (K_store.clone(), V_store.clone()))

        return O

B, H, N, D = 32, 8, 1024, 256
def get_inputs():
    Q    = torch.randn(B, H, N, D)
    K    = torch.randn(B, H, N, D)
    V    = torch.randn(B, H, N, D)
    mask = (torch.rand(B, N) > 0.2).float()
    return [Q, K, V, mask]


def get_init_inputs():
    return [64]  # max_cached_entries