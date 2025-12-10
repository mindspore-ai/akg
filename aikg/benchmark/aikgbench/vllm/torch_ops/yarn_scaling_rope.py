import math

import torch
from typing import Optional, Tuple
import torch.nn as nn

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py
# vLLM类: YaRNScalingRotaryEmbedding
# 功能: YaRN缩放的旋转位置编码
# ============================================================================


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (
        dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))


def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    low = yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    low = math.floor(low)
    high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype
) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class Model(nn.Module):
    """原生PyTorch实现"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float = 1.0,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self.mscale = (
            yarn_get_mscale(self.scaling_factor) * attn_factor
        )
        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # 基础inv_freq
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                / self.rotary_dim
            )
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = 1 - yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2, torch.float32
        )
        inv_freq = inv_freq / self.scaling_factor * (
            1 - inv_freq_mask
        ) + inv_freq * inv_freq_mask

        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = apply_rotary_emb_torch(
            query_rot, cos, sin, self.is_neox_style
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., : self.rotary_dim]
            key_pass = key[..., self.rotary_dim :]
            key_rot = apply_rotary_emb_torch(
                key_rot, cos, sin, self.is_neox_style
            )
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key


class ModelVLLM(nn.Module):
    """vLLM优化实现"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float = 1.0,
        extrapolation_factor: float = 1.0,
        attn_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        super().__init__()
        from vllm.model_executor.layers.rotary_embedding import (
            YaRNScalingRotaryEmbedding,
        )
        self.layer = YaRNScalingRotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
            scaling_factor=scaling_factor,
            extrapolation_factor=extrapolation_factor,
            attn_factor=attn_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.layer(positions, query, key)


def get_inputs():
    """生成测试输入"""
    batch_size = 16
    seq_len = 512
    num_heads = 8
    head_size = 64
    dtype = torch.float16

    positions = torch.randint(
        0, 2048, (batch_size * seq_len,), dtype=torch.long
    )
    query = torch.randn(
        batch_size * seq_len, num_heads * head_size, dtype=dtype
    )
    key = torch.randn(batch_size * seq_len, num_heads * head_size, dtype=dtype)

    return [positions, query, key]


def get_init_inputs():
    """生成初始化参数"""
    return [64, 64, 2048, 10000.0, True, torch.float16, 2.0, 1.0, 1.0, 32, 1]

