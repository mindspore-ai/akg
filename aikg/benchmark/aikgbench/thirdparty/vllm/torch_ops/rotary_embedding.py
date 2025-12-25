import torch
from typing import Optional, Tuple
import torch.nn as nn

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/rotary_embedding/base.py:92-241
# 测试文件: tests/kernels/core/test_rotary_embedding.py
# vLLM类: RotaryEmbedding
# 功能: 旋转位置编码
# ============================================================================


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """应用旋转位置编码的原生PyTorch实现"""
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


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        # 计算cos/sin缓存
        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        精确复制vllm的forward_static实现
        """
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
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ):
        super().__init__()
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
        self.layer = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.layer(positions, query, key)


def get_inputs():
    """
    生成测试输入
    参考: tests/kernels/core/test_rotary_embedding.py
    """
    batch_size = 16
    seq_len = 512
    num_heads = 8
    head_size = 64
    dtype = torch.float16

    # 位置索引
    positions = torch.randint(
        0, 2048, (batch_size * seq_len,), dtype=torch.long
    )

    # query和key张量 [num_tokens, num_heads * head_size]
    query = torch.randn(
        batch_size * seq_len, num_heads * head_size, dtype=dtype
    )
    key = torch.randn(batch_size * seq_len, num_heads * head_size, dtype=dtype)

    return [positions, query, key]


def get_init_inputs():
    """生成初始化参数"""
    return [
        64,  # head_size
        64,  # rotary_dim
        2048,  # max_position_embeddings
        10000.0,  # base
        True,  # is_neox_style
        torch.float16,  # dtype
    ]

