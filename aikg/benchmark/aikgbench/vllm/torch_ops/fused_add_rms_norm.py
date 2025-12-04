import torch
from typing import Optional, Tuple
import torch.nn as nn

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/layernorm.py:35-53, 91-253
# 测试文件: tests/kernels/core/test_layernorm.py
# vLLM函数/类: fused_add_rms_norm, RMSNorm
# 功能: 融合的残差加RMS归一化
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        精确复制vllm的forward_static实现
        返回 (normalized_output, updated_residual)
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32)

        # 融合加法
        x = x + residual
        residual = x.to(orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)

        if self.has_weight:
            x = x * self.weight

        return x, residual


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        from vllm.model_executor.layers.layernorm import RMSNorm
        self.layer = RMSNorm(
            hidden_size=hidden_size, eps=eps, has_weight=has_weight, dtype=dtype
        )

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.layer(x, residual)


def get_inputs():
    """
    生成测试输入
    参考: tests/kernels/core/test_layernorm.py
    """
    num_tokens = 83
    hidden_size = 2048
    dtype = torch.float16

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype)

    return [x, residual]


def get_init_inputs():
    """生成初始化参数"""
    return [2048, 1e-6, True, None]  # hidden_size, eps, has_weight, dtype

