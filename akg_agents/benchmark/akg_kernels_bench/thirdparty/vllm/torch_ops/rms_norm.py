import torch
from typing import Optional
import torch.nn as nn

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/layernorm.py:91-253
# 测试文件: tests/kernels/core/test_layernorm.py
# vLLM类: RMSNorm
# 功能: RMS归一化，计算 x -> w * x / sqrt(E[x^2] + eps)
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        weight_dtype = dtype or torch.get_default_dtype()
        self.has_weight = has_weight
        self.weight = torch.ones(hidden_size, dtype=weight_dtype)
        if self.has_weight:
            self.weight = nn.Parameter(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        精确复制vllm的forward_native/forward_static实现
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32)

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size to be {self.hidden_size}, "
                f"but found: {x.shape[-1]}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if self.hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {self.hidden_size}"
                )
            x_var = x[:, :, : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        if self.has_weight:
            x = x * self.weight
        return x


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        from vllm.model_executor.layers.layernorm import RMSNorm
        self.layer = RMSNorm(
            hidden_size=hidden_size,
            eps=eps,
            var_hidden_size=var_hidden_size,
            has_weight=has_weight,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def get_inputs():
    """
    生成测试输入
    参考: tests/kernels/core/test_layernorm.py
    NUM_TOKENS = [7, 83, 2048], HIDDEN_SIZES = [64, 769, 2048, 4096, 5120, 8192]
    """
    num_tokens = 83
    hidden_size = 2048
    dtype = torch.float16

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    return [x]


def get_init_inputs():
    """生成初始化参数"""
    return [2048, 1e-6, None, True, None]  # hidden_size, eps, var_hidden_size, has_weight, dtype

