import torch
import torch.nn as nn

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/layernorm.py:256-321
# 测试文件: tests/kernels/core/test_layernorm.py
# vLLM类: GemmaRMSNorm
# 功能: Gemma模型的RMS归一化，使用 x * (1 + w) 代替 x * w
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        精确复制vllm的forward_static实现
        Gemma: (x * w).to(dtype) 代替 x.to(dtype) * w
        """
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        # Gemma使用 (x * (1 + w)).to(dtype)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        return x


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        self.layer = GemmaRMSNorm(hidden_size=hidden_size, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def get_inputs():
    """生成测试输入"""
    num_tokens = 83
    hidden_size = 2048
    dtype = torch.float16

    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    return [x]


def get_init_inputs():
    """生成初始化参数"""
    return [2048, 1e-6]  # hidden_size, eps

