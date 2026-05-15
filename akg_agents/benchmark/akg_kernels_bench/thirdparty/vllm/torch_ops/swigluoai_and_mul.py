import torch
import torch.nn as nn

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/activation.py:260-286
# 测试文件: tests/kernels/core/test_activation.py
# vLLM类: SwigluOAIAndMul
# 功能: OpenAI风格的SwiGLU激活函数
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        精确复制vllm的forward_native实现
        """
        gate, up = x[..., ::2], x[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        return gated_output


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        from vllm.model_executor.layers.activation import SwigluOAIAndMul
        self.layer = SwigluOAIAndMul(alpha=alpha, limit=limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def get_inputs():
    """生成测试输入"""
    num_tokens = 83
    d = 512
    dtype = torch.float16

    x = torch.randn(num_tokens, 2 * d, dtype=dtype)

    return [x]


def get_init_inputs():
    """生成初始化参数"""
    return [1.702, 7.0]  # alpha, limit

