import math

import torch
import torch.nn as nn

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/activation.py:289-311
# 测试文件: tests/kernels/core/test_activation.py
# vLLM类: NewGELU
# 功能: 新版GELU激活函数
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        精确复制vllm的forward_native实现
        """
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(self):
        super().__init__()
        from vllm.model_executor.layers.activation import NewGELU
        self.layer = NewGELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def get_inputs():
    """生成测试输入"""
    num_tokens = 83
    d = 512
    dtype = torch.float16

    x = torch.randn(num_tokens, d, dtype=dtype)

    return [x]


def get_init_inputs():
    """生成初始化参数"""
    return []

