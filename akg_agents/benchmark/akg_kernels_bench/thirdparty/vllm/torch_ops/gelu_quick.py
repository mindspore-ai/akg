import torch
import torch.nn as nn

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/activation.py:338-365
# 测试文件: tests/kernels/core/test_activation.py
# vLLM类: QuickGELU
# 功能: 快速GELU激活函数（Huggingface版本）
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        精确复制vllm的forward_native实现
        """
        return x * torch.sigmoid(1.702 * x)


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(self):
        super().__init__()
        from vllm.model_executor.layers.activation import QuickGELU
        self.layer = QuickGELU()

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

