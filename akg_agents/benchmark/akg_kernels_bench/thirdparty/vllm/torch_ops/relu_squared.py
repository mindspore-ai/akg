import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/activation.py:368-380
# 测试文件: tests/kernels/core/test_activation.py
# vLLM类: ReLUSquaredActivation
# 功能: ReLU²激活函数，参考 https://arxiv.org/abs/2109.08668v2
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        精确复制vllm的forward_native实现
        计算: relu(x)²
        """
        return torch.square(F.relu(x))


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(self):
        super().__init__()
        from vllm.model_executor.layers.activation import ReLUSquaredActivation
        self.layer = ReLUSquaredActivation()

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

