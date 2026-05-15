import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/activation.py:25-58
# 测试文件: tests/kernels/core/test_activation.py
# vLLM类: FatreluAndMul
# 功能: 计算 x -> FATReLU(x[:d]) * x[d:], 用于MiniCPM模型
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        精确复制vllm的forward_native实现
        计算: threshold(x[:d]) * x[d:]
        """
        d = x.shape[-1] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        x1 = F.threshold(x1, self.threshold, 0.0)
        return x1 * x2


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        from vllm.model_executor.layers.activation import FatreluAndMul
        self.layer = FatreluAndMul(threshold=threshold)

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
    return [0.0]  # threshold

