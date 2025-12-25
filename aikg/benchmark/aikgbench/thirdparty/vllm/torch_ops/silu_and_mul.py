import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/activation.py:61-101
# 测试文件: tests/kernels/core/test_activation.py
# vLLM类: SiluAndMul
# 功能: 计算 x -> silu(x[:d]) * x[d:], 其中 d = x.shape[-1] // 2
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        精确复制vllm的forward_native实现
        计算: silu(x[:d]) * x[d:]
        """
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(self):
        super().__init__()
        from vllm.model_executor.layers.activation import SiluAndMul
        self.layer = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def get_inputs():
    """
    生成测试输入
    参考: tests/kernels/core/test_activation.py
    NUM_TOKENS = [7, 83, 2048], D = [512, 13824]
    """
    num_tokens = 83
    d = 512
    dtype = torch.float16

    # 输入形状为 (num_tokens, 2*d)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)

    return [x]


def get_init_inputs():
    """生成初始化参数"""
    return []

