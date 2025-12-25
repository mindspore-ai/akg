import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/layernorm.py:427-443
# vLLM类: LayerNorm
# 功能: 标准Layer归一化
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """精确复制vllm的forward实现"""
        return F.layer_norm(
            x.float(), (self.dim,), self.weight, self.bias, self.eps
        ).type_as(x)


class ModelVLLM(nn.Module):
    """vLLM实现（通过Layer类调用）"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        from vllm.model_executor.layers.layernorm import LayerNorm
        self.layer = LayerNorm(dim=dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def get_inputs():
    """生成测试输入"""
    num_tokens = 83
    dim = 2048
    dtype = torch.float16

    x = torch.randn(num_tokens, dim, dtype=dtype)

    return [x]


def get_init_inputs():
    """生成初始化参数"""
    return [2048, 1e-6]  # dim, eps

