import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/activation.py:142-195
# 测试文件: tests/kernels/core/test_activation.py
# vLLM类: GeluAndMulSparse
# 功能: 带稀疏性的GeluAndMul，用于Gemma3n模型
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(self, activation_sparsity: float, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate
        # 计算std_multiplier
        target_sparsity_tensor = torch.tensor(activation_sparsity, dtype=torch.float32)
        normal_dist = torch.distributions.normal.Normal(0, 1)
        self.std_multiplier = normal_dist.icdf(target_sparsity_tensor)

    def _gaussian_topk(self, x: torch.Tensor) -> torch.Tensor:
        """获取高斯分布的稀疏百分位"""
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        cutoff_x = mean + std * self.std_multiplier
        return F.relu(x - cutoff_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        精确复制vllm的forward_native实现
        """
        d = x.shape[-1] // 2
        out = self._gaussian_topk(x[..., :d])
        out = F.gelu(out, approximate=self.approximate)
        return out * x[..., d:]


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(self, activation_sparsity: float, approximate: str = "none"):
        super().__init__()
        from vllm.model_executor.layers.activation import GeluAndMulSparse
        self.layer = GeluAndMulSparse(
            activation_sparsity=activation_sparsity, approximate=approximate
        )

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
    return [0.5, "none"]  # activation_sparsity, approximate

