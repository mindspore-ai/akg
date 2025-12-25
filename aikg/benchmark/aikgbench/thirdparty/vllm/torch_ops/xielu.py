import torch
import torch.nn as nn

# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/model_executor/layers/activation.py:383-492
# 测试文件: tests/kernels/core/test_activation.py
# vLLM类: XIELU
# 功能: xIELU激活函数，参考 https://arxiv.org/abs/2411.13010
# ============================================================================


class Model(nn.Module):
    """原生PyTorch实现（从vllm的forward_native方法提取）"""

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.alpha_p = nn.Parameter(
            torch.log(
                torch.exp(torch.tensor(alpha_p_init, dtype=dtype)) - 1
            ).unsqueeze(0)
        )
        self.alpha_n = nn.Parameter(
            torch.log(
                torch.exp(torch.tensor(alpha_n_init - beta, dtype=dtype)) - 1
            ).unsqueeze(0)
        )
        self.register_buffer("beta", torch.tensor(beta, dtype=dtype))
        self.register_buffer("eps", torch.tensor(eps, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        精确复制vllm的_xielu_python实现
        """
        alpha_p = nn.functional.softplus(self.alpha_p)
        alpha_n = self.beta + nn.functional.softplus(self.alpha_n)
        return torch.where(
            x > 0,
            alpha_p * x * x + self.beta * x,
            (torch.expm1(torch.min(x, self.eps)) - x) * alpha_n + self.beta * x,
        )


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过Layer类调用）"""

    def __init__(
        self,
        alpha_p_init: float = 0.8,
        alpha_n_init: float = 0.8,
        beta: float = 0.5,
        eps: float = -1e-6,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from vllm.model_executor.layers.activation import XIELU
        self.layer = XIELU(
            alpha_p_init=alpha_p_init,
            alpha_n_init=alpha_n_init,
            beta=beta,
            eps=eps,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def get_inputs():
    """生成测试输入"""
    num_tokens = 83
    d = 512
    dtype = torch.bfloat16

    x = torch.randn(num_tokens, d, dtype=dtype)

    return [x]


def get_init_inputs():
    """生成初始化参数"""
    return [0.8, 0.8, 0.5, -1e-6, torch.bfloat16]

