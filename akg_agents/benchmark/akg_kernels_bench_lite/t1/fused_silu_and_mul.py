import torch
import torch.nn as nn
import torch_npu


class Model(nn.Module):
    """
    SwiGLU: silu(x) * y

    Input is a pre-concatenated tensor of shape (B, 2*H):
        combined = torch.cat([x, y], dim=-1)

    Equivalent to:
        output = silu(combined[..., :H]) * combined[..., H:]

    This is the standard SwiGLU activation used in LLaMA / Qwen / DeepSeek FFN.
    On NPU, use torch_npu.npu_swiglu(combined, dim=-1) as the fused baseline.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, combined):
        return torch_npu.npu_swiglu(combined, dim=-1)


def get_inputs():
    x = torch.randn(4096, 4096, dtype=torch.float32)
    y = torch.randn(4096, 4096, dtype=torch.float32)
    combined = torch.cat([x, y], dim=-1)
    return [combined]


def get_init_inputs():
    return []
