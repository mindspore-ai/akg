"""KernelBench kernel.py 模板 — catlass conv2d 实现"""
import torch
import torch.nn as nn
import os
import subprocess


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        catlass_op_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catlass_op")
        lib_path = os.path.join(catlass_op_dir, "build", "libcatlass.so")
        if not os.path.exists(lib_path):
            subprocess.run(["bash", "build.sh"], cwd=catlass_op_dir, check=True)
        torch.ops.load_library(lib_path)

        # 如果有 nn.Module 参数，需要设置随机种子并提取参数（与 kernel_verify 模板种子一致）
        torch.manual_seed(0)
        try:
            import torch_npu
            torch_npu.npu.manual_seed(0)
        except Exception:
            pass
        # 例如：
        # conv = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size))
        # self.weight = nn.Parameter(conv.weight.clone())
        # self.bias = nn.Parameter(conv.bias.clone()) if conv.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: NCHW layout 转换 + torch.ops.catlass.<your_op>(...)
        raise NotImplementedError


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3


def get_inputs():
    x = torch.randn(batch_size, in_channels, 256, 256)
    return [x]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
