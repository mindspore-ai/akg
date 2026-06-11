"""Direct-invoke AscendC matmul+leaky_relu wrapper example.

Expected task layout:

    kernel.py
    ascendc_op/
      CMakeLists.txt
      ...

The CMake project should build a shared library that registers
torch.ops.ascendc_examples.matmul_leakyrelu.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch_npu


class ModelNew(torch.nn.Module):
    def __init__(self, negative_slope=0.001):
        super().__init__()
        self.negative_slope = float(negative_slope)
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        build_dir = Path(__file__).with_name("ascendc_op") / "build"
        so_files = sorted(build_dir.rglob("*.so"))
        if not so_files:
            raise RuntimeError(f"no AscendC extension found under {build_dir}")
        torch.ops.load_library(str(so_files[0]))
        self._loaded = True

    def forward(self, a, b, bias):
        self._load()
        return torch.ops.ascendc_examples.matmul_leakyrelu(
            a,
            b,
            bias,
            self.negative_slope,
        )
