from __future__ import annotations

from pathlib import Path

import torch
import torch_npu  # noqa: F401


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        so_path = Path(__file__).with_name("catlass_op") / "build" / "libcatlass.so"
        if not so_path.is_file():
            raise RuntimeError(f"CATLASS library not found: {so_path}")
        torch.ops.load_library(str(so_path))
        self._loaded = True

    def forward(self, x, y):
        self._load()
        return torch.ops.catlass.basic_matmul(x, y)
