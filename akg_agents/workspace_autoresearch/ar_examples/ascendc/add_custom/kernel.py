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
        build_dir = Path(__file__).with_name("ascendc_op") / "build"
        preferred = build_dir / "libadd_custom_ops.so"
        so_path = preferred if preferred.is_file() else None
        if so_path is None:
            matches = sorted(build_dir.rglob("*.so"))
            if matches:
                so_path = matches[0]
        if so_path is None:
            raise RuntimeError(f"no AscendC extension found under {build_dir}")
        torch.ops.load_library(str(so_path))
        self._loaded = True

    def forward(self, x, y):
        self._load()
        return torch.ops.npu.add_custom(x, y)
