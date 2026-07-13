from __future__ import annotations

from pathlib import Path

import torch
try:
    import torch_npu  # noqa: F401
except ModuleNotFoundError:
    torch_npu = None  # type: ignore[assignment]


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        build_dir = Path(__file__).with_name("ascendc_op") / "build"
        preferred = build_dir / "libcannbench_ext_grouped_matmul_ops.so"
        so_path = preferred if preferred.is_file() else None
        if so_path is None:
            matches = sorted(build_dir.rglob("*.so"))
            if matches:
                so_path = matches[0]
        if so_path is None:
            raise RuntimeError(f"no AscendC extension found under {build_dir}")
        torch.ops.load_library(str(so_path))
        # Full fp32 Cube accumulation (not HF32) so fp32-input cases keep full
        # width before the chunked, compensated matmul.
        if torch_npu is not None:
            torch.npu.matmul.allow_hf32 = False
        self._loaded = True

    def forward(self, x, weight, bias=None, group_list=None, split_item=0, transpose_weight=False):
        self._load()
        return torch.ops.npu.cannbench_ext_grouped_matmul(x, weight, bias, group_list, split_item, transpose_weight)
