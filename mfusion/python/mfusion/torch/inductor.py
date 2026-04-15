# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mfuse fusion pipelines for Torch inductor."""

import re
from pathlib import Path
from typing import Optional

from ._pipeline import PipelineRunner

_PASS_ENTRY_PATTERN = re.compile(
    r'\{\s*"([^"]+)"\s*,\s*\[\]\(\)\s*\{\s*return\s+create.*?\(\);\s*\}\s*\}',
    re.DOTALL,
)


def _load_internal_passes_from_cpp(
    relative_cpp_path: str,
    base_dir: Optional[Path] = None,
) -> tuple[str, ...]:
    """Read pass order from C++ composite pass table."""
    mfusion_root = base_dir if base_dir is not None else Path(__file__).resolve().parents[3]
    cpp_path = mfusion_root / relative_cpp_path
    try:
        text = cpp_path.read_text(encoding="utf-8")
    except OSError:
        return ()

    passes = tuple(_PASS_ENTRY_PATTERN.findall(text))
    return passes


_TORCH_FUSION_INTERNAL_PASSES = _load_internal_passes_from_cpp(
    "lib/Conversion/TorchToMfuse/TorchFusion.cc",
)

_MFUSE_FUSION_INTERNAL_PASSES = _load_internal_passes_from_cpp(
    "lib/Dialect/Mfuse/Transforms/Fusion/MfuseFusion.cc",
)


def _run_composite_fusion_stage(
    runner: PipelineRunner,
    stage_label: str,
    composite_pipeline: str,
    internal_passes: tuple[str, ...],
) -> None:
    """Run torch-fusion or mfuse-fusion: one PM when quiet, per-pass when verbose IR is enabled."""
    if runner.enabled_verbose_internal_ir:
        for name in internal_passes:
            runner.run(f"builtin.module({name})", f"{stage_label} / {name}")
        runner.run("builtin.module(canonicalize)", f"{stage_label} / canonicalize")
    else:
        runner.run(composite_pipeline, stage_label)


def fuse_and_optimize(torch_dialect_str: str, kernel_generator: str = "dvm") -> str:
    """
    Fuse and optimize the given Torch dialect MLIR string.

    Args:
        torch_dialect_str: The Torch dialect MLIR string to optimize.
        kernel_generator: The kernel generator type ('dvm', 'akg' or 'bisheng'). Defaults to 'dvm'.

    Observability (see ``mfusion.torch._pipeline``): ``MFUSION_PRINT_IR`` / ``MFUSION_SAVE_IR``
    set to ``1`` enables stage-level print/save; ``2`` additionally runs ``torch-fusion`` and
    ``mfuse-fusion`` as separate sub-passes so IR is printed/saved after each internal pass.
    With level ``2``, ``MFUSION_VERBOSE_IR_DUMP_ON_CHANGE=1`` restricts those internal sub-pass
    dumps to steps where the IR actually changes. Optional custom files can use
    ``get_verbose_ir_directory()`` from ``mfusion.torch._pipeline``.
    """
    kernel_generator = kernel_generator.lower()
    if kernel_generator not in ["dvm", "akg", "bisheng"]:
        print(f"Warning: Invalid kernel_generator value '{kernel_generator}', using 'dvm' as default")
        kernel_generator = "dvm"

    runner = PipelineRunner.from_torch_dialect_str(torch_dialect_str)

    _run_composite_fusion_stage(
        runner,
        "Torch Fusion",
        "builtin.module(torch-fusion,canonicalize)",
        _TORCH_FUSION_INTERNAL_PASSES,
    )

    runner.run(
        pipeline="builtin.module(convert-torch-to-mfuse,convert-torch-symbol-to-mfuse,canonicalize)",
        stage="Convert Torch to Mfuse Dialect Module",
    )

    runner.run(
        pipeline="builtin.module(decompose{pattern-type=BEFORE_MANUAL_FUSION}, canonicalize)",
        stage="Decompose aclnn ops to meta ops",
    )

    _run_composite_fusion_stage(
        runner,
        "Mfuse Fusion",
        "builtin.module(mfuse-fusion,canonicalize)",
        _MFUSE_FUSION_INTERNAL_PASSES,
    )

    runner.run(
        pipeline="builtin.module(decompose{pattern-type=AFTER_MANUAL_FUSION}, canonicalize)",
        stage="Decompose complex ops to meta ops",
    )

    runner.run(
        pipeline=f"builtin.module(func.func(mfuse-{kernel_generator}-cluster),canonicalize)",
        stage=f"Mfuse {kernel_generator} Clustering",
    )

    runner.run(
        pipeline=f"builtin.module(func.func(split{{kernel-generator={kernel_generator}}}),canonicalize)",
        stage="Mfuse Split",
    )

    runner.run(
        pipeline="builtin.module(recompose, canonicalize)",
        stage="Recompose meta ops to aclnn ops",
    )

    runner.run(
        pipeline="builtin.module(outline-mfuse-fused-subgraphs,copy-fused-subgraphs,canonicalize)",
        stage="Outline Fused Subgraphs",
    )

    runner.run(
        pipeline=(
            f"builtin.module(convert-mfuse-to-dvm,convert-fused-subgraph-to-custom-call"
            f"{{kernel-generator={kernel_generator}}},canonicalize)"
        ),
        stage="Lower Fused Subgraphs to CustomOp Calls",
    )

    runner.run(
        pipeline=(
            f"builtin.module(convert-mfuse-to-torch{{kernel-generator={kernel_generator}}},"
            f"reconcile-unrealized-casts,canonicalize)"
        ),
        stage="Convert Mfuse to Torch Dialect Module",
    )

    return str(runner.module)
