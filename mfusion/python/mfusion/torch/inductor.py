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

from ._pipeline import PipelineRunner


def fuse_and_optimize(torch_dialect_str: str, kernel_generator: str = "dvm") -> str:
    """
    Fuse and optimize the given Torch dialect MLIR string.

    Args:
        torch_dialect_str: The Torch dialect MLIR string to optimize.
        kernel_generator: The kernel generator type ('dvm', 'akg' or 'bisheng'). Defaults to 'dvm'.
    """
    kernel_generator = kernel_generator.lower()
    if kernel_generator not in ["dvm", "akg", "bisheng"]:
        print(f"Warning: Invalid kernel_generator value '{kernel_generator}', using 'dvm' as default")
        kernel_generator = "dvm"

    runner = PipelineRunner.from_torch_dialect_str(torch_dialect_str)

    runner.run(
        pipeline="builtin.module(torch-fusion,canonicalize)",
        stage="Torch Fusion",
    )

    runner.run(
        pipeline="builtin.module(convert-torch-to-mfuse,convert-torch-symbol-to-mfuse,canonicalize)",
        stage="Convert Torch to Mfuse Dialect Module",
    )

    runner.run(
        pipeline="builtin.module(decompose{pattern-type=BEFORE_MANUAL_FUSION}, canonicalize)",
        stage="Decompose aclnn ops to meta ops",
    )

    runner.run(
        pipeline="builtin.module(mfuse-fusion,canonicalize)",
        stage="Mfuse Fusion",
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
        pipeline=f"builtin.module(convert-mfuse-to-dvm,convert-fused-subgraph-to-custom-call{{kernel-generator={kernel_generator}}},canonicalize)",
        stage="Lower Fused Subgraphs to CustomOp Calls",
    )

    runner.run(
        pipeline="builtin.module(convert-mfuse-to-torch,reconcile-unrealized-casts,canonicalize)",
        stage="Convert Mfuse to Torch Dialect Module",
    )

    return str(runner.module)
