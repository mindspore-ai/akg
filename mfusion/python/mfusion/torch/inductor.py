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

import os
from mfusion import ir
from mfusion.passmanager import PassManager
from mfusion.dialects import torch as torch_d
from mfusion import register_muse_dialect

def _print_verbose(stage_title: str, content):
    """Print verbose information with formatting.

    Args:
        stage_title: Title of the stage
        content: Content to print
    """
    enabled = os.environ.get("MFUSION_PRINT_IR") == "1"
    if not enabled:
        return
    print("=" * 80)
    print(stage_title)
    print("=" * 80)
    print(content)
    print()

def _parse_mlir_module_from_text(text: str):
    """Parse MLIR module from text IR."""
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    register_muse_dialect(ctx)
    return ir.Module.parse(text, ctx)

def _run_pipeline(module: ir.Module, pipeline: str, stage: str) -> ir.Module:
    """Run an MLIR pass pipeline and optionally dump the IR."""
    with module.context:
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)
    _print_verbose(stage, module)

def fuse_and_optimize(torch_dialect_str: str) -> str:
    """
    Fuse and optimize the given Torch dialect MLIR string.
    """
    module =_parse_mlir_module_from_text(torch_dialect_str)
    _print_verbose("Original MLIR Module", module)

    _run_pipeline(module,
        pipeline="builtin.module(convert-torch-to-muse,canonicalize)",
        stage="Convert Torch to Muse Dialect Module")

    _run_pipeline(module,
        pipeline="builtin.module(convert-muse-to-torch,reconcile-unrealized-casts,canonicalize)",
        stage="Convert Muse to Torch Dialect Module")

    return str(module)
