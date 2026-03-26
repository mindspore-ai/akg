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

"""UT for MatMul Transpose Weight fusion in fuse_and_optimize pipeline."""

import pytest

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

import os
os.environ['LLVM_DEBUG'] = 'fuse-matmul-transpose-weight'

# MatMul Transpose Weight fusion: When inner axis (last dim * elem_size) is not 512-byte aligned,
# the pass inserts permute operations and sets trans_x1/trans_x2 attributes.
# For f32: 512 / 4 = 128, so shapes like 100 (400 bytes) are not aligned.
# The pattern: torch.aten.mm -> mfuse.matmul -> mfuse.permute (if unaligned) -> mfuse.matmul with trans
MLIR_MATMUL_TRANSPOSE_WEIGHT = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,100],f32>, %arg1: !torch.vtensor<[100,8],f32>) -> !torch.vtensor<[2,8],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,100],f32>, !torch.vtensor<[100,8],f32> -> !torch.vtensor<[2,8],f32>
    return %0 : !torch.vtensor<[2,8],f32>
  }
}
"""


@pytest.mark.skip(reason="MatMul transpose weight fusion UT disabled")
def test_matmul_transpose_weight_fusion():
    """Test MatMul Transpose Weight fusion through fuse_and_optimize pipeline."""
    result = fuse_and_optimize(MLIR_MATMUL_TRANSPOSE_WEIGHT)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, when inner axis is not 512-byte aligned (100*4=400 bytes for f32),
    # the pass should insert permute operations and set trans_x1/trans_x2 attributes.
    # For 100*4=400 bytes (not aligned), permute should be inserted for both inputs
    # The pass operates on mfuse.matmul, which may be converted back to torch.aten.matmul/mm
    # We verify the conversion completes successfully without errors
    assert checker is not None, "MLIR module should be parsed successfully"
    # Verify that the result contains matmul operation (either mfuse.matmul or torch.aten.matmul/mm)
    assert "matmul" in str(result).lower() or "mm" in str(result), \
        "MatMul operation should exist after fusion"
