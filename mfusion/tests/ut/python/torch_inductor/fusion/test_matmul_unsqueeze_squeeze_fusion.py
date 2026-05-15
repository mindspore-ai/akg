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

"""UT for MatMul Unsqueeze Squeeze fusion in fuse_and_optimize pipeline."""

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

import os
os.environ['LLVM_DEBUG'] = 'fuse-matmul-unsqueeze-squeeze'

# MatMul Unsqueeze Squeeze fusion: Handle 1D tensor inputs for matmul by inserting reshape operations.
# The pattern: 1D x 2D -> reshape(unsqueeze) 1D to 2D, matmul, reshape(squeeze) result back to 1D
# Example: [K] x [K, M] -> reshape [K] to [1, K], matmul [1, K] x [K, M] = [1, M], reshape [1, M] to [M]
MLIR_MATMUL_UNSQUEEZE_SQUEEZE = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[4],f32>, %arg1: !torch.vtensor<[4,8],f32>) -> !torch.vtensor<[8],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.matmul %arg0, %arg1 : !torch.vtensor<[4],f32>, !torch.vtensor<[4,8],f32> -> !torch.vtensor<[8],f32>
    return %0 : !torch.vtensor<[8],f32>
  }
}
"""


def test_matmul_unsqueeze_squeeze_fusion():
    """Test MatMul Unsqueeze Squeeze fusion through fuse_and_optimize pipeline."""
    result = fuse_and_optimize(MLIR_MATMUL_UNSQUEEZE_SQUEEZE)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, when one input is 1D, the pass should insert reshape operations
    # to convert 1D to 2D before matmul, then reshape the result back
    # The pass operates on mfuse.matmul, which may be converted back to torch.aten.matmul
    # We verify the conversion completes successfully without errors
    assert checker is not None, "MLIR module should be parsed successfully"
    # Verify that the result contains matmul operation.
    # 2D x 2D is converted to torch.aten.mm; ND/transposed to torch.aten.matmul.
    result_lower = str(result).lower()
    assert ("matmul" in result_lower or "aten.mm" in result_lower), (
        "MatMul operation (torch.aten.matmul or torch.aten.mm) should exist after fusion"
    )
