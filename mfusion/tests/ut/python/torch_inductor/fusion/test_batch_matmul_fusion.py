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

"""UT for FuseBatchMatMul: transpose elimination (permute into trans) and BatchMatMul 2D -> MatMul."""

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

# Transpose (swap last two dims) + mm: x [2,4] -> transpose -> [4,2], mm([4,2], [2,8]) = [4,8].
# At mfuse level FuseBatchMatMul eliminates permute by using permute input and trans_x1=true.
MLIR_TRANSPOSE_MM = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[2,8],f32>) -> !torch.vtensor<[4,8],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.transpose.int %arg0, %int0, %int1 : !torch.vtensor<[2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,2],f32>
    %1 = torch.aten.mm %0, %arg1 : !torch.vtensor<[4,2],f32>, !torch.vtensor<[2,8],f32> -> !torch.vtensor<[4,8],f32>
    return %1 : !torch.vtensor<[4,8],f32>
  }
}
"""

# BatchMatMul 3D: bmm [2,4,8] @ [2,8,16] -> [2,4,16]. Pipeline runs; no 2D conversion here.
MLIR_BMM_3D = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,4,8],f32>, %arg1: !torch.vtensor<[2,8,16],f32>) -> !torch.vtensor<[2,4,16],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,8,16],f32> -> !torch.vtensor<[2,4,16],f32>
    return %0 : !torch.vtensor<[2,4,16],f32>
  }
}
"""


def test_batch_matmul_fusion_transpose_mm():
    """Test transpose + mm: FuseBatchMatMul eliminates permute at mfuse level.

    Pipeline runs; result has mm/matmul (transpose may be folded into trans flag).
    """
    result = fuse_and_optimize(MLIR_TRANSPOSE_MM)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.mm") or checker.check_has_op("torch.aten.matmul"), (
        checker.error or "matmul op should exist after pipeline"
    )


def test_batch_matmul_fusion_bmm_3d():
    """Test bmm 3D: pipeline runs; BatchMatMul 2D->MatMul does not apply (inputs are 3D)."""
    result = fuse_and_optimize(MLIR_BMM_3D)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.bmm") or checker.check_has_op("torch.aten.matmul"), (
        checker.error or "bmm/matmul op should exist after pipeline"
    )
