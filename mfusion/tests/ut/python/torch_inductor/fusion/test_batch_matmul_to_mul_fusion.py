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

"""UT for BatchMatMul to Mul fusion in fuse_and_optimize pipeline."""

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

import os
os.environ['LLVM_DEBUG'] = 'fuse-batchmatmul-to-mul'

# Test case 1: Matmul with k=1 (f32)
MLIR_MATMUL_TO_MUL_F32 = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[3, 1],f32>, %arg1: !torch.vtensor<[1, 4],f32>) -> !torch.vtensor<[3, 4],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3, 1],f32>, !torch.vtensor<[1, 4],f32> -> !torch.vtensor<[3, 4],f32>
    return %0 : !torch.vtensor<[3, 4],f32>
  }
}
"""

# Test case 2: Matmul with transpose (bf16)
MLIR_MATMUL_TO_MUL_WITH_TRANSPOSE = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1, 3],bf16>, %arg1: !torch.vtensor<[1, 4],bf16>) -> !torch.vtensor<[3, 4],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.transpose.int %arg0, %int0, %int1 : !torch.vtensor<[1, 3],bf16>, !torch.int, !torch.int -> !torch.vtensor<[3, 1],bf16>
    %1 = torch.aten.mm %0, %arg1 : !torch.vtensor<[3, 1],bf16>, !torch.vtensor<[1, 4],bf16> -> !torch.vtensor<[3, 4],bf16>
    return %1 : !torch.vtensor<[3, 4],bf16>
  }
}
"""

# Test case 3: Matmul with k=1 (f16)
MLIR_MATMUL_TO_MUL_F16 = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[3, 1],f16>, %arg1: !torch.vtensor<[1, 4],f16>) -> !torch.vtensor<[3, 4],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[3, 1],f16>, !torch.vtensor<[1, 4],f16> -> !torch.vtensor<[3, 4],f16>
    return %0 : !torch.vtensor<[3, 4],f16>
  }
}
"""

# Test case 4: Matmul with transpose on both inputs (bf16)
MLIR_MATMUL_TO_MUL_WITH_DOUBLE_TRANSPOSE = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1, 3],bf16>, %arg1: !torch.vtensor<[4, 1],bf16>) -> !torch.vtensor<[3, 4],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %0 = torch.aten.transpose.int %arg0, %int0, %int1 : !torch.vtensor<[1, 3],bf16>, !torch.int, !torch.int -> !torch.vtensor<[3, 1],bf16>
    %1 = torch.aten.transpose.int %arg1, %int0, %int1 : !torch.vtensor<[4, 1],bf16>, !torch.int, !torch.int -> !torch.vtensor<[1, 4],bf16>
    %2 = torch.aten.mm %0, %1 : !torch.vtensor<[3, 1],bf16>, !torch.vtensor<[1, 4],bf16> -> !torch.vtensor<[3, 4],bf16>
    return %2 : !torch.vtensor<[3, 4],bf16>
  }
}
"""

# Test case 5: BatchMatmul with rank=3 and k=1: [B, N, 1] @ [B, 1, M] -> [B, N, M].
# Guards FuseBatchMatMulToMul when input rank > 2 (k dimension uses last-two indices).
MLIR_BATCH_MATMUL_RANK3_K1_TO_MUL = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2, 3, 1],f32>, %arg1: !torch.vtensor<[2, 1, 4],f32>) -> !torch.vtensor<[2, 3, 4],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[2, 3, 1],f32>, !torch.vtensor<[2, 1, 4],f32> -> !torch.vtensor<[2, 3, 4],f32>
    return %0 : !torch.vtensor<[2, 3, 4],f32>
  }
}
"""


def test_matmul_to_mul_fusion_f32():
    """Test Matmul to Mul fusion with Float32."""
    result = fuse_and_optimize(MLIR_MATMUL_TO_MUL_F32)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, the mm should be replaced with mul
    assert checker.check_no_op("torch.aten.mm"), checker.error or "torch.aten.mm should be eliminated after fusion"
    # Check that mul operation exists
    assert checker.check_has_op("torch.aten.mul.Tensor"), checker.error or "torch.aten.mul.Tensor should exist after fusion"


def test_matmul_to_mul_fusion_f16():
    """Test Matmul to Mul fusion with Float16."""
    result = fuse_and_optimize(MLIR_MATMUL_TO_MUL_F16)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, the mm should be replaced with mul
    assert checker.check_no_op("torch.aten.mm"), checker.error or "torch.aten.mm should be eliminated after fusion"
    # Check that mul operation exists
    assert checker.check_has_op("torch.aten.mul.Tensor"), checker.error or "torch.aten.mul.Tensor should exist after fusion"


def test_matmul_to_mul_fusion_with_transpose():
    """Test Matmul to Mul fusion with transpose."""
    result = fuse_and_optimize(MLIR_MATMUL_TO_MUL_WITH_TRANSPOSE)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, the mm should be replaced with mul
    assert checker.check_no_op("torch.aten.mm"), checker.error or "torch.aten.mm should be eliminated after fusion"
    # Check that mul operation exists
    assert checker.check_has_op("torch.aten.mul.Tensor"), checker.error or "torch.aten.mul.Tensor should exist after fusion"


def test_matmul_to_mul_fusion_with_double_transpose():
    """Test Matmul to Mul fusion with transpose on both inputs."""
    result = fuse_and_optimize(MLIR_MATMUL_TO_MUL_WITH_DOUBLE_TRANSPOSE)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, the mm should be replaced with mul
    assert checker.check_no_op("torch.aten.mm"), checker.error or "torch.aten.mm should be eliminated after fusion"
    # Check that mul operation exists
    assert checker.check_has_op("torch.aten.mul.Tensor"), checker.error or "torch.aten.mul.Tensor should exist after fusion"


def test_batch_matmul_rank3_k1_to_mul_fusion():
    """Test BatchMatmul (rank=3, k=1) to Mul fusion. Guards rank>2 k-dimension logic."""
    result = fuse_and_optimize(MLIR_BATCH_MATMUL_RANK3_K1_TO_MUL)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, bmm should be replaced with mul
    assert checker.check_no_op("torch.aten.bmm"), checker.error or "torch.aten.bmm should be eliminated after fusion"
    assert checker.check_has_op("torch.aten.mul.Tensor"), checker.error or "torch.aten.mul.Tensor should exist after fusion"
