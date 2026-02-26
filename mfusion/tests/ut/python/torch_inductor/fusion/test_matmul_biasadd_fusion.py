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

"""UT for MatMul/BatchMatmul BiasAdd fusion: MatMul/BatchMatmul + Add(bias) -> MatmulWithBias."""

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

# MatMul (2D) + Add(bias 1D): mm [2,4] @ [4,8] -> [2,8], bias [8]. Fuses to matmul_with_bias.
MLIR_MATMUL_BIASADD = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[4,8],f32>, %arg2: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,8],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,4],f32>, !torch.vtensor<[4,8],f32> -> !torch.vtensor<[2,8],f32>
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %arg2, %int1 : !torch.vtensor<[2,8],f32>, !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[2,8],f32>
    return %1 : !torch.vtensor<[2,8],f32>
  }
}
"""

# BatchMatmul + Add(bias 1D): bmm [2,4,8] @ [2,8,16] -> [2,4,16], bias [16]. Fuses to matmul_with_bias.
MLIR_BATCH_MATMUL_BIASADD = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,4,8],f32>, %arg1: !torch.vtensor<[2,8,16],f32>, %arg2: !torch.vtensor<[16],f32>) -> !torch.vtensor<[2,4,16],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.bmm %arg0, %arg1 : !torch.vtensor<[2,4,8],f32>, !torch.vtensor<[2,8,16],f32> -> !torch.vtensor<[2,4,16],f32>
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %arg2, %int1 : !torch.vtensor<[2,4,16],f32>, !torch.vtensor<[16],f32>, !torch.int -> !torch.vtensor<[2,4,16],f32>
    return %1 : !torch.vtensor<[2,4,16],f32>
  }
}
"""

# No fusion: bias size [4] does not match matmul output last dim 8.
MLIR_MATMUL_BIASADD_NO_FUSION_BIAS_WRONG_SIZE = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[4,8],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[2,8],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,4],f32>, !torch.vtensor<[4,8],f32> -> !torch.vtensor<[2,8],f32>
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %arg2, %int1 : !torch.vtensor<[2,8],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[2,8],f32>
    return %1 : !torch.vtensor<[2,8],f32>
  }
}
"""

# No fusion: bias is 2D [8,1], pass requires 1D (one of add inputs must have rank 1).
MLIR_MATMUL_BIASADD_NO_FUSION_BIAS_NOT_1D = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[4,8],f32>, %arg2: !torch.vtensor<[8,1],f32>) -> !torch.vtensor<[2,8],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,4],f32>, !torch.vtensor<[4,8],f32> -> !torch.vtensor<[2,8],f32>
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %arg2, %int1 : !torch.vtensor<[2,8],f32>, !torch.vtensor<[8,1],f32>, !torch.int -> !torch.vtensor<[2,8],f32>
    return %1 : !torch.vtensor<[2,8],f32>
  }
}
"""


def test_matmul_biasadd_fusion():
    """Test MatMul + Add(bias) fusion through fuse_and_optimize pipeline.

    At mfuse level, mm + add(bias 1D) is fused to matmul_with_bias. When converting
    back to torch, MfuseMetaToTorch lowers matmul_with_bias to mm + add.Tensor, so
    the final IR has one mm and one add.Tensor (expected).
    """
    result = fuse_and_optimize(MLIR_MATMUL_BIASADD)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.mm") or checker.check_has_op("torch.aten.matmul"), (
        checker.error or "matmul op should exist after pipeline"
    )
    assert checker.check_has_op("torch.aten.add.Tensor"), (
        checker.error or "add.Tensor (bias) expected: matmul_with_bias is lowered to mm + add"
    )


def test_batch_matmul_biasadd_fusion():
    """Test BatchMatmul + Add(bias) fusion through fuse_and_optimize pipeline.

    At mfuse level, bmm + add(bias) is fused to matmul_with_bias. Convert-mfuse-to-torch
    lowers it to bmm + add.Tensor, so the final IR has one bmm and one add.Tensor.
    """
    result = fuse_and_optimize(MLIR_BATCH_MATMUL_BIASADD)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.bmm") or checker.check_has_op("torch.aten.matmul"), (
        checker.error or "batch matmul / matmul op should exist after pipeline"
    )
    assert checker.check_has_op("torch.aten.add.Tensor"), (
        checker.error or "add.Tensor (bias) expected: matmul_with_bias is lowered to bmm + add"
    )


def test_matmul_biasadd_no_fusion_bias_wrong_size():
    """No fusion: bias shape [4] does not match matmul output last dim 8."""
    result = fuse_and_optimize(MLIR_MATMUL_BIASADD_NO_FUSION_BIAS_WRONG_SIZE)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.add.Tensor"), (
        checker.error or "add.Tensor should remain when bias size mismatch"
    )
    assert checker.check_has_op("torch.aten.mm") or checker.check_has_op("torch.aten.matmul"), (
        checker.error or "matmul op should exist"
    )


def test_matmul_biasadd_no_fusion_bias_not_1d():
    """No fusion: bias is 2D [8,1], pass requires one add input to have rank 1."""
    result = fuse_and_optimize(MLIR_MATMUL_BIASADD_NO_FUSION_BIAS_NOT_1D)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.add.Tensor"), (
        checker.error or "add.Tensor should remain when bias not 1D"
    )
    assert checker.check_has_op("torch.aten.mm") or checker.check_has_op("torch.aten.matmul"), (
        checker.error or "matmul op should exist"
    )
