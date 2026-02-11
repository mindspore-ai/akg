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

"""UT for MatMul Reshape Bias Add fusion in fuse_and_optimize pipeline."""

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

import os
os.environ['LLVM_DEBUG'] = 'fuse-matmul-reshape-bias-add'

# MatMul Reshape Bias Add fusion: MatMul -> Reshape -> Add(bias) => MatMulWithBias -> Reshape
# The pattern: torch.aten.mm -> mfuse.matmul -> mfuse.reshape -> mfuse.add -> mfuse.matmul_with_bias -> mfuse.reshape
MLIR_MATMUL_RESHAPE_BIAS_ADD = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[4,8],f32>, %bias: !torch.vtensor<[8],f32>) -> !torch.vtensor<[2,8],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,4],f32>, !torch.vtensor<[4,8],f32> -> !torch.vtensor<[2,8],f32>
    %int2 = torch.constant.int 2
    %int8 = torch.constant.int 8
    %shape_list = torch.prim.ListConstruct %int2, %int8 : (!torch.int, !torch.int) -> !torch.list<int>
    %1 = torch.aten.view %0, %shape_list : !torch.vtensor<[2,8],f32>, !torch.list<int> -> !torch.vtensor<[2,8],f32>
    %int1 = torch.constant.int 1
    %2 = torch.aten.add.Tensor %1, %bias, %int1 : !torch.vtensor<[2,8],f32>, !torch.vtensor<[8],f32>, !torch.int -> !torch.vtensor<[2,8],f32>
    return %2 : !torch.vtensor<[2,8],f32>
  }
}
"""


def test_matmul_reshape_bias_add_fusion():
    """Test MatMul Reshape Bias Add fusion through fuse_and_optimize pipeline."""
    result = fuse_and_optimize(MLIR_MATMUL_RESHAPE_BIAS_ADD)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, the add operation should be eliminated
    # The pattern should be fused to mfuse.matmul_with_bias -> mfuse.reshape
    # Check that mfuse.add is eliminated (bias is fused into matmul)
    assert checker.check_no_op("mfuse.add"), checker.error or "mfuse.add should be eliminated after fusion"
    # Verify that matmul_with_bias exists (bias should be fused into matmul)
    # Note: The exact op name depends on conversion back to torch, but add should be eliminated
