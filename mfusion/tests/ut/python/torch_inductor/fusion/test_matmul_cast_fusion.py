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

"""UT for MatMul Cast fusion in fuse_and_optimize pipeline."""

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

import os
os.environ['LLVM_DEBUG'] = 'fuse-matmul-cast'

# MatMul Cast fusion: MatMul (f16) + Cast (f32) -> MatMul (f32 output)
# The pattern: torch.aten.mm (f16) -> mfuse.aclnn.mm (f16) -> mfuse.matmul (f16) + mfuse.cast (f32) -> mfuse.matmul (f32)
MLIR_MATMUL_CAST = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[4,8],f16>) -> !torch.vtensor<[2,8],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[4,8],f16> -> !torch.vtensor<[2,8],f16>
    %int6 = torch.constant.int 6
    %false = torch.constant.bool false
    %none = torch.constant.none
    %1 = torch.aten.to.dtype %0, %int6, %false, %false, %none : !torch.vtensor<[2,8],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[2,8],f32>
    return %1 : !torch.vtensor<[2,8],f32>
  }
}
"""


def test_matmul_cast_fusion():
    """Test MatMul Cast fusion through fuse_and_optimize pipeline."""
    result = fuse_and_optimize(MLIR_MATMUL_CAST)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, the cast operation should be eliminated
    # The matmul should directly produce f32 output instead of f16 + cast to f32
    # Check that cast operations are eliminated (both mfuse.cast and torch.aten.to.dtype)
    assert checker.check_no_op("mfuse.cast"), checker.error or "mfuse.cast should be eliminated after fusion"
    assert checker.check_no_op("torch.aten.to.dtype"), checker.error or "torch.aten.to.dtype should be eliminated after fusion"
