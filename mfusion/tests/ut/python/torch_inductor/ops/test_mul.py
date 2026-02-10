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

"""UT for mul operator."""

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

MLIR_MUL = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2,2],f32>, %arg1: !torch.vtensor<[2,2],f32>) -> !torch.vtensor<[2,2],f32> {
    %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[2,2],f32>, !torch.vtensor<[2,2],f32> -> !torch.vtensor<[2,2],f32>
    return %0 : !torch.vtensor<[2,2],f32>
  }
}
"""

def test_mul():
    """Verify mul op is preserved after fuse and optimize."""
    result = fuse_and_optimize(MLIR_MUL)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.mul.Tensor", count=1), checker.error
    assert checker.check_no_op("builtin.unrealized_conversion_cast"), checker.error


MLIR_SYMBOLIC_MUL = r"""
module {
  func.func @main(%arg0: !torch.int, %arg1: !torch.vtensor<[2,?],f32>, %arg2: !torch.vtensor<[2,?],f32>) -> !torch.vtensor<[2,?],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.symbolic_int "s10" {min_val = 2, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg1, [%0], affine_map<()[s10] -> (2, s10)> : !torch.vtensor<[2,?],f32>
    torch.bind_symbolic_shape %arg2, [%0], affine_map<()[s10] -> (2, s10)> : !torch.vtensor<[2,?],f32>
    %1 = torch.aten.mul.Tensor %arg2, %arg1 : !torch.vtensor<[2,?],f32>, !torch.vtensor<[2,?],f32> -> !torch.vtensor<[2,?],f32>
    torch.bind_symbolic_shape %1, [%0], affine_map<()[s10] -> (2, s10)> : !torch.vtensor<[2,?],f32>
    return %1 : !torch.vtensor<[2,?],f32>
  }
}
"""

def test_mul_dynamic_shape():
    """Verify symbolic mul lowers to mfuse and roundtrips."""
    result = fuse_and_optimize(MLIR_SYMBOLIC_MUL)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.mul.Tensor", count=1), checker.error
    assert checker.check_no_op("torch.bind_symbolic_shape"), checker.error
    assert checker.check_no_op("mfuse.mul"), checker.error
    assert checker.check_no_op("builtin.unrealized_conversion_cast"), checker.error
