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
"""UT for disable aten full/ones/zeros fold: verify they are not folded to torch.vtensor.literal."""

import pytest

from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker

MLIR_ATEN_FULL = r"""
module {
  func.func @main() -> !torch.vtensor<[2, 3], f32> {
    %c2 = torch.constant.int 2
    %c3 = torch.constant.int 3
    %size = torch.prim.ListConstruct %c2, %c3 : (!torch.int, !torch.int) -> !torch.list<int>
    %fill_value = torch.constant.float 5.000000e+00
    %none = torch.constant.none
    %false = torch.constant.bool 0
    %0 = torch.aten.full %size, %fill_value, %none, %none, %none, %false : !torch.list<int>, !torch.float, !torch.none, !torch.none, !torch.none, !torch.bool -> !torch.vtensor<[2, 3], f32>
    return %0 : !torch.vtensor<[2, 3], f32>
  }
}
"""

MLIR_ATEN_ONES = r"""
module {
  func.func @main() -> !torch.vtensor<[3, 4], f32> {
    %c3 = torch.constant.int 3
    %c4 = torch.constant.int 4
    %size = torch.prim.ListConstruct %c3, %c4 : (!torch.int, !torch.int) -> !torch.list<int>
    %dtype = torch.constant.int 6
    %none = torch.constant.none
    %0 = torch.aten.ones %size, %dtype, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[3, 4], f32>
    return %0 : !torch.vtensor<[3, 4], f32>
  }
}
"""

MLIR_ATEN_ZEROS = r"""
module {
  func.func @main() -> !torch.vtensor<[4, 5], f32> {
    %c4 = torch.constant.int 4
    %c5 = torch.constant.int 5
    %size = torch.prim.ListConstruct %c4, %c5 : (!torch.int, !torch.int) -> !torch.list<int>
    %dtype = torch.constant.int 6
    %none = torch.constant.none
    %0 = torch.aten.zeros %size, %dtype, %none, %none, %none : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[4, 5], f32>
    return %0 : !torch.vtensor<[4, 5], f32>
  }
}
"""


@pytest.mark.parametrize(
    "mlir_module,expected_op",
    [
        (MLIR_ATEN_FULL, "torch.aten.full"),
        (MLIR_ATEN_ONES, "torch.aten.ones"),
        (MLIR_ATEN_ZEROS, "torch.aten.zeros"),
    ],
)
def test_aten_fill_ops_not_folded(mlir_module, expected_op):
    """Verify aten full/ones/zeros are preserved after fuse_and_optimize (not folded to vtensor.literal)."""
    result = fuse_and_optimize(mlir_module)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op(expected_op, count=1), checker.error
    assert checker.check_no_op("torch.vtensor.literal"), checker.error

