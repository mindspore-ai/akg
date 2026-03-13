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

"""End-to-end UT for two mul ops lowered to DVM call."""

from mfusion.torch.inductor import fuse_and_optimize

import sys
from pathlib import Path
import pytest

# Add ut_utils to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ut_utils.mlir_checker import MlirChecker

MLIR_TWO_MUL = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[2],f32>, %arg1: !torch.vtensor<[2],f32>) -> !torch.vtensor<[2],f32> {
    %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>
    %1 = torch.aten.mul.Tensor %0, %arg0 : !torch.vtensor<[2],f32>, !torch.vtensor<[2],f32> -> !torch.vtensor<[2],f32>
    return %1 : !torch.vtensor<[2],f32>
  }
}
"""


@pytest.mark.parametrize("kernel_generator", ["dvm", "akg", "bisheng"])
def test_with_kernel_generator(kernel_generator: str):
    """Verify two mul ops lower to a DVM call with subgraph MLIR."""
    result = fuse_and_optimize(MLIR_TWO_MUL, kernel_generator=kernel_generator)

    checker = MlirChecker.parse_torch_module(result)

    # Check 1: Verify the call operator exists with correct attributes
    assert checker.check_has_torch_operator(
        f"torch.mfusion.{kernel_generator}_call__i2_o1",
        attrs={"mfusion.is_dynamic": False},
        attr_keys=["mfusion.subgraph_mlir"],
        count=1,
    ), checker.error

    # Check 2: Verify the old mfusion.subgraph attribute is not present
    assert checker.check_text_not_contains("mfusion.subgraph ="), checker.error

    # Check 3: Verify no unrealized_conversion_cast operations remain
    assert checker.check_no_op("builtin.unrealized_conversion_cast"), checker.error

    # Check 4: Verify main function exists
    assert checker.check_has_function("main"), checker.error

    # Check 5: Verify outlined fused subgraph function exists
    assert checker.check_has_function("main_fused_0_"), checker.error

    # Check 6: Verify the main function has correct op sequence
    assert checker.check_func_op_sequence("main", [
        "torch.constant.str",
        "torch.operator",
        "func.return",
    ]), checker.error

    # Check 7: Verify the fused subgraph function has correct op sequence (two muls)
    assert checker.check_func_op_sequence("main_fused_0_", [
        "torch.aten.mul.Tensor",
        "torch.aten.mul.Tensor",
        "func.return",
    ]), checker.error

    # Check 8: Verify there are exactly 2 func.func operations (main + main_fused_0_)
    assert checker.check_has_op("func.func", count=2), checker.error

    # Check 9: Verify no torch.aten.mul.Tensor in main function (they should be fused)
    main_func = checker._find_func_op("main")
    main_has_mul = any(
        op.operation.name == "torch.aten.mul.Tensor"
        for op in main_func.regions[0].blocks[0].operations
    )
    assert not main_has_mul, f"main function should not have torch.aten.mul.Tensor, but found one"

    # Check 10: Verify the subgraph function name in subgraph_mlir
    assert checker.check_text_contains('subgraph_mlir = "module {'), checker.error
    assert checker.check_text_contains('@entry'), checker.error
