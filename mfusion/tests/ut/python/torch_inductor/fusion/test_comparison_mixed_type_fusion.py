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

"""End-to-end UT for mixed-type comparison fusion."""

import textwrap

from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker
# pylint: disable=line-too-long

MLIR_MUL_THEN_GT_MIXED_TYPES = textwrap.dedent(
    r"""
    module {
      func.func @main(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f16>, %arg2: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],i1> attributes {torch.assume_strict_symbolic_shapes} {
        %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f16> -> !torch.vtensor<[2,4],f16>
        %1 = torch.aten.gt.Tensor %0, %arg2 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],i1>
        return %1 : !torch.vtensor<[2,4],i1>
      }
    }
    """
)


def test_comparison_mixed_type_chain_is_fused_end_to_end():
    """Verify a mixed-type comparison chain is fused and promoted inside the outlined subgraph."""
    result = fuse_and_optimize(MLIR_MUL_THEN_GT_MIXED_TYPES)
    checker = MlirChecker.parse_torch_module(result)

    assert checker.check_has_torch_operator(
        "torch.mfusion.dvm_call__i3_o1",
        attrs={"mfusion.is_dynamic": False},
        attr_keys=["mfusion.subgraph_mlir"],
        count=1,
    ), checker.error
    assert checker.check_has_function("main_fused_0_"), checker.error
    assert checker.check_no_op("builtin.unrealized_conversion_cast"), checker.error

    assert checker.check_func_op_sequence(
        "main",
        [
            "torch.constant.str",
            "torch.operator",
            "func.return",
        ],
    ), checker.error

    assert checker.check_func_op_sequence(
        "main_fused_0_",
        [
            "torch.constant.int",
            "torch.aten.mul.Tensor",
            "torch.prims.convert_element_type",
            "torch.aten.gt.Tensor",
            "func.return",
        ],
    ), checker.error

    assert checker.check_text_contains(
        "torch.prims.convert_element_type"
    ), checker.error
    assert checker.check_text_contains("dvm.binary Greater"), checker.error
