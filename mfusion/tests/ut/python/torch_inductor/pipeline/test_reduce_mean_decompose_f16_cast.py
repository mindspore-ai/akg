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

"""Pipeline tests for f16/bf16 reduce_mean decompose via f32 sum+div."""

import textwrap

from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker


def _gap_mean_dim_mlir(dtype: str) -> str:
    return textwrap.dedent(
        f"""
        module {{
          func.func @gap_mean(%x: !torch.vtensor<[2,4,7,7],{dtype}>) -> !torch.vtensor<[2,4,1,1],{dtype}> {{
            %none = torch.constant.none
            %true = torch.constant.bool true
            %c2 = torch.constant.int 2
            %c3 = torch.constant.int 3
            %dims = torch.prim.ListConstruct %c2, %c3 : (!torch.int, !torch.int) -> !torch.list<int>
            %pool = torch.aten.mean.dim %x, %dims, %true, %none
                : !torch.vtensor<[2,4,7,7],{dtype}>, !torch.list<int>, !torch.bool, !torch.none
                -> !torch.vtensor<[2,4,1,1],{dtype}>
            return %pool : !torch.vtensor<[2,4,1,1],{dtype}>
          }}
        }}
    """
    )


def _assert_low_precision_mean_decomposed(checker: MlirChecker) -> None:
    """mean.dim becomes f32 cast + sum + div + cast, not aten.mean.dim."""
    assert checker.check_no_op("torch.aten.mean.dim"), checker.error
    assert checker.check_text_contains("torch.prims.convert_element_type"), checker.error
    ir = str(checker.module)
    assert ir.count("torch.prims.convert_element_type") >= 2, (
        "expected cast to f32 before reduce and cast back after div"
    )
    assert checker.check_text_contains("torch.aten.sum.dim_IntList"), checker.error
    has_div = checker.check_text_contains("torch.aten.div.Tensor") or checker.check_text_contains(
        "torch.aten.div.Scalar"
    )
    assert has_div, checker.error


def test_f16_mean_dim_decomposes_to_sum_div_not_mean():
    """f16 mean.dim lowers to sum+div with f32 compute after mfusion decompose."""
    result = fuse_and_optimize(_gap_mean_dim_mlir("f16"))
    _assert_low_precision_mean_decomposed(MlirChecker.parse_torch_module(result))


def test_bf16_mean_dim_decomposes_to_sum_div_not_mean():
    """bf16 mean.dim lowers to sum+div with f32 compute after mfusion decompose."""
    result = fuse_and_optimize(_gap_mean_dim_mlir("bf16"))
    _assert_low_precision_mean_decomposed(MlirChecker.parse_torch_module(result))
