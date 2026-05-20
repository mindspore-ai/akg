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
"""UT for adaptive_avg_pool2d operator."""

import textwrap
from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker


def test_ut_adaptive_avg_pool2d_basic():
    """Verify adaptive_avg_pool2d pass handles basic case."""
    text = textwrap.dedent("""
        module {
          func.func @main(%input: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32> {
            %c2 = torch.constant.int 2
            %output_size = torch.prim.ListConstruct %c2, %c2 : (!torch.int, !torch.int) -> !torch.list<int>
            %0 = torch.aten._adaptive_avg_pool2d %input, %output_size : !torch.vtensor<[1,1,4,4],f32>, !torch.list<int> -> !torch.vtensor<[1,1,2,2],f32>
            return %0 : !torch.vtensor<[1,1,2,2],f32>
          }
        }
    """)
    result = fuse_and_optimize(text)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten._adaptive_avg_pool2d", count=1), checker.error

def test_ut_adaptive_avg_pool2d_different_output_size():
    """Verify adaptive_avg_pool2d pass handles different output size."""
    text = textwrap.dedent("""
        module {
          func.func @main(%input: !torch.vtensor<[1,3,8,8],f32>) -> !torch.vtensor<[1,3,4,4],f32> {
            %c4 = torch.constant.int 4
            %output_size = torch.prim.ListConstruct %c4, %c4 : (!torch.int, !torch.int) -> !torch.list<int>
            %0 = torch.aten._adaptive_avg_pool2d %input, %output_size : !torch.vtensor<[1,3,8,8],f32>, !torch.list<int> -> !torch.vtensor<[1,3,4,4],f32>
            return %0 : !torch.vtensor<[1,3,4,4],f32>
          }
        }
    """)
    result = fuse_and_optimize(text)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten._adaptive_avg_pool2d", count=1), checker.error
