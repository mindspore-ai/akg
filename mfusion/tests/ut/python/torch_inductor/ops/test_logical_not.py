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

"""UT for logical_not operator."""

import textwrap
from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker

def test_logical_not_int():
    """Test logical not with int tensor input."""
    # Torch dialect MLIR string with logical_not operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_logical_not(%arg0: !torch.vtensor<[4,4],si32>) -> !torch.vtensor<[4,4],i1> {
            %0 = torch.aten.logical_not %arg0 : !torch.vtensor<[4,4],si32> -> !torch.vtensor<[4,4],i1>
            return %0 : !torch.vtensor<[4,4],i1>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.logical_not"), checker.error
