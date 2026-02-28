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

"""Tests for recompose pipeline."""
import textwrap
from ut_utils.mlir_checker import MlirChecker
from mfusion.torch.inductor import fuse_and_optimize


def test_recompose_pipeline_with_add():
    """Test recompose pipeline with add operations."""
    # Torch dialect MLIR string with add operations
    torch_mlir = textwrap.dedent("""
      module {
        func.func @main(%arg0: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],f32> {
          %int3 = torch.constant.int 3
          %0 = torch.aten.add.Tensor %arg0, %arg0, %int3 : !torch.vtensor<[2,3,32,32],f32>, !torch.vtensor<[2,3,32,32],f32>, !torch.int -> !torch.vtensor<[2,3,32,32],f32>
          return %0 : !torch.vtensor<[2,3,32,32],f32>
        }
      }
    """)
    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.add.Tensor"), checker.error
    assert checker.check_has_op("torch.constant.int"), checker.error
