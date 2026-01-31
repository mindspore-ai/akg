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

import textwrap
from ut_utils.mlir_checker import MlirChecker
from mfusion.torch.inductor import fuse_and_optimize

def test_decompose_pipeline_with_tanh():
    """Test decompose pipeline with tanh operation."""
    # Torch dialect MLIR string with tanh operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_tanh(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
            %0 = torch.aten.tanh %arg0 : !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
            return %0 : !torch.vtensor<[4,4],f32>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_no_op("torch.aten.tanh"), checker.error


def test_decompose_pipeline_with_add():
    """Test decompose pipeline with add"""
    # Torch dialect MLIR string with add and rmsnorm operations
    torch_mlir = textwrap.dedent("""
      module {
        func.func @test_add_rms_norm(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[1],f32>, %alpha: !torch.float) -> !torch.vtensor<[4,4],f32> {
          %0 = torch.aten.add.Tensor %arg0, %arg1, %alpha : !torch.vtensor<[4,4],f32>, !torch.vtensor<[1],f32>, !torch.float -> !torch.vtensor<[4,4],f32>
          return %0 : !torch.vtensor<[4,4],f32>
        }
      }
    """)
    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.add.Tensor"), checker.error
    assert checker.check_no_op("torch.aten.mul.Tensor"), checker.error


def test_decompose_pipeline_with_add_2():
    """Test decompose pipeline with add"""
    # Torch dialect MLIR string with add and rmsnorm operations
    torch_mlir = textwrap.dedent("""
      module {
        func.func @test_add_rms_norm(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[1],f32>) -> !torch.vtensor<[4,4],f32> {
          %alpha = torch.constant.float 1.0
          %0 = torch.aten.add.Tensor %arg0, %arg1, %alpha : !torch.vtensor<[4,4],f32>, !torch.vtensor<[1],f32>, !torch.float -> !torch.vtensor<[4,4],f32>
          return %0 : !torch.vtensor<[4,4],f32>
        }
      }
    """)
    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.add.Tensor"), checker.error
    assert checker.check_no_op("torch.aten.mul.Tensor"), checker.error


def test_decompose_pipeline_with_add_rmsnorm():
    """Test decompose pipeline with add and rmsnorm operations for fusion."""
    # Torch dialect MLIR string with add and rmsnorm operations
    torch_mlir = textwrap.dedent("""
      module {
        func.func @test_add_rms_norm(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4,4],f32>, %arg2: !torch.vtensor<[4],f32>) -> (!torch.vtensor<[4,4],f32>, !torch.vtensor<[4,1],f32>) {
          %alpha = torch.constant.float 1.0
          %eps = torch.constant.float 1.0e-5
          %dim = torch.constant.int 4
          %0 = torch.aten.add.Tensor %arg0, %arg1, %alpha : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32>, !torch.float -> !torch.vtensor<[4,4],f32>
          %1:2 = torch.operator "torch.npu.npu_rms_norm"(%0, %arg2, %eps) : (!torch.vtensor<[4,4],f32>, !torch.vtensor<[4],f32>, !torch.float) -> (!torch.vtensor<[4,4],f32>, !torch.vtensor<[4,1],f32>)
          return %1#0, %1#1 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,1],f32>
        }
      }
    """)
    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.operator \"torch.npu.npu_add_rms_norm\""), checker.error
