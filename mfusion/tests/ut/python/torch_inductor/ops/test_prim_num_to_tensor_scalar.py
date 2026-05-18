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

"""UT for torch.prim.NumToTensor.Scalar operator."""

import textwrap
from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker


def test_convert_prim_num_to_tensor_scalar_cast_fold():
    """Verify that prim.NumToTensor.Scalar + cast is converted to torch.aten.full after optimization."""
    MLIR_PRIM_NUM_TO_TENSOR_SCALAR = textwrap.dedent("""
      module {
        func.func @convert_prim_num_to_tensor_scalar_no_const_fold(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
          %c2 = torch.constant.int 2
          %0 = torch.prim.NumToTensor.Scalar %c2 : !torch.int -> !torch.vtensor<[],si64>
          %1 = torch.aten.sub.Tensor %arg0, %0, %c2 : !torch.vtensor<[],f32>, !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[],f32>
          return %1 : !torch.vtensor<[],f32>
        }
      }
    """)
    result = fuse_and_optimize(MLIR_PRIM_NUM_TO_TENSOR_SCALAR)
    checker = MlirChecker.parse_torch_module(result)
    # Based on the original MLIR test CHECK comments:
    # After conversion, we expect torch.prim.NumToTensor.Scalar to be preserved
    # and torch.vtensor.literal not to be generated
    assert checker.check_no_op("torch.vtensor.literal"), checker.error
    # The full operation should be present
    assert checker.check_text_not_contains("torch.npu._npu_dtype_cast"), checker.error
    assert checker.check_has_op("torch.aten.full"), checker.error


def test_convert_prim_num_to_tensor_scalar_fold():
    """Verify that prim.NumToTensor.Scalar is converted to torch.aten.full after optimization."""
    MLIR_PRIM_NUM_TO_TENSOR_SCALAR = textwrap.dedent("""
      module {
        func.func @convert_prim_num_to_tensor_scalar_no_const_fold(%arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
          %c2 = torch.constant.float 2.0
          %0 = torch.prim.NumToTensor.Scalar %c2 : !torch.float -> !torch.vtensor<[],f32>
          %1 = torch.aten.sub.Tensor %arg0, %0, %c2 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
          return %1 : !torch.vtensor<[],f32>
        }
      }
    """)
    result = fuse_and_optimize(MLIR_PRIM_NUM_TO_TENSOR_SCALAR)
    checker = MlirChecker.parse_torch_module(result)
    # Based on the original MLIR test CHECK comments:
    # After conversion, we expect torch.prim.NumToTensor.Scalar to be preserved
    # and torch.vtensor.literal not to be generated
    assert checker.check_no_op("torch.vtensor.literal"), checker.error
    # The full operation should be present
    assert checker.check_has_op("torch.aten.full"), checker.error


def test_convert_prim_num_to_tensor_variable_scalar_fold():
    """Verify that prim.NumToTensor.Scalar is converted to torch.aten.full after optimization."""
    MLIR_PRIM_NUM_TO_TENSOR_SCALAR = textwrap.dedent("""
      module {
        func.func @convert_prim_num_to_tensor_scalar_no_const_fold(%cst2: !torch.float, %arg0: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
          %0 = torch.prim.NumToTensor.Scalar %cst2 : !torch.float -> !torch.vtensor<[],f32>
          %1 = torch.aten.sub.Tensor %arg0, %0, %cst2 : !torch.vtensor<[],f32>, !torch.vtensor<[],f32>, !torch.float -> !torch.vtensor<[],f32>
          return %1 : !torch.vtensor<[],f32>
        }
      }
    """)
    result = fuse_and_optimize(MLIR_PRIM_NUM_TO_TENSOR_SCALAR)
    checker = MlirChecker.parse_torch_module(result)
    # Based on the original MLIR test CHECK comments:
    # After conversion, we expect torch.prim.NumToTensor.Scalar to be preserved
    # and torch.vtensor.literal not to be generated
    assert checker.check_no_op("torch.vtensor.literal"), checker.error
    # The full operation should be present
    assert checker.check_has_op("torch.aten.full"), checker.error
