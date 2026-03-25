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

"""Test binary ops pipeline with different operations."""

import textwrap
from ut_utils.mlir_checker import MlirChecker
from mfusion.torch.inductor import fuse_and_optimize


def test_binary_ops_pipeline_with_add_tensor():
    """Test binary ops pipeline with add.Tensor operation."""
    # Torch dialect MLIR string with add.Tensor operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_add_tensor(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
            %alpha = torch.constant.float 1.0
            %0 = torch.aten.add.Tensor %arg0, %arg1, %alpha : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32>, !torch.float -> !torch.vtensor<[4,4],f32>
            return %0 : !torch.vtensor<[4,4],f32>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.add.Tensor"), checker.error


def test_binary_ops_pipeline_with_add_scalar():
    """Test binary ops pipeline with add.Scalar operation."""
    # Torch dialect MLIR string with add.Scalar operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_add_scalar(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.float) -> !torch.vtensor<[4,4],f32> {
            %alpha = torch.constant.float 1.0
            %0 = torch.aten.add.Scalar %arg0, %arg1, %alpha : !torch.vtensor<[4,4],f32>, !torch.float, !torch.float -> !torch.vtensor<[4,4],f32>
            return %0 : !torch.vtensor<[4,4],f32>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.add.Scalar"), checker.error


def test_binary_ops_pipeline_with_sub_tensor():
    """Test binary ops pipeline with sub.Tensor operation."""
    # Torch dialect MLIR string with sub.Tensor operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_sub_tensor(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
            %alpha = torch.constant.float 1.0
            %0 = torch.aten.sub.Tensor %arg0, %arg1, %alpha : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32>, !torch.float -> !torch.vtensor<[4,4],f32>
            return %0 : !torch.vtensor<[4,4],f32>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.sub.Tensor"), checker.error


def test_binary_ops_pipeline_with_sub_scalar():
    """Test binary ops pipeline with sub.Scalar operation."""
    # Torch dialect MLIR string with sub.Scalar operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_sub_scalar(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.float) -> !torch.vtensor<[4,4],f32> {
            %alpha = torch.constant.float 1.0
            %0 = torch.aten.sub.Scalar %arg0, %arg1, %alpha : !torch.vtensor<[4,4],f32>, !torch.float, !torch.float -> !torch.vtensor<[4,4],f32>
            return %0 : !torch.vtensor<[4,4],f32>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.sub.Scalar"), checker.error


def test_binary_ops_pipeline_with_mul_tensor():
    """Test binary ops pipeline with mul.Tensor operation."""
    # Torch dialect MLIR string with mul.Tensor operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_mul_tensor(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
            %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
            return %0 : !torch.vtensor<[4,4],f32>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.mul.Tensor"), checker.error


def test_binary_ops_pipeline_with_mul_scalar():
    """Test binary ops pipeline with mul.Scalar operation."""
    # Torch dialect MLIR string with mul.Scalar operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_mul_scalar(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.float) -> !torch.vtensor<[4,4],f32> {
            %0 = torch.aten.mul.Scalar %arg0, %arg1 : !torch.vtensor<[4,4],f32>, !torch.float -> !torch.vtensor<[4,4],f32>
            return %0 : !torch.vtensor<[4,4],f32>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.mul.Scalar"), checker.error


def test_decompose_pipeline_with_mul_scalar2():
    """Test decompose pipeline with mul"""
    # Torch dialect MLIR string with mul
    torch_mlir = textwrap.dedent("""
      module {
        func.func @test_add(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
          %arg1 = torch.constant.int 2
          %0 = torch.aten.mul.Scalar %arg0, %arg1 : !torch.vtensor<[4,4],f32>, !torch.int -> !torch.vtensor<[4,4],f32>
          return %0 : !torch.vtensor<[4,4],f32>
        }
      }
    """)
    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_no_op("torch.operator"), checker.error
    assert checker.check_has_op("torch.aten.mul.Scalar"), checker.error
    assert checker.check_text_contains("torch.constant.float 2.0"), checker.error


def test_binary_ops_pipeline_with_div_tensor():
    """Test binary ops pipeline with div.Tensor operation."""
    # Torch dialect MLIR string with div.Tensor operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_div_tensor(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
            %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
            return %0 : !torch.vtensor<[4,4],f32>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.div.Tensor"), checker.error


def test_binary_ops_pipeline_with_div_scalar():
    """Test binary ops pipeline with div.Scalar operation."""
    # Torch dialect MLIR string with div.Scalar operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_div_scalar(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.float) -> !torch.vtensor<[4,4],f32> {
            %0 = torch.aten.div.Scalar %arg0, %arg1 : !torch.vtensor<[4,4],f32>, !torch.float -> !torch.vtensor<[4,4],f32>
            return %0 : !torch.vtensor<[4,4],f32>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.div.Scalar"), checker.error


def test_binary_ops_pipeline_with_div_scalar_int():
    """Test binary ops pipeline with div.Scalar operation."""
    # Torch dialect MLIR string with div.Scalar operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_div_scalar_int32(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.int) -> !torch.vtensor<[4,4],f32> {
            %0 = torch.aten.div.Scalar %arg0, %arg1 : !torch.vtensor<[4,4],f32>, !torch.int -> !torch.vtensor<[4,4],f32>
            return %0 : !torch.vtensor<[4,4],f32>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.div.Scalar"), checker.error


def test_binary_ops_pipeline_with_div_scalar_f16():
    """Test binary ops pipeline with div.Scalar operation."""
    # Torch dialect MLIR string with div.Scalar operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_div_scalar_f16(%arg0: !torch.vtensor<[4,4],f16>, %arg1: !torch.float) -> !torch.vtensor<[4,4],f16> {
            %0 = torch.aten.div.Scalar %arg0, %arg1 : !torch.vtensor<[4,4],f16>, !torch.float -> !torch.vtensor<[4,4],f16>
            return %0 : !torch.vtensor<[4,4],f16>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.div.Scalar"), checker.error


def test_binary_ops_pipeline_with_le_scalar():
    """Test binary ops pipeline with le.Scalar operation."""
    # Torch dialect MLIR string with le.Scalar operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_le_scalar(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.float) -> !torch.vtensor<[4,4],i1> {
            %0 = torch.aten.le.Scalar %arg0, %arg1 : !torch.vtensor<[4,4],f32>, !torch.float -> !torch.vtensor<[4,4],i1>
            return %0 : !torch.vtensor<[4,4],i1>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.le.Scalar"), checker.error


def test_binary_ops_pipeline_with_le_scalar_with_bool():
    """Test binary ops pipeline with le.Scalar operation."""
    # Torch dialect MLIR string with le.Scalar operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_le_scalar(%arg0: !torch.vtensor<[4,4],i1>) -> !torch.vtensor<[4,4],i1> {
            %cst = torch.constant.int 1
            %0 = torch.aten.le.Scalar %arg0, %cst : !torch.vtensor<[4,4],i1>, !torch.int -> !torch.vtensor<[4,4],i1>
            return %0 : !torch.vtensor<[4,4],i1>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.le.Scalar"), checker.error


def test_binary_ops_pipeline_with_le_scalar_with_bool2():
    """Test binary ops pipeline with le.Scalar operation."""
    # Torch dialect MLIR string with le.Scalar operation
    torch_mlir = textwrap.dedent("""
        module {
          func.func @test_le_scalar(%arg0: !torch.vtensor<[4,4],i1>, %arg1: !torch.int) -> !torch.vtensor<[4,4],i1> {
            %0 = torch.aten.le.Scalar %arg0, %arg1 : !torch.vtensor<[4,4],i1>, !torch.int -> !torch.vtensor<[4,4],i1>
            return %0 : !torch.vtensor<[4,4],i1>
          }
        }
    """)

    # Run fuse and optimize
    result = fuse_and_optimize(torch_mlir)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains("torch.aten.le.Scalar"), checker.error
