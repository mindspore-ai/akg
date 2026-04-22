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
"""UT for squeeze and unsqueeze operators in DVM cluster."""
# pylint: disable=line-too-long

import textwrap
from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker


def test_unsqueeze_at_beginning():
    """Test that unsqueeze at beginning is correctly handled in DVMCluster."""
    # Unsqueeze at the beginning (from (4,4) to (1,4,4))
    mlir_input = textwrap.dedent(
    """
    module {
      func.func @main(%arg0: !torch.vtensor<[4, 4],f32>) -> !torch.vtensor<[1, 4, 4],f32> {
        %int0 = torch.constant.int 0
        %int2 = torch.constant.int 2
        %0 = torch.aten.unsqueeze %arg0, %int0 : !torch.vtensor<[4, 4],f32>, !torch.int -> !torch.vtensor<[1, 4, 4],f32>
        %1 = torch.aten.mul.Scalar %0, %int2 : !torch.vtensor<[1, 4, 4],f32>, !torch.int -> !torch.vtensor<[1, 4, 4],f32>
        return %1 : !torch.vtensor<[1, 4, 4],f32>
      }
    }
    """
    )
    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.reshape"), checker.error
    assert checker.check_text_contains("torch.mfusion.dvm_call"), checker.error


def test_unsqueeze_in_middle():
    """Test that unsqueeze in middle is correctly handled in DVMCluster."""
    # Unsqueeze in the middle (from (4,4) to (4,1,4))
    mlir_input = textwrap.dedent(
    """
    module {
      func.func @main(%arg0: !torch.vtensor<[4, 4],f32>) -> !torch.vtensor<[4, 1, 4],f32> {
        %int1 = torch.constant.int 1
        %int2 = torch.constant.int 2
        %0 = torch.aten.unsqueeze %arg0, %int1 : !torch.vtensor<[4, 4],f32>, !torch.int -> !torch.vtensor<[4, 1, 4],f32>
        %1 = torch.aten.mul.Scalar %0, %int2 : !torch.vtensor<[4, 1, 4],f32>, !torch.int -> !torch.vtensor<[4, 1, 4],f32>
        return %1 : !torch.vtensor<[4, 1, 4],f32>
      }
    }
    """
    )
    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.reshape"), checker.error
    assert checker.check_text_contains("torch.mfusion.dvm_call"), checker.error


def test_unsqueeze_at_end():
    """Test that unsqueeze at end is correctly handled in DVMCluster."""
    # Unsqueeze at the end (from (2,3) to (2,3,1))
    mlir_input = textwrap.dedent(
    """
    module {
      func.func @main(%arg0: !torch.vtensor<[2, 3],f32>) -> !torch.vtensor<[2, 3, 1],f32> {
        %int2 = torch.constant.int 2
        %0 = torch.aten.unsqueeze %arg0, %int2 : !torch.vtensor<[2, 3],f32>, !torch.int -> !torch.vtensor<[2, 3, 1],f32>
        %1 = torch.aten.mul.Scalar %0, %int2 : !torch.vtensor<[2, 3, 1],f32>, !torch.int -> !torch.vtensor<[2, 3, 1],f32>
        return %1 : !torch.vtensor<[2, 3, 1],f32>
      }
    }
    """
    )

    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.reshape"), checker.error
    assert checker.check_text_contains("torch.mfusion.dvm_call"), checker.error


def test_squeeze_at_beginning():
    """Test that squeeze at beginning is correctly handled in DVMCluster."""
    # Squeeze at the beginning (from (1, 4, 1, 1, 4) to (4,4))
    mlir_input = textwrap.dedent(
    """
    module {
      func.func @main(%arg0: !torch.vtensor<[1, 4, 1, 1, 4],f32>) -> !torch.vtensor<[4, 4],f32> {
        %0 = torch.aten.squeeze %arg0 : !torch.vtensor<[1, 4, 1, 1, 4],f32> -> !torch.vtensor<[4, 4],f32>
        %int2 = torch.constant.int 2
        %1 = torch.aten.mul.Scalar %0, %int2 : !torch.vtensor<[4, 4],f32>, !torch.int -> !torch.vtensor<[4, 4],f32>
        return %1 : !torch.vtensor<[4, 4],f32>
      }
    }
    """
    )
    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.reshape"), checker.error
    assert checker.check_text_contains("torch.mfusion.dvm_call"), checker.error


def test_matmul_followed_by_unsqueeze():
    """Test that matmul followed by unsqueeze is correctly clustered."""
    # Unsqueeze after matmul (matmul followed by unsqueeze)
    mlir_input = textwrap.dedent(
    """
    module {
      func.func @main(%arg0: !torch.vtensor<[2, 3],f32>, %arg1: !torch.vtensor<[3, 4],f32>) -> !torch.vtensor<[2, 4, 1],f32> {
        %int2 = torch.constant.int 2
        %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2, 3],f32>, !torch.vtensor<[3, 4],f32> -> !torch.vtensor<[2, 4],f32>
        %1 = torch.aten.unsqueeze %0, %int2 : !torch.vtensor<[2, 4],f32>, !torch.int -> !torch.vtensor<[2, 4, 1],f32>
        %2 = torch.aten.mul.Scalar %1, %int2 : !torch.vtensor<[2, 4, 1],f32>, !torch.int -> !torch.vtensor<[2, 4, 1],f32>
        return %2 : !torch.vtensor<[2, 4, 1],f32>
      }
    }
    """
    )

    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.mm"), checker.error
    assert checker.check_has_op("torch.aten.reshape"), checker.error
    assert checker.check_text_contains("torch.mfusion.dvm_call"), checker.error


def test_squeeze_followed_by_matmul():
    """Test that squeeze followed by matmul is correctly clustered."""
    # Squeeze before matmul (squeeze followed by matmul)
    mlir_input = textwrap.dedent(
    """
    module {
      func.func @main(%arg0: !torch.vtensor<[1, 2, 3],f32>, %arg1: !torch.vtensor<[3, 4],f32>) -> !torch.vtensor<[2, 4],f32> {
        %int2 = torch.constant.int 2
        %0 = torch.aten.mul.Scalar %arg0, %int2 : !torch.vtensor<[1, 2, 3],f32>, !torch.int -> !torch.vtensor<[1, 2, 3],f32>
        %1 = torch.aten.squeeze %0 : !torch.vtensor<[1, 2, 3],f32> -> !torch.vtensor<[2, 3],f32>
        %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[2, 3],f32>, !torch.vtensor<[3, 4],f32> -> !torch.vtensor<[2, 4],f32>
        return %2 : !torch.vtensor<[2, 4],f32>
      }
    }
    """
    )
    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.reshape"), checker.error
    assert checker.check_has_op("torch.aten.mm"), checker.error
    assert checker.check_text_contains("torch.mfusion.dvm_call"), checker.error


def test_squeeze_dim_followed_by_matmul():
    """Test that squeeze followed by matmul is correctly clustered."""
    # Squeeze before matmul (squeeze followed by matmul)
    mlir_input = textwrap.dedent(
    """
    module {
      func.func @main(%arg0: !torch.vtensor<[1, 2, 3],f32>, %arg1: !torch.vtensor<[3, 4],f32>) -> !torch.vtensor<[2, 4],f32> {
        %int2 = torch.constant.int 2
        %int0 = torch.constant.int 0
        %0 = torch.aten.mul.Scalar %arg0, %int2 : !torch.vtensor<[1, 2, 3],f32>, !torch.int -> !torch.vtensor<[1, 2, 3],f32>
        %1 = torch.aten.squeeze.dim %0, %int0 : !torch.vtensor<[1, 2, 3],f32>, !torch.int -> !torch.vtensor<[2, 3],f32>
        %2 = torch.aten.mm %1, %arg1 : !torch.vtensor<[2, 3],f32>, !torch.vtensor<[3, 4],f32> -> !torch.vtensor<[2, 4],f32>
        return %2 : !torch.vtensor<[2, 4],f32>
      }
    }
    """
    )
    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.reshape"), checker.error
    assert checker.check_has_op("torch.aten.mm"), checker.error
    assert checker.check_text_contains("torch.mfusion.dvm_call"), checker.error
