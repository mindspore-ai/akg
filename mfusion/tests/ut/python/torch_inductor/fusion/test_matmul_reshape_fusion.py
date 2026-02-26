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

"""UT for MatMul Reshape fusion in fuse_and_optimize pipeline."""

import os
import pytest

os.environ['LLVM_DEBUG'] = 'fuse-matmul-reshape'

# pylint: disable=wrong-import-position
from mfusion.torch.inductor import fuse_and_optimize

# pylint: disable=wrong-import-position
from ut_utils.mlir_checker import MlirChecker


# K size threshold for FP16/BF16: fusion only when K < 27392 (must match C++ kMaxKForFp16 + 1).
K_MAX_FP16_RESHAPE_FUSION = 27392

# Test case 1: MatMul with N=1, F32, should add reshape to second input
MLIR_MATMUL_RESHAPE_F32 = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[16, 1024],f32>, %arg1: !torch.vtensor<[1024, 1],f32>) -> !torch.vtensor<[16, 1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[16, 1024],f32>, !torch.vtensor<[1024, 1],f32> -> !torch.vtensor<[16, 1],f32>
    return %0 : !torch.vtensor<[16, 1],f32>
  }
}
"""

# Test case 2: MatMul with N=1, F16, K < K_MAX_FP16_RESHAPE_FUSION, should add reshape
MLIR_MATMUL_RESHAPE_F16 = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[16, 20000],f16>, %arg1: !torch.vtensor<[20000, 1],f16>) -> !torch.vtensor<[16, 1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[16, 20000],f16>, !torch.vtensor<[20000, 1],f16> -> !torch.vtensor<[16, 1],f16>
    return %0 : !torch.vtensor<[16, 1],f16>
  }
}
"""

# Test case 3: MatMul with N=1, F16, K >= K_MAX_FP16_RESHAPE_FUSION, should NOT add reshape
MLIR_MATMUL_RESHAPE_F16_LARGE_K = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[16, 30000],f16>, %arg1: !torch.vtensor<[30000, 1],f16>) -> !torch.vtensor<[16, 1],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[16, 30000],f16>, !torch.vtensor<[30000, 1],f16> -> !torch.vtensor<[16, 1],f16>
    return %0 : !torch.vtensor<[16, 1],f16>
  }
}
"""

# Test case 4: MatMul with N != 1, should NOT add reshape to second input
MLIR_MATMUL_NO_RESHAPE_N_NOT_1 = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[16, 1024],f32>, %arg1: !torch.vtensor<[1024, 8],f32>) -> !torch.vtensor<[16, 8],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[16, 1024],f32>, !torch.vtensor<[1024, 8],f32> -> !torch.vtensor<[16, 8],f32>
    return %0 : !torch.vtensor<[16, 8],f32>
  }
}
"""

# Test case 5: MatMul with bias and N=1, F32, should add reshape to second input
MLIR_MATMUL_WITH_BIAS_RESHAPE_F32 = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[16, 1024],f32>, %arg1: !torch.vtensor<[1024, 1],f32>, %bias: !torch.vtensor<[1],f32>) -> !torch.vtensor<[16, 1],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[16, 1024],f32>, !torch.vtensor<[1024, 1],f32> -> !torch.vtensor<[16, 1],f32>
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %bias, %int1 : !torch.vtensor<[16, 1],f32>, !torch.vtensor<[1],f32>, !torch.int -> !torch.vtensor<[16, 1],f32>
    return %1 : !torch.vtensor<[16, 1],f32>
  }
}
"""


# All the test cases are skipped because the fuse-matmul-reshape pass is disabled.
# Refer to FuseMatmulReshape.cc for more details.
@pytest.mark.skip(reason="fuse-matmul-reshape pass is disabled")
def test_matmul_reshape_fusion_f32():
    """Test MatMul Reshape fusion with F32 data type."""
    result = fuse_and_optimize(MLIR_MATMUL_RESHAPE_F32)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, the second input should have a reshape operation
    # Check that torch.aten.reshape exists (mfuse.reshape is converted to torch.aten.reshape)
    assert checker.check_has_op("torch.aten.reshape"), checker.error or "torch.aten.reshape should exist after fusion"


@pytest.mark.skip(reason="fuse-matmul-reshape pass is disabled")
def test_matmul_reshape_fusion_f16():
    """Test MatMul Reshape fusion with F16 data type and K < 27392."""
    result = fuse_and_optimize(MLIR_MATMUL_RESHAPE_F16)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, the second input should have a reshape operation
    # Check that torch.aten.reshape exists (mfuse.reshape is converted to torch.aten.reshape)
    assert checker.check_has_op("torch.aten.reshape"), checker.error or "torch.aten.reshape should exist after fusion"


@pytest.mark.skip(reason="fuse-matmul-reshape pass is disabled")
def test_matmul_reshape_fusion_f16_large_k():
    """Test MatMul Reshape fusion with F16 data type and K >= 27392."""
    result = fuse_and_optimize(MLIR_MATMUL_RESHAPE_F16_LARGE_K)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, the second input should NOT have a reshape operation
    # Check that torch.aten.reshape does not exist (mfuse.reshape is converted to torch.aten.reshape)
    assert checker.check_no_op("torch.aten.reshape"), (
        checker.error or "torch.aten.reshape should not exist after fusion for large K"
    )
    assert checker.check_no_op("torch.aten.view"), checker.error or "torch.aten.view should not exist after fusion"


@pytest.mark.skip(reason="fuse-matmul-reshape pass is disabled")
def test_matmul_no_reshape_n_not_1():
    """Test MatMul Reshape fusion with N != 1."""
    result = fuse_and_optimize(MLIR_MATMUL_NO_RESHAPE_N_NOT_1)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, the second input should NOT have a reshape operation
    # Check that torch.aten.reshape does not exist (mfuse.reshape is converted to torch.aten.reshape)
    assert checker.check_no_op("torch.aten.reshape"), (
        checker.error or "torch.aten.reshape should not exist after fusion for N != 1"
    )
    assert checker.check_no_op("torch.aten.view"), checker.error or "torch.aten.view should not exist after fusion"


@pytest.mark.skip(reason="fuse-matmul-reshape pass is disabled")
def test_matmul_with_bias_reshape_fusion_f32():
    """Test MatMulWithBias Reshape fusion with F32 data type."""
    result = fuse_and_optimize(MLIR_MATMUL_WITH_BIAS_RESHAPE_F32)
    checker = MlirChecker.parse_torch_module(result)
    # After fusion, the second input should have a reshape operation
    # Check that torch.aten.reshape exists (mfuse.reshape is converted to torch.aten.reshape)
    assert checker.check_has_op("torch.aten.reshape"), checker.error or "torch.aten.reshape should exist after fusion"
