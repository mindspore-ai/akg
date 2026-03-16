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

"""UT for FuseRmsNorm: fuse RmsNorm patterns to aclnn.rms_norm."""
# pylint: disable=line-too-long

import textwrap

from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker

# RmsNorm in Torch: add(mean, eps) -> rsqrt -> mul(x, rsqrt) -> mul(gamma).
# After convert-torch-to-mfuse and mfuse-fusion, should become aclnn.rms_norm and then
# convert-mfuse-to-torch yields torch.npu.npu_rms_norm.
MLIR_RMS_NORM = textwrap.dedent(
    """
module {
  func.func @main(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>, %mean: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4],f32>
    %r = torch.aten.rsqrt %add : !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    %norm = torch.aten.mul.Tensor %x, %r : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    %out = torch.aten.mul.Tensor %gamma, %norm : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    return %out : !torch.vtensor<[2,4],f32>
  }
}
"""
)


def test_fuse_rms_norm_fusion():
    """RmsNorm chain (add+rsqrt+mul+mul) should fuse to aclnn.rms_norm and lower to npu_rms_norm."""
    result = fuse_and_optimize(MLIR_RMS_NORM)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains('torch.operator "torch.npu.npu_rms_norm"'), (
        checker.error or "Expected torch.npu.npu_rms_norm after FuseRmsNorm + convert-mfuse-to-torch"
    )


# Mixed-precision RmsNorm matching the PM02 pattern: bf16 input, f32 computation, bf16 output.
# npu_dtype_cast(bf16→f32) → pow → mean → add(eps) → rsqrt → mul(x,rsqrt) → npu_dtype_cast(f32→bf16) → mul(gamma)
MLIR_RMS_NORM_MIXED_PRECISION = textwrap.dedent(
    """
module {
  func.func @main(%x_bf16: !torch.vtensor<[1,4,8],bf16>, %gamma: !torch.vtensor<[8],bf16>) -> !torch.vtensor<[1,4,8],bf16> attributes {torch.assume_strict_symbolic_shapes} {
    %int6 = torch.constant.int 6
    %int15 = torch.constant.int 15
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 9.999990e-07
    %x_f32 = torch.operator "torch.npu._npu_dtype_cast"(%x_bf16, %int6) : (!torch.vtensor<[1,4,8],bf16>, !torch.int) -> !torch.vtensor<[1,4,8],f32>
    %pow = torch.aten.pow.Tensor_Scalar %x_f32, %int2 : !torch.vtensor<[1,4,8],f32>, !torch.int -> !torch.vtensor<[1,4,8],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[1,4,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,4,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[1,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,4,1],f32>
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[1,4,1],f32> -> !torch.vtensor<[1,4,1],f32>
    %norm = torch.aten.mul.Tensor %x_f32, %rsqrt : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,1],f32> -> !torch.vtensor<[1,4,8],f32>
    %norm_bf16 = torch.operator "torch.npu._npu_dtype_cast"(%norm, %int15) : (!torch.vtensor<[1,4,8],f32>, !torch.int) -> !torch.vtensor<[1,4,8],bf16>
    %out = torch.aten.mul.Tensor %gamma, %norm_bf16 : !torch.vtensor<[8],bf16>, !torch.vtensor<[1,4,8],bf16> -> !torch.vtensor<[1,4,8],bf16>
    return %out : !torch.vtensor<[1,4,8],bf16>
  }
}
"""
)


def test_fuse_rms_norm_mixed_precision():
    """Mixed-precision RmsNorm (bf16 input, f32 compute, bf16 output) should fuse across npu_dtype_cast."""
    result = fuse_and_optimize(MLIR_RMS_NORM_MIXED_PRECISION)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains('torch.operator "torch.npu.npu_rms_norm"'), (
        checker.error or "Expected torch.npu.npu_rms_norm after FuseRmsNorm with mixed precision"
    )


# Training graph: rsqrt result returned for backward pass (rsqrt has 2 uses after convert-torch-to-mfuse).
# Simplified from PM02 QKV RmsNorm pattern: bf16 input, f32 compute, bf16 output, rsqrt returned.
MLIR_RMS_NORM_TRAINING_RSQRT_RETURNED = textwrap.dedent(
    """
module {
  func.func @main(%x_bf16: !torch.vtensor<[1,4,8],bf16>, %gamma: !torch.vtensor<[8],bf16>) -> (!torch.vtensor<[1,4,8],bf16>, !torch.vtensor<[1,4,1],f32>) attributes {torch.assume_strict_symbolic_shapes} {
    %int6 = torch.constant.int 6
    %int15 = torch.constant.int 15
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 9.999990e-07
    %x_f32 = torch.operator "torch.npu._npu_dtype_cast"(%x_bf16, %int6) : (!torch.vtensor<[1,4,8],bf16>, !torch.int) -> !torch.vtensor<[1,4,8],f32>
    %pow = torch.aten.pow.Tensor_Scalar %x_f32, %int2 : !torch.vtensor<[1,4,8],f32>, !torch.int -> !torch.vtensor<[1,4,8],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[1,4,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,4,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[1,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,4,1],f32>
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[1,4,1],f32> -> !torch.vtensor<[1,4,1],f32>
    %norm = torch.aten.mul.Tensor %x_f32, %rsqrt : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,1],f32> -> !torch.vtensor<[1,4,8],f32>
    %norm_bf16 = torch.operator "torch.npu._npu_dtype_cast"(%norm, %int15) : (!torch.vtensor<[1,4,8],f32>, !torch.int) -> !torch.vtensor<[1,4,8],bf16>
    %out = torch.aten.mul.Tensor %gamma, %norm_bf16 : !torch.vtensor<[8],bf16>, !torch.vtensor<[1,4,8],bf16> -> !torch.vtensor<[1,4,8],bf16>
    return %out, %rsqrt : !torch.vtensor<[1,4,8],bf16>, !torch.vtensor<[1,4,1],f32>
  }
}
"""
)


def test_fuse_rms_norm_training_rsqrt_returned():
    """Training graph: rsqrt returned for backward pass (multi-use rsqrt) should still fuse."""
    result = fuse_and_optimize(MLIR_RMS_NORM_TRAINING_RSQRT_RETURNED)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains('torch.operator "torch.npu.npu_rms_norm"'), (
        checker.error or "Expected torch.npu.npu_rms_norm in training graph with rsqrt returned for backward"
    )


def test_fuse_rms_norm_no_fusion_multiple_uses():
    """When normalized value has multiple uses, fusion should not apply; mul/rsqrt remain."""
    mlir_multi_use = textwrap.dedent(
        """
module {
  func.func @main(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>, %mean: !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>) attributes {torch.assume_strict_symbolic_shapes} {
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.0e-05
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4],f32>
    %r = torch.aten.rsqrt %add : !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    %norm = torch.aten.mul.Tensor %x, %r : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    %out = torch.aten.mul.Tensor %gamma, %norm : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    return %out, %norm : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>
  }
}
"""
    )
    result = fuse_and_optimize(mlir_multi_use)
    checker = MlirChecker.parse_torch_module(result)
    assert not checker.check_text_contains('torch.operator "torch.npu.npu_rms_norm"'), (
        "Normalized value has multiple uses; FuseRmsNorm should NOT fuse this pattern"
    )

