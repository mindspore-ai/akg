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

"""UT for TorchFuseRmsNorm: fuse RmsNorm patterns on Torch dialect."""
# pylint: disable=line-too-long

import re
import textwrap

from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker

# Complete RMSNorm formula: pow(x, 2) -> mean -> add(eps) -> rsqrt -> mul(x, rsqrt) -> mul(gamma)
MLIR_RMS_NORM_COMPLETE = textwrap.dedent(
    """
module {
  func.func @main(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05

    // Step 1: pow(x, 2)
    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>

    // Step 2: mean(pow, dim=-1, keepdim=True)
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>

    // Step 3: add(mean, eps)
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>

    // Step 4: rsqrt(add)
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>

    // Step 5: mul(x, rsqrt) - normalization
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>

    // Step 6: mul(norm, gamma) - scaling
    %out = torch.aten.mul.Tensor %gamma, %norm : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    return %out : !torch.vtensor<[2,4],f32>
  }
}
"""
)


def test_torch_fuse_rms_norm_complete():
    """Complete RMSNorm formula (pow -> mean -> add -> rsqrt -> mul -> mul) should fuse."""
    result = fuse_and_optimize(MLIR_RMS_NORM_COMPLETE)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains('torch.operator "torch.npu.npu_rms_norm"'), (
        checker.error
        or "Expected torch.npu.npu_rms_norm after torch-fuse-rms-norm (complete formula)"
    )
    # Verify the original operations are fused away
    assert not checker.check_text_contains(
        "torch.aten.pow.Tensor_Scalar"
    ), "pow should be fused"
    assert not checker.check_text_contains(
        "torch.aten.mean.dim"
    ), "mean should be fused"
    assert not checker.check_text_contains(
        "torch.aten.add.Scalar"
    ), "add should be fused"
    assert not checker.check_text_contains("torch.aten.rsqrt"), "rsqrt should be fused"
    assert not checker.check_text_contains(
        "torch.aten.mul.Tensor"
    ), "mul operations should be fused"


# Smallest inference RmsNorm: rsqrt is only used by mul(x, rsqrt); return only the final mul output.
MLIR_RMS_NORM_INFERENCE_SECOND_UNUSED = textwrap.dedent(
    """
module {
  func.func @main(%x: !torch.vtensor<[1,2],f32>, %gamma: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,2],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05
    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[1,2],f32>, !torch.int -> !torch.vtensor<[1,2],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[1,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[1,2,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,2,1],f32>
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[1,2,1],f32> -> !torch.vtensor<[1,2,1],f32>
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[1,2],f32>, !torch.vtensor<[1,2,1],f32> -> !torch.vtensor<[1,2],f32>
    %out = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[1,2],f32>, !torch.vtensor<[1,2],f32> -> !torch.vtensor<[1,2],f32>
    return %out : !torch.vtensor<[1,2],f32>
  }
}
"""
)


def test_torch_fuse_rms_norm_inference_second_result_unused():
    """Guard inference RmsNorm where npu_rms_norm has two results but only #0 is used in IR.

    aten.rsqrt has a single use (mul with x), so torch-fuse-rms-norm does not run
    replaceAllUsesWith(rsqrt, fused#1). The final mul is replaced by fused#0; after erasing
    the decomposed chain, fused#1 has no SSA users. That is expected (second result exists for
    API/training); it is not a missing replacement.
    """
    result = fuse_and_optimize(MLIR_RMS_NORM_INFERENCE_SECOND_UNUSED)
    assert (
        ':2 = torch.operator "torch.npu.npu_rms_norm"' in result
    ), "Expected 2-result fused op"
    assert re.search(
        r"return %[0-9]+#0 : !torch\.vtensor<\[1,2\],f32>", result
    ), "Expected return to use only fused result #0 (second result unused in this graph)"


# Complete RMSNorm formula with mixed precision: bf16 input, f32 compute, bf16 output
MLIR_RMS_NORM_MIXED_PRECISION_COMPLETE = textwrap.dedent(
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

    // Cast bf16 to f32 for computation
    %x_f32 = torch.operator "torch.npu._npu_dtype_cast"(%x_bf16, %int6) : (!torch.vtensor<[1,4,8],bf16>, !torch.int) -> !torch.vtensor<[1,4,8],f32>

    // Step 1: pow(x, 2)
    %pow = torch.aten.pow.Tensor_Scalar %x_f32, %int2 : !torch.vtensor<[1,4,8],f32>, !torch.int -> !torch.vtensor<[1,4,8],f32>

    // Step 2: mean(pow, dim=-1, keepdim=True)
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[1,4,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,4,1],f32>

    // Step 3: add(mean, eps)
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[1,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,4,1],f32>

    // Step 4: rsqrt(add)
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[1,4,1],f32> -> !torch.vtensor<[1,4,1],f32>

    // Step 5: mul(x, rsqrt) - normalization
    %norm = torch.aten.mul.Tensor %x_f32, %rsqrt : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,1],f32> -> !torch.vtensor<[1,4,8],f32>

    // Cast f32 back to bf16
    %norm_bf16 = torch.operator "torch.npu._npu_dtype_cast"(%norm, %int15) : (!torch.vtensor<[1,4,8],f32>, !torch.int) -> !torch.vtensor<[1,4,8],bf16>

    // Step 6: mul(norm, gamma) - scaling
    %out = torch.aten.mul.Tensor %gamma, %norm_bf16 : !torch.vtensor<[8],bf16>, !torch.vtensor<[1,4,8],bf16> -> !torch.vtensor<[1,4,8],bf16>

    return %out : !torch.vtensor<[1,4,8],bf16>
  }
}
"""
)


def test_torch_fuse_rms_norm_mixed_precision_complete():
    """Complete RMSNorm formula with mixed precision should fuse across dtype casts."""
    result = fuse_and_optimize(MLIR_RMS_NORM_MIXED_PRECISION_COMPLETE)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains('torch.operator "torch.npu.npu_rms_norm"'), (
        checker.error
        or "Expected torch.npu.npu_rms_norm after torch-fuse-rms-norm (mixed precision, complete)"
    )


# RMSNorm should not fuse when power is not 2
MLIR_RMS_NORM_NO_FUSION_WRONG_POWER = textwrap.dedent(
    """
module {
  func.func @main(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int3 = torch.constant.int 3  // Wrong power
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05

    // Step 1: pow(x, 3) - wrong power
    %pow = torch.aten.pow.Tensor_Scalar %x, %int3 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>

    // Step 2: mean(pow, dim=-1, keepdim=True)
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>

    // Step 3: add(mean, eps)
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>

    // Step 4: rsqrt(add)
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>

    // Step 5: mul(x, rsqrt) - normalization
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>

    // Step 6: mul(norm, gamma) - scaling
    %out = torch.aten.mul.Tensor %gamma, %norm : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    return %out : !torch.vtensor<[2,4],f32>
  }
}
"""
)


def test_torch_fuse_rms_norm_no_fusion_wrong_power():
    """RMSNorm should not fuse when power is not 2."""
    result = fuse_and_optimize(MLIR_RMS_NORM_NO_FUSION_WRONG_POWER)
    checker = MlirChecker.parse_torch_module(result)
    assert not checker.check_text_contains(
        'torch.operator "torch.npu.npu_rms_norm"'
    ), "RMSNorm with wrong power should not be fused"
    # Verify the original operations are preserved
    assert checker.check_text_contains(
        "torch.aten.pow.Tensor_Scalar"
    ), "pow should be preserved"
    assert checker.check_text_contains(
        "torch.aten.mean.dim"
    ), "mean should be preserved"
    assert checker.check_text_contains(
        "torch.aten.add.Scalar"
    ), "add should be preserved"
    assert checker.check_text_contains("torch.aten.rsqrt"), "rsqrt should be preserved"
    assert checker.check_text_contains(
        "torch.aten.mul.Tensor"
    ), "mul operations should be preserved"


# RMSNorm should not fuse when add alpha is not 1
MLIR_RMS_NORM_NO_FUSION_WRONG_ADD_ALPHA = textwrap.dedent(
    """
module {
  func.func @main(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %two = torch.constant.int 2  // Wrong alpha
    %eps = torch.constant.float 1.000000e-05

    // Step 1: pow(x, 2)
    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>

    // Step 2: mean(pow, dim=-1, keepdim=True)
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>

    // Step 3: add(mean, eps) with wrong alpha
    %add = torch.aten.add.Scalar %mean, %eps, %two : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>

    // Step 4: rsqrt(add)
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>

    // Step 5: mul(x, rsqrt) - normalization
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>

    // Step 6: mul(norm, gamma) - scaling
    %out = torch.aten.mul.Tensor %gamma, %norm : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    return %out : !torch.vtensor<[2,4],f32>
  }
}
"""
)


def test_torch_fuse_rms_norm_no_fusion_wrong_add_alpha():
    """RMSNorm should not fuse when add alpha is not 1."""
    result = fuse_and_optimize(MLIR_RMS_NORM_NO_FUSION_WRONG_ADD_ALPHA)
    checker = MlirChecker.parse_torch_module(result)
    assert not checker.check_text_contains(
        'torch.operator "torch.npu.npu_rms_norm"'
    ), "RMSNorm with wrong add alpha should not be fused"


# RMSNorm should not fuse when norm has multiple uses
MLIR_RMS_NORM_NO_FUSION_MULTIPLE_NORM_USES = textwrap.dedent(
    """
module {
  func.func @main(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>) attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05

    // Step 1: pow(x, 2)
    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>

    // Step 2: mean(pow, dim=-1, keepdim=True)
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>

    // Step 3: add(mean, eps)
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>

    // Step 4: rsqrt(add)
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>

    // Step 5: mul(x, rsqrt) - normalization
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>
    
    // Multiple uses of norm - should prevent fusion
    %out1 = torch.aten.mul.Tensor %gamma, %norm : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    %out2 = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    return %out1, %out2 : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>
  }
}
"""
)


def test_torch_fuse_rms_norm_no_fusion_multiple_norm_uses():
    """RMSNorm should not fuse when norm has multiple uses."""
    result = fuse_and_optimize(MLIR_RMS_NORM_NO_FUSION_MULTIPLE_NORM_USES)
    checker = MlirChecker.parse_torch_module(result)
    assert not checker.check_text_contains(
        'torch.operator "torch.npu.npu_rms_norm"'
    ), "RMSNorm with multiple norm uses should not be fused"


# Training graph: rsqrt result returned for backward pass (multi-use rsqrt)
MLIR_RMS_NORM_TRAINING_GRAPH = textwrap.dedent(
    """
module {
  func.func @main(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32>) attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05

    // Step 1: pow(x, 2)
    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>

    // Step 2: mean(pow, dim=-1, keepdim=True)
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>

    // Step 3: add(mean, eps)
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>

    // Step 4: rsqrt(add)
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>

    // Step 5: mul(x, rsqrt) - normalization
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>

    // Step 6: mul(norm, gamma) - scaling
    %out = torch.aten.mul.Tensor %gamma, %norm : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    // Training graph: return both output and rsqrt for backward pass
    return %out, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32>
  }
}
"""
)


def test_torch_fuse_rms_norm_training_graph():
    """Training graph with rsqrt returned for backward pass should still fuse."""
    result = fuse_and_optimize(MLIR_RMS_NORM_TRAINING_GRAPH)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains('torch.operator "torch.npu.npu_rms_norm"'), (
        checker.error
        or "Expected torch.npu.npu_rms_norm in training graph with rsqrt returned for backward"
    )


MLIR_RMS_NORM_ORIGINAL = textwrap.dedent(
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


def test_torch_not_fuse_rms_norm_original():
    """Original (add+rsqrt+mul+mul) should not fuse"""
    result = fuse_and_optimize(MLIR_RMS_NORM_ORIGINAL)
    checker = MlirChecker.parse_torch_module(result)
    assert not checker.check_text_contains('torch.operator "torch.npu.npu_rms_norm"'), (
        checker.error or "Not fuse torch.npu.npu_rms_norm after torch-fuse-rms-norm"
    )


# Mixed-precision RmsNorm matching the PM02 pattern: bf16 input, f32 computation, bf16 output.
# npu_dtype_cast(bf16→f32) → pow → mean → add(eps) → rsqrt → mul(x,rsqrt) → npu_dtype_cast(f32→bf16) → mul(gamma)
MLIR_RMS_NORM_MIXED_PRECISION_ORIGINAL = textwrap.dedent(
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


def test_torch_fuse_rms_norm_mixed_precision_original():
    """Mixed-precision RMSNorm (original simplified) should still fuse."""
    result = fuse_and_optimize(MLIR_RMS_NORM_MIXED_PRECISION_ORIGINAL)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains('torch.operator "torch.npu.npu_rms_norm"'), (
        checker.error
        or "Expected torch.npu.npu_rms_norm after torch-fuse-rms-norm (mixed precision, original)"
    )


# Training graph: rsqrt result returned for backward pass (multi-use rsqrt)
# Simplified from PM02 QKV RmsNorm pattern: bf16 input, f32 compute, bf16 output, rsqrt returned.
MLIR_RMS_NORM_TRAINING_RSQRT_RETURNED_ORIGINAL = textwrap.dedent(
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


def test_torch_fuse_rms_norm_training_rsqrt_returned_original():
    """Training graph (original simplified) with rsqrt returned should still fuse."""
    result = fuse_and_optimize(MLIR_RMS_NORM_TRAINING_RSQRT_RETURNED_ORIGINAL)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains('torch.operator "torch.npu.npu_rms_norm"'), (
        checker.error
        or "Expected torch.npu.npu_rms_norm in training graph with rsqrt returned (original)"
    )


# When normalized value has multiple uses, fusion should not apply; mul/rsqrt remain.
MLIR_RMS_NORM_NO_FUSION_MULTIPLE_USES_ORIGINAL = textwrap.dedent(
    """
module {
  func.func @main(%x: !torch.vtensor<[1,2],f32>, %gamma: !torch.vtensor<[1,2],f32>) -> (!torch.vtensor<[1,2],f32>, !torch.vtensor<[1,2],f32>) attributes {torch.assume_strict_symbolic_shapes} {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05
    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[1,2],f32>, !torch.int -> !torch.vtensor<[1,2],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[1,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[1,2,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,2,1],f32>
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[1,2,1],f32> -> !torch.vtensor<[1,2,1],f32>
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[1,2],f32>, !torch.vtensor<[1,2,1],f32> -> !torch.vtensor<[1,2],f32>
    %out = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[1,2],f32>, !torch.vtensor<[1,2],f32> -> !torch.vtensor<[1,2],f32>
    return %out, %norm : !torch.vtensor<[1,2],f32>, !torch.vtensor<[1,2],f32>
  }
}
"""
)


def test_torch_fuse_rms_norm_no_fusion_multiple_uses_original():
    """When normalized value has multiple uses, fusion should not apply (original simplified)."""
    result = fuse_and_optimize(MLIR_RMS_NORM_NO_FUSION_MULTIPLE_USES_ORIGINAL)
    checker = MlirChecker.parse_torch_module(result)
    assert not checker.check_text_contains(
        'torch.operator "torch.npu.npu_rms_norm"'
    ), "Normalized value has multiple uses; torch-fuse-rms-norm should NOT fuse this pattern (original)"
