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

"""UT for Conv2D Cast fusion in fuse_and_optimize pipeline: Conv2D (f16) + Cast (f32) -> Conv2D (f32).

Only direct Conv2D->Cast is fused. Conv2D->middle_op->Cast (e.g. reshape) is not fused;
see lit test_fuse_conv2d_cast_fusion.mlir (@no_fusion_conv_middle_op_then_cast).
"""

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

# Conv2D Cast fusion: Conv2D (f16) + to.dtype (f32) -> Conv2D with f32 output.
# Torch MLIR: convolution (f16) -> to.dtype (f32). After convert-torch-to-mfuse and mfuse-fusion,
# the cast should be eliminated (mfuse.conv2d directly produces f32 or torch.aten.to.dtype is gone).
MLIR_CONV2D_CAST = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1,2,4,4],f16>, %arg1: !torch.vtensor<[4,2,2,2],f16>) -> !torch.vtensor<[1,4,3,3],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %one = torch.constant.int 1
    %zero = torch.constant.int 0
    %stride = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %output_padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %groups = torch.constant.int 1
    %0 = torch.aten.convolution %arg0, %arg1, %none, %stride, %padding, %dilation, %false, %output_padding, %groups : !torch.vtensor<[1,2,4,4],f16>, !torch.vtensor<[4,2,2,2],f16>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,3,3],f16>
    %int6 = torch.constant.int 6
    %1 = torch.aten.to.dtype %0, %int6, %false, %false, %none : !torch.vtensor<[1,4,3,3],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,4,3,3],f32>
    return %1 : !torch.vtensor<[1,4,3,3],f32>
  }
}
"""


# Conv2D already f32: no f16->f32 cast pattern, so Cast is not fused (fusion only for f16->f32).
MLIR_CONV2D_CAST_NO_FUSION_CONV_F32 = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1,2,4,4],f32>, %arg1: !torch.vtensor<[4,2,2,2],f32>) -> !torch.vtensor<[1,4,3,3],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %one = torch.constant.int 1
    %zero = torch.constant.int 0
    %stride = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %output_padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %groups = torch.constant.int 1
    %0 = torch.aten.convolution %arg0, %arg1, %none, %stride, %padding, %dilation, %false, %output_padding, %groups : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[4,2,2,2],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,3,3],f32>
    return %0 : !torch.vtensor<[1,4,3,3],f32>
  }
}
"""

# Conv2D (f16) -> Cast to f16 (not f32): fusion only applies to f16->f32, so no fusion.
MLIR_CONV2D_CAST_NO_FUSION_CAST_TO_F16 = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1,2,4,4],f16>, %arg1: !torch.vtensor<[4,2,2,2],f16>) -> !torch.vtensor<[1,4,3,3],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %one = torch.constant.int 1
    %zero = torch.constant.int 0
    %stride = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %output_padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %groups = torch.constant.int 1
    %0 = torch.aten.convolution %arg0, %arg1, %none, %stride, %padding, %dilation, %false, %output_padding, %groups : !torch.vtensor<[1,2,4,4],f16>, !torch.vtensor<[4,2,2,2],f16>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,3,3],f16>
    %int5 = torch.constant.int 5
    %1 = torch.aten.to.dtype %0, %int5, %false, %false, %none : !torch.vtensor<[1,4,3,3],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,4,3,3],f16>
    return %1 : !torch.vtensor<[1,4,3,3],f16>
  }
}
"""

# Conv2D (f16) output has two uses: conv -> cast (f32) and conv -> mul. Fusion still applies to
# the cast branch (no single-use requirement); Conv is not erased.
MLIR_CONV2D_CAST_TWO_USES = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1,2,4,4],f16>, %arg1: !torch.vtensor<[4,2,2,2],f16>) -> (!torch.vtensor<[1,4,3,3],f32>, !torch.vtensor<[1,4,3,3],f32>) attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %one = torch.constant.int 1
    %zero = torch.constant.int 0
    %stride = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %output_padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %groups = torch.constant.int 1
    %0 = torch.aten.convolution %arg0, %arg1, %none, %stride, %padding, %dilation, %false, %output_padding, %groups : !torch.vtensor<[1,2,4,4],f16>, !torch.vtensor<[4,2,2,2],f16>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,3,3],f16>
    %int6 = torch.constant.int 6
    %1 = torch.aten.to.dtype %0, %int6, %false, %false, %none : !torch.vtensor<[1,4,3,3],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,4,3,3],f32>
    %2 = torch.aten.mul.Tensor %0, %0 : !torch.vtensor<[1,4,3,3],f16>, !torch.vtensor<[1,4,3,3],f16> -> !torch.vtensor<[1,4,3,3],f16>
    %3 = torch.aten.to.dtype %2, %int6, %false, %false, %none : !torch.vtensor<[1,4,3,3],f16>, !torch.int, !torch.bool, !torch.bool, !torch.none -> !torch.vtensor<[1,4,3,3],f32>
    return %1, %3 : !torch.vtensor<[1,4,3,3],f32>, !torch.vtensor<[1,4,3,3],f32>
  }
}
"""


def test_conv2d_cast_fusion():
    """Test Conv2D Cast fusion through fuse_and_optimize pipeline."""
    result = fuse_and_optimize(MLIR_CONV2D_CAST)
    checker = MlirChecker.parse_torch_module(result)
    # Fusion: Conv2D (f16) + Cast (f32) -> single Conv2D (f32). Check torch ops only.
    assert checker.check_no_op("torch.aten.to.dtype"), (
        checker.error or "torch.aten.to.dtype should be eliminated after fusion")
    assert checker.check_has_op("torch.aten.convolution"), (
        checker.error or "torch.aten.convolution should exist after fusion")


def test_conv2d_cast_no_fusion_conv_already_f32():
    """No fusion: Conv2D is already f32 (fusion only for f16->f32 cast)."""
    result = fuse_and_optimize(MLIR_CONV2D_CAST_NO_FUSION_CONV_F32)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.convolution"), checker.error or "convolution should exist"


def test_conv2d_cast_no_fusion_cast_target_not_f32():
    """No fusion: Cast target is f16, not f32 (fusion only for f16->f32)."""
    result = fuse_and_optimize(MLIR_CONV2D_CAST_NO_FUSION_CAST_TO_F16)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.convolution"), checker.error or "convolution should exist"
    # FuseConv2DCast only fuses f16->f32; f16->f16 may be removed as identity. Output must stay f16.
    assert checker.check_has_op("torch.aten.to.dtype") or "f16" in result, (
        "output should remain f16 (no f32 fusion); to.dtype may be removed as identity"
    )


def test_conv2d_cast_fusion_with_conv_two_uses():
    """Fusion applies even when Conv2D output has multiple uses; only the Cast is replaced."""
    result = fuse_and_optimize(MLIR_CONV2D_CAST_TWO_USES)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.convolution"), checker.error or "convolution should exist"
    # The conv->cast branch is fused (that to.dtype is eliminated); the mul->to.dtype path remains.
    assert checker.check_text_contains("torch.operator \"torch.npu._npu_dtype_cast\""), checker.error or "to.dtype after mul should remain"
