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

"""UT for BiasAdd Conv fusion in fuse_and_optimize pipeline: Conv2D (no bias) + Add (bias) -> Conv2DWithBias."""

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

# BiasAdd Conv fusion: Conv2D (no bias) + Add (bias) -> Conv2DWithBias.
# Torch MLIR: convolution (no bias) -> add.Tensor (bias). After mfuse-fusion, mfuse.add is eliminated.
MLIR_BIASADD_CONV = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1,2,4,4],f32>, %arg1: !torch.vtensor<[4,2,2,2],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,3,3],f32> attributes {torch.assume_strict_symbolic_shapes} {
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
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %arg2, %int1 : !torch.vtensor<[1,4,3,3],f32>, !torch.vtensor<[4],f32>, !torch.int -> !torch.vtensor<[1,4,3,3],f32>
    return %1 : !torch.vtensor<[1,4,3,3],f32>
  }
}
"""

# Conv + Add but bias shape [2] (out channels 4): bias size must match, so no fusion.
MLIR_BIASADD_CONV_NO_FUSION_BIAS_WRONG_SIZE = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1,2,4,4],f32>, %arg1: !torch.vtensor<[4,2,2,2],f32>, %arg2: !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,4,3,3],f32> attributes {torch.assume_strict_symbolic_shapes} {
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
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %arg2, %int1 : !torch.vtensor<[1,4,3,3],f32>, !torch.vtensor<[2],f32>, !torch.int -> !torch.vtensor<[1,4,3,3],f32>
    return %1 : !torch.vtensor<[1,4,3,3],f32>
  }
}
"""

# Conv + Add but bias is 2D [4,1]: pass requires bias rank 1, so no fusion.
MLIR_BIASADD_CONV_NO_FUSION_BIAS_NOT_1D = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1,2,4,4],f32>, %arg1: !torch.vtensor<[4,2,2,2],f32>, %arg2: !torch.vtensor<[4,1],f32>) -> !torch.vtensor<[1,4,3,3],f32> attributes {torch.assume_strict_symbolic_shapes} {
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
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %arg2, %int1 : !torch.vtensor<[1,4,3,3],f32>, !torch.vtensor<[4,1],f32>, !torch.int -> !torch.vtensor<[1,4,3,3],f32>
    return %1 : !torch.vtensor<[1,4,3,3],f32>
  }
}
"""

# Conv + Add(conv, view(bias)): bias 1D [4] viewed to [1,4,1,1]. Pass only accepts 1D [C], so no fusion (bias is view output).
MLIR_BIASADD_CONV_BIAS_FROM_VIEW = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1,2,4,4],f32>, %arg1: !torch.vtensor<[4,2,2,2],f32>, %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,3,3],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %one = torch.constant.int 1
    %zero = torch.constant.int 0
    %four = torch.constant.int 4
    %stride = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %output_padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %groups = torch.constant.int 1
    %0 = torch.aten.convolution %arg0, %arg1, %none, %stride, %padding, %dilation, %false, %output_padding, %groups : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[4,2,2,2],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,4,3,3],f32>
    %shape_list = torch.prim.ListConstruct %one, %four, %one, %one : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %bias_viewed = torch.aten.view %arg2, %shape_list : !torch.vtensor<[4],f32>, !torch.list<int> -> !torch.vtensor<[1,4,1,1],f32>
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %bias_viewed, %int1 : !torch.vtensor<[1,4,3,3],f32>, !torch.vtensor<[1,4,1,1],f32>, !torch.int -> !torch.vtensor<[1,4,3,3],f32>
    return %1 : !torch.vtensor<[1,4,3,3],f32>
  }
}
"""


def test_bias_add_conv_fusion():
    """Test BiasAdd Conv fusion through fuse_and_optimize pipeline.

    Conv2D (no bias) + Add (bias) is fused to Conv2DWithBias; convert-mfuse-to-torch
    lowers conv2d_with_bias to torch.aten.convolution with bias, so add.Tensor is eliminated.
    """
    result = fuse_and_optimize(MLIR_BIASADD_CONV)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_no_op("torch.aten.add.Tensor"), (
        checker.error or "torch.aten.add.Tensor (bias add) should be eliminated after fusion")
    assert checker.check_has_op("torch.aten.convolution"), (
        checker.error or "torch.aten.convolution should exist after fusion")


def test_bias_add_conv_no_fusion_bias_wrong_size():
    """No fusion: bias shape [2] does not match conv output channels 4."""
    result = fuse_and_optimize(MLIR_BIASADD_CONV_NO_FUSION_BIAS_WRONG_SIZE)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.add.Tensor"), (
        checker.error or "add.Tensor should remain when bias size mismatch")
    assert checker.check_has_op("torch.aten.convolution"), checker.error or "convolution should exist"


def test_bias_add_conv_no_fusion_bias_not_1d():
    """No fusion: bias is 2D [4,1], pass requires 1D [C]."""
    result = fuse_and_optimize(MLIR_BIASADD_CONV_NO_FUSION_BIAS_NOT_1D)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.add.Tensor"), (
        checker.error or "add.Tensor should remain when bias not 1D")
    assert checker.check_has_op("torch.aten.convolution"), checker.error or "convolution should exist"


def test_bias_add_conv_fusion_bias_from_view():
    """No fusion when bias comes from view: Conv + Add(conv, view(bias_1d)).

    Pass only accepts 1D [C] bias (no Reshape/view); view output is 4D [1,4,1,1], so fusion is not applied.
    add.Tensor and view remain.
    """
    result = fuse_and_optimize(MLIR_BIASADD_CONV_BIAS_FROM_VIEW)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.add.Tensor"), (
        checker.error or "torch.aten.add.Tensor should remain when bias is view output (pass requires 1D bias)"
    )
    assert checker.check_has_op("torch.aten.convolution"), (
        checker.error or "torch.aten.convolution should exist"
    )
