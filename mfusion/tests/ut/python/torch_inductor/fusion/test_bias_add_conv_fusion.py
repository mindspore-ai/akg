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

"""UT for BiasAdd Conv fusion in fuse_and_optimize pipeline.

Guard / shape / dtype edge cases are covered by mfusion-opt lit tests.
This file keeps end-to-end smoke: bare bias and view->[1,C,1,1].
"""

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker

# Conv2D (no bias) + Add (bias [C]) -> fused; add.Tensor eliminated.
MLIR_BIASADD_CONV = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1,2,4,4],f32>, %arg1: !torch.vtensor<[4,2,2,2],f32>,
                  %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,3,3],f32>
      attributes {torch.assume_strict_symbolic_shapes} {
    %none = torch.constant.none
    %one = torch.constant.int 1
    %zero = torch.constant.int 0
    %stride = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %dilation = torch.prim.ListConstruct %one, %one : (!torch.int, !torch.int) -> !torch.list<int>
    %false = torch.constant.bool false
    %output_padding = torch.prim.ListConstruct %zero, %zero : (!torch.int, !torch.int) -> !torch.list<int>
    %groups = torch.constant.int 1
    %0 = torch.aten.convolution %arg0, %arg1, %none, %stride, %padding, %dilation, %false,
        %output_padding, %groups
        : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[4,2,2,2],f32>, !torch.none,
          !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>,
          !torch.int -> !torch.vtensor<[1,4,3,3],f32>
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %arg2, %int1
        : !torch.vtensor<[1,4,3,3],f32>, !torch.vtensor<[4],f32>, !torch.int
          -> !torch.vtensor<[1,4,3,3],f32>
    return %1 : !torch.vtensor<[1,4,3,3],f32>
  }
}
"""

# Bias viewed to [1,C,1,1]: frontend path distinct from lit mfuse.reshape.
MLIR_BIASADD_CONV_BIAS_FROM_VIEW = r"""
module {
  func.func @main(%arg0: !torch.vtensor<[1,2,4,4],f32>, %arg1: !torch.vtensor<[4,2,2,2],f32>,
                  %arg2: !torch.vtensor<[4],f32>) -> !torch.vtensor<[1,4,3,3],f32>
      attributes {torch.assume_strict_symbolic_shapes} {
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
    %0 = torch.aten.convolution %arg0, %arg1, %none, %stride, %padding, %dilation, %false,
        %output_padding, %groups
        : !torch.vtensor<[1,2,4,4],f32>, !torch.vtensor<[4,2,2,2],f32>, !torch.none,
          !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>,
          !torch.int -> !torch.vtensor<[1,4,3,3],f32>
    %shape_list = torch.prim.ListConstruct %one, %four, %one, %one
        : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %bias_viewed = torch.aten.view %arg2, %shape_list
        : !torch.vtensor<[4],f32>, !torch.list<int> -> !torch.vtensor<[1,4,1,1],f32>
    %int1 = torch.constant.int 1
    %1 = torch.aten.add.Tensor %0, %bias_viewed, %int1
        : !torch.vtensor<[1,4,3,3],f32>, !torch.vtensor<[1,4,1,1],f32>, !torch.int
          -> !torch.vtensor<[1,4,3,3],f32>
    return %1 : !torch.vtensor<[1,4,3,3],f32>
  }
}
"""


def test_bias_add_conv_fusion():
    """E2E: Conv + Add(bias[C]) fused; add.Tensor eliminated."""
    result = fuse_and_optimize(MLIR_BIASADD_CONV)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_no_op("torch.aten.add.Tensor"), (
        checker.error or "torch.aten.add.Tensor (bias add) should be eliminated after fusion")
    assert checker.check_has_op("torch.aten.convolution"), (
        checker.error or "torch.aten.convolution should exist after fusion")


def test_bias_add_conv_fusion_bias_from_view():
    """E2E: Conv + Add(view(bias->[1,C,1,1])) fused; add.Tensor eliminated."""
    result = fuse_and_optimize(MLIR_BIASADD_CONV_BIAS_FROM_VIEW)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_no_op("torch.aten.add.Tensor"), (
        checker.error or "torch.aten.add.Tensor should be eliminated after conv+bias fusion")
    assert checker.check_has_op("torch.aten.convolution"), (
        checker.error or "torch.aten.convolution should exist after fusion")
