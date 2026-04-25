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

"""UT for TorchFuseLayerNorm: fuse LayerNorm patterns on Torch dialect."""
# pylint: disable=line-too-long

import textwrap

from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker

# Complete LayerNorm formula: var_mean(x) → add(eps) → rsqrt → sub(x, mean) → mul → mul(gamma) → add(beta)


def test_torch_fuse_layer_norm_complete():
    """Complete LayerNorm formula should fuse."""
    mlir_input = textwrap.dedent(
        """
module {
  func.func @main(%x: !torch.vtensor<[4,197,384],f32>, %gamma: !torch.vtensor<[384],f32>, %beta: !torch.vtensor<[384],f32>) -> !torch.vtensor<[4,197,384],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int2 = torch.constant.int 2
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %eps = torch.constant.float 9.9999999999999995E-7
    %none = torch.constant.none
    // Step 1: mean(x)
    %dims = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
    %result0, %result1 = torch.aten.var_mean.correction %x, %dims, %int0, %true : !torch.vtensor<[4,197,384],f32>, !torch.list<int>, !torch.int, !torch.bool -> !torch.vtensor<[4,197,1],f32>, !torch.vtensor<[4,197,1],f32>
    // Step 2: add(eps)
    %15 = torch.aten.add.Scalar %result0, %eps, %int1 : !torch.vtensor<[4,197,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,197,1],f32>
    // Step 3: rsqrt
    %16 = torch.aten.rsqrt %15 : !torch.vtensor<[4,197,1],f32> -> !torch.vtensor<[4,197,1],f32>
    // Step 4: sub(x, mean)
    %17 = torch.aten.sub.Tensor %x, %result1, %int1 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[4,197,1],f32>, !torch.int -> !torch.vtensor<[4,197,384],f32>
    // Step 5: mul(x - mean, rsqrt)
    %18 = torch.aten.mul.Tensor %17, %16 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[4,197,1],f32> -> !torch.vtensor<[4,197,384],f32>
    // Step 6: mul(gamma)
    %19 = torch.aten.mul.Tensor %18, %gamma : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[384],f32> -> !torch.vtensor<[4,197,384],f32>
    // Step 7: add(beta)
    %20 = torch.aten.add.Tensor %19, %beta, %int1 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[4,197,384],f32>
    return %20 : !torch.vtensor<[4,197,384],f32>
  }
}
"""
    )
    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.layer_norm")
    # Verify the original operations are fused away
    assert checker.check_no_op("torch.aten.var.dim")
    assert checker.check_no_op("torch.aten.mean.dim")
    assert checker.check_no_op("torch.aten.add.Scalar")
    assert checker.check_no_op("torch.aten.rsqrt")
    assert checker.check_no_op("torch.aten.sub.Tensor")
    assert checker.check_no_op("torch.aten.mul.Tensor")
    assert checker.check_no_op("torch.aten.add.Tensor")


# LayerNorm should not fuse when add alpha is not 1


def test_torch_fuse_layer_norm_no_fusion_wrong_add_alpha():
    """LayerNorm should not fuse when add alpha is not 1."""
    mlir_input = textwrap.dedent(
        """
module {
  func.func @main(%x: !torch.vtensor<[4,197,384],f32>, %gamma: !torch.vtensor<[384],f32>, %beta: !torch.vtensor<[384],f32>) -> !torch.vtensor<[4,197,384],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %int2 = torch.constant.int 2  // Wrong alpha
    %eps = torch.constant.float 9.9999999999999995E-7
    %none = torch.constant.none
    // Step 1: mean(x)
    %dims = torch.prim.ListConstruct %int2 : (!torch.int) -> !torch.list<int>
    %result0, %result1 = torch.aten.var_mean.correction %x, %dims, %int0, %true : !torch.vtensor<[4,197,384],f32>, !torch.list<int>, !torch.int, !torch.bool -> !torch.vtensor<[4,197,1],f32>, !torch.vtensor<[4,197,1],f32>
    // Step 2: add(eps) with wrong alpha
    %15 = torch.aten.add.Scalar %result0, %eps, %int2 : !torch.vtensor<[4,197,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,197,1],f32>
    // Step 3: rsqrt
    %16 = torch.aten.rsqrt %15 : !torch.vtensor<[4,197,1],f32> -> !torch.vtensor<[4,197,1],f32>
    // Step 4: sub(x, mean)
    %int1 = torch.constant.int 1
    %17 = torch.aten.sub.Tensor %x, %result1, %int1 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[4,197,1],f32>, !torch.int -> !torch.vtensor<[4,197,384],f32>
    // Step 5: mul(x - mean, rsqrt)
    %18 = torch.aten.mul.Tensor %17, %16 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[4,197,1],f32> -> !torch.vtensor<[4,197,384],f32>
    // Step 6: mul(gamma)
    %19 = torch.aten.mul.Tensor %18, %gamma : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[384],f32> -> !torch.vtensor<[4,197,384],f32>
    // Step 7: add(beta)
    %20 = torch.aten.add.Tensor %19, %beta, %int1 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[4,197,384],f32>
    return %20 : !torch.vtensor<[4,197,384],f32>
  }
}
"""
    )
    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_no_op("torch.aten.layer_norm")


# LayerNorm should not fuse when normalized value has multiple uses


def test_torch_fuse_layer_norm_no_fusion_multiple_uses():
    """LayerNorm should not fuse when normalized value has multiple uses."""
    mlir_input = textwrap.dedent(
        """
module {
  func.func @main(%x: !torch.vtensor<[4,197,384],f32>, %gamma: !torch.vtensor<[384],f32>, %beta: !torch.vtensor<[384],f32>) -> (!torch.vtensor<[4,197,384],f32>, !torch.vtensor<[4,197,384],f32>) attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %true = torch.constant.bool true
    %int1 = torch.constant.int 1
    %eps = torch.constant.float 9.9999999999999995E-7
    %none = torch.constant.none
    // Step 1: mean(x)
    %dims = torch.prim.ListConstruct %int0 : (!torch.int) -> !torch.list<int>
    %result0, %result1 = torch.aten.var_mean.correction %x, %dims, %int0, %true : !torch.vtensor<[4,197,384],f32>, !torch.list<int>, !torch.int, !torch.bool -> !torch.vtensor<[4,197,1],f32>, !torch.vtensor<[4,197,1],f32>
    // Step 2: add(eps)
    %15 = torch.aten.add.Scalar %result0, %eps, %int1 : !torch.vtensor<[4,197,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,197,1],f32>
    // Step 3: rsqrt
    %16 = torch.aten.rsqrt %15 : !torch.vtensor<[4,197,1],f32> -> !torch.vtensor<[4,197,1],f32>
    // Step 4: sub(x, mean)
    %17 = torch.aten.sub.Tensor %x, %result1, %int1 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[4,197,1],f32>, !torch.int -> !torch.vtensor<[4,197,384],f32>
    // Step 5: mul(x - mean, rsqrt)
    %18 = torch.aten.mul.Tensor %17, %16 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[4,197,1],f32> -> !torch.vtensor<[4,197,384],f32>
    // Multiple uses of normalized value - should prevent fusion
    %19 = torch.aten.mul.Tensor %18, %gamma : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[384],f32> -> !torch.vtensor<[4,197,384],f32>
    %20 = torch.aten.add.Tensor %19, %beta, %int1 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[4,197,384],f32>
    return %20, %18 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[4,197,384],f32>
  }
}
"""
    )
    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_no_op("torch.aten.layer_norm")


def test_torch_fuse_layer_norm_no_fusion_duplicate_dims():
    """LayerNorm should not fuse when dims contain duplicates."""
    mlir_input = textwrap.dedent(
        """
module {
  func.func @main(%x: !torch.vtensor<[4,197,384],f32>, %gamma: !torch.vtensor<[384],f32>, %beta: !torch.vtensor<[384],f32>) -> !torch.vtensor<[4,197,384],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int-1 = torch.constant.int -1
    %true = torch.constant.bool true
    %eps = torch.constant.float 9.9999999999999995E-7
    %none = torch.constant.none
    // Wrong dims: [2, 2] contains duplicate dimensions
    %dims = torch.prim.ListConstruct %int2, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %result0, %result1 = torch.aten.var_mean.correction %x, %dims, %int0, %true : !torch.vtensor<[4,197,384],f32>, !torch.list<int>, !torch.int, !torch.bool -> !torch.vtensor<[4,197,1],f32>, !torch.vtensor<[4,197,1],f32>
    // Step 2: add(eps)
    %15 = torch.aten.add.Scalar %result0, %eps, %int1 : !torch.vtensor<[4,197,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[4,197,1],f32>
    // Step 3: rsqrt
    %16 = torch.aten.rsqrt %15 : !torch.vtensor<[4,197,1],f32> -> !torch.vtensor<[4,197,1],f32>
    // Step 4: sub(x, mean)
    %17 = torch.aten.sub.Tensor %x, %result1, %int1 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[4,197,1],f32>, !torch.int -> !torch.vtensor<[4,197,384],f32>
    // Step 5: mul(x - mean, rsqrt)
    %18 = torch.aten.mul.Tensor %17, %16 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[4,197,1],f32> -> !torch.vtensor<[4,197,384],f32>
    // Step 6: mul(gamma)
    %19 = torch.aten.mul.Tensor %18, %gamma : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[384],f32> -> !torch.vtensor<[4,197,384],f32>
    // Step 7: add(beta)
    %20 = torch.aten.add.Tensor %19, %beta, %int1 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[384],f32>, !torch.int -> !torch.vtensor<[4,197,384],f32>
    return %20 : !torch.vtensor<[4,197,384],f32>
  }
}
"""
    )
    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_no_op("torch.aten.layer_norm")


def test_torch_fuse_layer_norm_fusion_multi_dims():
    """LayerNorm should fuse when multi dims are valid."""
    mlir_input = textwrap.dedent(
        """
module {
  func.func @main(%x: !torch.vtensor<[4,197,384],f32>, %gamma: !torch.vtensor<[197,384],f32>, %beta: !torch.vtensor<[197,384],f32>) -> !torch.vtensor<[4,197,384],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %int-2 = torch.constant.int -2
    %true = torch.constant.bool true
    %eps = torch.constant.float 9.9999999999999995E-7
    %none = torch.constant.none
    %dims = torch.prim.ListConstruct %int2, %int-2 : (!torch.int, !torch.int) -> !torch.list<int>
    %result0, %result1 = torch.aten.var_mean.correction %x, %dims, %int0, %true : !torch.vtensor<[4,197,384],f32>, !torch.list<int>, !torch.int, !torch.bool -> !torch.vtensor<[1,197,384],f32>, !torch.vtensor<[1,197,384],f32>
    // Step 2: add(eps)
    %15 = torch.aten.add.Scalar %result0, %eps, %int1 : !torch.vtensor<[1,197,384],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,197,384],f32>
    // Step 3: rsqrt
    %16 = torch.aten.rsqrt %15 : !torch.vtensor<[1,197,384],f32> -> !torch.vtensor<[1,197,384],f32>
    // Step 4: sub(x, mean)
    %17 = torch.aten.sub.Tensor %x, %result1, %int1 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[1,197,384],f32>, !torch.int -> !torch.vtensor<[4,197,384],f32>
    // Step 5: mul(x - mean, rsqrt)
    %18 = torch.aten.mul.Tensor %17, %16 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[1,197,384],f32> -> !torch.vtensor<[4,197,384],f32>
    // Step 6: mul(gamma)
    %19 = torch.aten.mul.Tensor %18, %gamma : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[197,384],f32> -> !torch.vtensor<[4,197,384],f32>
    // Step 7: add(beta)
    %20 = torch.aten.add.Tensor %19, %beta, %int1 : !torch.vtensor<[4,197,384],f32>, !torch.vtensor<[197,384],f32>, !torch.int -> !torch.vtensor<[4,197,384],f32>
    return %20 : !torch.vtensor<[4,197,384],f32>
  }
}
"""
    )
    result = fuse_and_optimize(mlir_input)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_has_op("torch.aten.layer_norm")
