// RUN: mfusion-opt %s --canonicalize | FileCheck %s
//
// mfuse.permute chains: identity elimination; permute(permute(x, inner), outer) folded to
// a single permute(x, fused) or to x when fused is identity (Torch-style: out dim i from in dim perm[i]).

// CHECK-LABEL: func.func @drop_identity
// CHECK-NOT: mfuse.permute
// CHECK: return %arg0 : tensor<2x3x4xf32>
func.func @drop_identity(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = mfuse.permute %arg0, [0, 1, 2] : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @double_swap_is_identity
// CHECK-NOT: mfuse.permute
// CHECK: return %arg0 : tensor<3x4xf32>
func.func @double_swap_is_identity(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %a = mfuse.permute %arg0, [1, 0] : (tensor<3x4xf32>) -> tensor<4x3xf32>
  %b = mfuse.permute %a, [1, 0] : (tensor<4x3xf32>) -> tensor<3x4xf32>
  return %b : tensor<3x4xf32>
}

// Inner permute is still used: outer round-trip folds to %arg0; inner stays.
// CHECK-LABEL: func.func @inner_result_shared
// CHECK-DAG: mfuse.permute %arg0, [1, 0]
// CHECK-DAG: return %arg0
func.func @inner_result_shared(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<4x3xf32>) {
  %a = mfuse.permute %arg0, [1, 0] : (tensor<3x4xf32>) -> tensor<4x3xf32>
  %b = mfuse.permute %a, [1, 0] : (tensor<4x3xf32>) -> tensor<3x4xf32>
  return %b, %a : tensor<3x4xf32>, tensor<4x3xf32>
}

// CHECK-LABEL: func.func @compose_to_single
// CHECK-COUNT-1: mfuse.permute
// CHECK: [1, 2, 0]
func.func @compose_to_single(%arg0: tensor<2x3x4xf32>) -> tensor<3x4x2xf32> {
  %a = mfuse.permute %arg0, [0, 2, 1] : (tensor<2x3x4xf32>) -> tensor<2x4x3xf32>
  %b = mfuse.permute %a, [2, 1, 0] : (tensor<2x4x3xf32>) -> tensor<3x4x2xf32>
  return %b : tensor<3x4x2xf32>
}
