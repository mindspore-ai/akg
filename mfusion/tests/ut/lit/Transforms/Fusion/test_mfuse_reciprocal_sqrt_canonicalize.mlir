// RUN: mfusion-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @fold_reciprocal_sqrt
// CHECK: %[[RSQRT:.+]] = mfuse.rsqrt %arg0 : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT: return %[[RSQRT]] : tensor<2x4xf32>
func.func @fold_reciprocal_sqrt(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %sqrt = mfuse.sqrt %arg0 : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %rec = mfuse.reciprocal %sqrt : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %rec : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @keep_plain_reciprocal
// CHECK: %[[REC:.+]] = mfuse.reciprocal %arg0 : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NEXT: return %[[REC]] : tensor<2x4xf32>
func.func @keep_plain_reciprocal(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %rec = mfuse.reciprocal %arg0 : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %rec : tensor<2x4xf32>
}
