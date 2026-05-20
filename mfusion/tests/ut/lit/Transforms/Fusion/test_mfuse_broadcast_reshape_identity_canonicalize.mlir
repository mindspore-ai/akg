// RUN: mfusion-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @broadcast_to_identity
// CHECK-NOT: mfuse.broadcast_to
// CHECK: return %arg0
func.func @broadcast_to_identity(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = mfuse.broadcast_to %arg0 : (tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @reshape_identity
// CHECK-NOT: mfuse.reshape
// CHECK: return %arg0
func.func @reshape_identity(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> {
  %0 = mfuse.reshape %arg0 : (tensor<3x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}
