// RUN: mfusion-opt %s --canonicalize | FileCheck %s
//
// Redundant mfuse.reshape chains: reshape(reshape(a, shape1), shape2) -> reshape(a, shape2)

// CHECK-LABEL: func.func @reshape_reshape_canonicalize
// CHECK: mfuse.reshape
// CHECK-NOT: mfuse.reshape
func.func @reshape_reshape_canonicalize(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  %a = mfuse.reshape %arg0 : (tensor<2x4xf32>) -> tensor<4x2xf32>
  %b = mfuse.reshape %a : (tensor<4x2xf32>) -> tensor<8xf32>
  return %b : tensor<8xf32>
}

// CHECK-LABEL: func.func @reshape_reshape_dynamic
// CHECK: mfuse.reshape
// CHECK-NOT: mfuse.reshape
func.func @reshape_reshape_dynamic(%arg0: tensor<?x4xf32>) -> tensor<?xf32> {
  %a = mfuse.reshape %arg0 : (tensor<?x4xf32>) -> tensor<?x2x2xf32>
  %b = mfuse.reshape %a : (tensor<?x2x2xf32>) -> tensor<?xf32>
  return %b : tensor<?xf32>
}
