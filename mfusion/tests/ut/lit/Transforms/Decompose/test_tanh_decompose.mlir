// RUN: mfusion-opt %s -decompose -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @tanh_test(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // decompose tanhOp with f32 input
  %0 = "muse.aclnn.tanh"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: muse.aclnn.tanh
  // CHECK: muse.add
}

func.func @tanh_test_no_decompose(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
  // do not decompose tanhOp with f64 input
  %0 = "muse.aclnn.tanh"(%arg0) : (tensor<4x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
  // CHECK: muse.aclnn.tanh
}