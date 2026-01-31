// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=tanh" -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @tanh_test(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // decompose tanhOp with f32 input
  %0 = "mfuse.aclnn.tanh"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.tanh
  // CHECK: mfuse.add
}

func.func @tanh_test_no_decompose(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
  // do not decompose tanhOp with f64 input
  %0 = "mfuse.aclnn.tanh"(%arg0) : (tensor<4x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
  // CHECK: mfuse.aclnn.tanh
}