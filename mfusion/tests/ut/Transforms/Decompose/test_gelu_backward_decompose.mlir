// RUN: mfusion-opt %s -decompose -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @gelu_backward_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // geluBackwardOp
  %mode = muse.constant.string "tanh" : !muse.string
  %0 = "muse.aclnn.gelu_backward"(%arg0, %arg1, %mode) : (tensor<4x4xf32>, tensor<4x4xf32>, !muse.string) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: muse.aclnn.gelu_backward
  // CHECK: muse.mul
}


func.func @gelu_backward_no_decompose_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // geluBackwardOp
  %mode = muse.constant.string "none" : !muse.string
  %0 = "muse.aclnn.gelu_backward"(%arg0, %arg1, %mode) : (tensor<4x4xf32>, tensor<4x4xf32>, !muse.string) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK: muse.aclnn.gelu_backward
}