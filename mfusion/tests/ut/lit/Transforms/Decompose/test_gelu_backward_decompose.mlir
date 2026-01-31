// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=gelu_backward" -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @gelu_backward_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // geluBackwardOp
  %mode = mfuse.constant.string "tanh" : !mfuse.string
  %0 = "mfuse.aclnn.gelu_backward"(%arg0, %arg1, %mode) : (tensor<4x4xf32>, tensor<4x4xf32>, !mfuse.string) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.gelu_backward
  // CHECK: mfuse.mul
}


func.func @gelu_backward_no_decompose_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // geluBackwardOp
  %mode = mfuse.constant.string "none" : !mfuse.string
  %0 = "mfuse.aclnn.gelu_backward"(%arg0, %arg1, %mode) : (tensor<4x4xf32>, tensor<4x4xf32>, !mfuse.string) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK: mfuse.aclnn.gelu_backward
}