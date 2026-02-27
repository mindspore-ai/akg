// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=gelu_backward" -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @gelu_backward_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // geluBackwardOp
  %0 = mfuse.aclnn.gelu_backward %arg0, %arg1 {approximate = "tanh"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.gelu_backward
  // CHECK: mfuse.mul
}
