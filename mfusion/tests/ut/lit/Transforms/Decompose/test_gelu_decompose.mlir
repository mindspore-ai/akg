// RUN: mfusion-opt %s -decompose -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @gelu_test(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // geluOp
  %0 = "mfuse.aclnn.gelu"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.gelu
  // CHECK: mfuse.exp
}