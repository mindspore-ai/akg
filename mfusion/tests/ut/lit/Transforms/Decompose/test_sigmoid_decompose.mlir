// RUN: mfusion-opt %s -decompose -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @sigmoid_test(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // sigmoidOp
  %0 = "mfuse.aclnn.sigmoid"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.sigmoid
  // CHECK: mfuse.exp
}
