// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=sigmoid" -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @test_sigmoid(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.aclnn.sigmoid %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.sigmoid
  // CHECK: mfuse.exp
}
