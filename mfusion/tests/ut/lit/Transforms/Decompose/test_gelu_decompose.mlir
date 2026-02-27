// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=gelu" -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @test_gelu(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.aclnn.gelu %arg0 {approximate = "none"} : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.gelu
  // CHECK: mfuse.exp
}