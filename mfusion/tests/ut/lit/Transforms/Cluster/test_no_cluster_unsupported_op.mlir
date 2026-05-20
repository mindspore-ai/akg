// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// Test scenario: Single operation should not be clustered (no fusion benefit)
// CHECK-LABEL: func @test_single_op_no_fusion
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-NOT: mfuse.fused
// CHECK: %[[ADD:.*]] = mfuse.add %arg0, %arg1
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: return %[[ADD]]
func.func @test_single_op_no_fusion(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Single add operation - should not create a fused op (only one op)
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
}