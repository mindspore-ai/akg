// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// Test scenario: Clustering operations that produce multiple outputs
// A cluster can yield multiple intermediate values as outputs
// CHECK-LABEL: func @test_multi_output_cluster
// CHECK: %[[FUSED:.*]]:2 = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG2]]
// CHECK: %[[SUB:.*]] = mfuse.sub %[[ADD]], %[[ARG3]]
// CHECK: mfuse.yield %[[MUL]], %[[SUB]]
// CHECK: return %[[FUSED]]#0, %[[FUSED]]#1
func.func @test_multi_output_cluster(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  // Add operation
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // Mul operation (uses add result)
  %1 = mfuse.mul %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // Sub operation (uses add result)
  %2 = mfuse.sub %0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1, %2 : tensor<4x4xf32>, tensor<4x4xf32>
}
}