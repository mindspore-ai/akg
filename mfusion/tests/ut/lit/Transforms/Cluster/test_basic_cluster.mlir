// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// Basic cluster test: Simple sequence of supported operations should be clustered
// CHECK-LABEL: func @test_basic_add_mul_cluster
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG2]]
// CHECK: mfuse.yield %[[MUL]]
// CHECK: return %[[FUSED]]
func.func @test_basic_add_mul_cluster(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Add operation
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // Mul operation (uses add result)
  %1 = mfuse.mul %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Multiple operation chains that merge into a single cluster
// Operations from different chains can be clustered together when they have a common consumer
// CHECK-LABEL: func @test_multiple_clusters
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>, %[[ARG5:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD1:.*]] = mfuse.add %[[ARG3]], %[[ARG4]]
// CHECK: %[[MUL1:.*]] = mfuse.mul %[[ADD1]], %[[ARG3]]
// CHECK: %[[ADD2:.*]] = mfuse.add %[[ARG4]], %[[ARG5]]
// CHECK: %[[MUL2:.*]] = mfuse.mul %[[ADD2]], %[[ARG4]]
// CHECK: %[[ADD3:.*]] = mfuse.add %[[MUL1]], %[[MUL2]]
// CHECK: mfuse.yield %[[ADD3]]
// CHECK: return %[[FUSED]]
func.func @test_multiple_clusters(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.mul %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mfuse.add %arg1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %3 = mfuse.mul %2, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %4 = mfuse.add %1, %3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %4 : tensor<4x4xf32>
}

// Test scenario: Long chain of element-wise operations with different tensor sizes
// All arithmetic operations (add, mul, sub, div) can be clustered together
// CHECK-LABEL: func @test_element_wise_chain
// CHECK-SAME: %arg0: tensor<8x8xf32>
// CHECK-SAME: %arg1: tensor<8x8xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<8x8xf32>, %[[ARG3:.*]]: tensor<8x8xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG2]]
// CHECK: %[[SUB:.*]] = mfuse.sub %[[MUL]], %[[ARG3]]
// CHECK: %[[DIV:.*]] = mfuse.div %[[SUB]], %[[ARG2]]
// CHECK: mfuse.yield %[[DIV]]
// CHECK: return %[[FUSED]]
func.func @test_element_wise_chain(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = mfuse.mul %0, %arg0 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %2 = mfuse.sub %1, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %3 = mfuse.div %2, %arg0 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %3 : tensor<8x8xf32>
}
}