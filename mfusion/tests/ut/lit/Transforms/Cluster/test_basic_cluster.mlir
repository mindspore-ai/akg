// RUN: mfusion-opt %s --muse-dvm-cluster | FileCheck %s

module {
// Basic cluster test: Simple sequence of supported operations should be clustered
// CHECK-LABEL: func @test_basic_add_mul_cluster
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = muse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = muse.mul %[[ADD]], %[[ARG2]]
// CHECK: muse.yield %[[MUL]]
// CHECK: return %[[FUSED]]
func.func @test_basic_add_mul_cluster(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Add operation
  %0 = muse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // Mul operation (uses add result)
  %1 = muse.mul %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Multiple operation chains that merge into a single cluster
// Operations from different chains can be clustered together when they have a common consumer
// CHECK-LABEL: func @test_multiple_clusters
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>, %[[ARG5:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD1:.*]] = muse.add %[[ARG3]], %[[ARG4]]
// CHECK: %[[MUL1:.*]] = muse.mul %[[ADD1]], %[[ARG3]]
// CHECK: %[[ADD2:.*]] = muse.add %[[ARG4]], %[[ARG5]]
// CHECK: %[[MUL2:.*]] = muse.mul %[[ADD2]], %[[ARG4]]
// CHECK: %[[ADD3:.*]] = muse.add %[[MUL1]], %[[MUL2]]
// CHECK: muse.yield %[[ADD3]]
// CHECK: return %[[FUSED]]
func.func @test_multiple_clusters(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.mul %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = muse.add %arg1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %3 = muse.mul %2, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %4 = muse.add %1, %3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %4 : tensor<4x4xf32>
}

// Test scenario: Long chain of element-wise operations with different tensor sizes
// All arithmetic operations (add, mul, sub, div) can be clustered together
// CHECK-LABEL: func @test_element_wise_chain
// CHECK-SAME: %arg0: tensor<8x8xf32>
// CHECK-SAME: %arg1: tensor<8x8xf32>
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<8x8xf32>, %[[ARG3:.*]]: tensor<8x8xf32>):
// CHECK: %[[ADD:.*]] = muse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = muse.mul %[[ADD]], %[[ARG2]]
// CHECK: %[[SUB:.*]] = muse.sub %[[MUL]], %[[ARG3]]
// CHECK: %[[DIV:.*]] = muse.div %[[SUB]], %[[ARG2]]
// CHECK: muse.yield %[[DIV]]
// CHECK: return %[[FUSED]]
func.func @test_element_wise_chain(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = muse.add %arg0, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = muse.mul %0, %arg0 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %2 = muse.sub %1, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %3 = muse.div %2, %arg0 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %3 : tensor<8x8xf32>
}
}