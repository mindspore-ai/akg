// RUN: mfusion-opt %s --muse-dvm-cluster | FileCheck %s

module {
// Test comparison operations clustering: eq, ne, gt, ge, lt, le

// Test scenario: Comparison operation (eq) combined with arithmetic operation
// CHECK-LABEL: func @test_eq_comparison
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = muse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[EQ:.*]] = muse.eq %[[ADD]], %[[ARG2]]
// CHECK: muse.yield %[[EQ]]
// CHECK: return %[[FUSED]]
func.func @test_eq_comparison(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
  %0 = muse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.eq %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  return %1 : tensor<4x4xi1>
}

// Test scenario: Greater-than comparison (gt) after arithmetic operation
// CHECK-LABEL: func @test_gt_ge_comparison_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[MUL:.*]] = muse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[GT:.*]] = muse.gt %[[MUL]], %[[ARG2]]
// CHECK: muse.yield %[[GT]]
// CHECK: return %[[FUSED]]
func.func @test_gt_ge_comparison_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
  %0 = muse.mul %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.gt %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  return %1 : tensor<4x4xi1>
}

// Test scenario: Less-than comparison (lt) after arithmetic operation
// CHECK-LABEL: func @test_lt_le_comparison_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[SUB:.*]] = muse.sub %[[ARG2]], %[[ARG3]]
// CHECK: %[[LT:.*]] = muse.lt %[[SUB]], %[[ARG2]]
// CHECK: muse.yield %[[LT]]
// CHECK: return %[[FUSED]]
func.func @test_lt_le_comparison_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
  %0 = muse.sub %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.lt %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  return %1 : tensor<4x4xi1>
}

// Test scenario: Not-equal comparison (ne) after arithmetic operation
// CHECK-LABEL: func @test_ne_comparison_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[DIV:.*]] = muse.div %[[ARG2]], %[[ARG3]]
// CHECK: %[[NE:.*]] = muse.ne %[[DIV]], %[[ARG2]]
// CHECK: muse.yield %[[NE]]
// CHECK: return %[[FUSED]]
func.func @test_ne_comparison_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
  %0 = muse.div %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.ne %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  return %1 : tensor<4x4xi1>
}

// Test scenario: Multiple independent comparison operations in the same function
// Each comparison chain forms its own cluster
// CHECK-LABEL: func @test_multiple_comparisons
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED1:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = muse.add %[[ARG3]], %[[ARG4]]
// CHECK: %[[MUL:.*]] = muse.mul %[[ADD]], %[[ARG3]]
// CHECK: %[[GT:.*]] = muse.gt %[[MUL]], %[[ARG4]]
// CHECK: muse.yield %[[GT]]
// CHECK: %[[FUSED2:.*]] = muse.fused %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[SUB:.*]] = muse.sub %[[ARG3]], %[[ARG4]]
// CHECK: %[[LT:.*]] = muse.lt %[[SUB]], %[[ARG4]]
// CHECK: muse.yield %[[LT]]
// CHECK: return %[[FUSED1]], %[[FUSED2]]
func.func @test_multiple_comparisons(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> (tensor<4x4xi1>, tensor<4x4xi1>) {
  %0 = muse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.mul %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = muse.gt %1, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  %3 = muse.sub %arg1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %4 = muse.lt %3, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  return %2, %4 : tensor<4x4xi1>, tensor<4x4xi1>
}
}
