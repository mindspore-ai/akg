// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// Test logical operations clustering: logical_and, logical_or, logical_not, select

// Test logical_and and logical_or operations.
// CHECK-LABEL: func @test_logical_and_or
// CHECK-SAME: %arg0: tensor<4x4xi1>
// CHECK-SAME: %arg1: tensor<4x4xi1>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xi1>, %[[ARG3:.*]]: tensor<4x4xi1>):
// CHECK: %[[AND:.*]] = mfuse.logical_and %[[ARG2]], %[[ARG3]]
// CHECK: %[[OR:.*]] = mfuse.logical_or %[[AND]], %[[ARG2]]
// CHECK: mfuse.yield %[[OR]]
// CHECK: return %[[FUSED]]
func.func @test_logical_and_or(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xi1>) -> tensor<4x4xi1> {
  %0 = mfuse.logical_and %arg0, %arg1 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  %1 = mfuse.logical_or %0, %arg0 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  return %1 : tensor<4x4xi1>
}

// Test logical_not clustering with bool input/output.
// CHECK-LABEL: func @test_logical_not_and
// CHECK-SAME: %arg0: tensor<4x4xi1>
// CHECK-SAME: %arg1: tensor<4x4xi1>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xi1>, %[[ARG3:.*]]: tensor<4x4xi1>):
// CHECK: %[[NOT:.*]] = mfuse.logical_not %[[ARG2]]
// CHECK: %[[AND:.*]] = mfuse.logical_and %[[NOT]], %[[ARG3]]
// CHECK: mfuse.yield %[[AND]]
// CHECK: return %[[FUSED]]
func.func @test_logical_not_and(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xi1>) -> tensor<4x4xi1> {
  %0 = mfuse.logical_not %arg0 : (tensor<4x4xi1>) -> tensor<4x4xi1>
  %1 = mfuse.logical_and %0, %arg1 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  return %1 : tensor<4x4xi1>
}

// Test scenario: eq followed by logical_not clusters with a non-finite
// rank-0 f64 scalar constant.
// CHECK-LABEL: func @test_eq_logical_not_with_non_finite_scalar
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x12x512x512xf32>) -> tensor<4x12x512x512xi1>
// CHECK: ^bb0(%[[ARG1:.*]]: tensor<4x12x512x512xf32>):
// CHECK: %[[CST:.*]] = mfuse.constant dense<0xFFF0000000000000> : tensor<f64, {is_scalar = ""}>
// CHECK: %[[EQ:.*]] = mfuse.eq %[[ARG1]], %[[CST]]
// CHECK: %[[NOT:.*]] = mfuse.logical_not %[[EQ]]
// CHECK: mfuse.yield %[[NOT]]
// CHECK: return %[[FUSED]]
func.func @test_eq_logical_not_with_non_finite_scalar(
    %arg0: tensor<4x12x512x512xf32>) -> tensor<4x12x512x512xi1> {
  %0 = mfuse.constant dense<0xFFF0000000000000> : tensor<f64, {is_scalar = ""}>
  %1 = mfuse.eq %arg0, %0 : (tensor<4x12x512x512xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x12x512x512xi1>
  %2 = mfuse.logical_not %1 : (tensor<4x12x512x512xi1>) -> tensor<4x12x512x512xi1>
  return %2 : tensor<4x12x512x512xi1>
}

// Non-bool logical ops should still be rejected by DVM cluster.
// CHECK-LABEL: func @test_logical_and_non_bool_not_clustered
// CHECK-NOT: mfuse.fused
// CHECK: %[[AND:.*]] = mfuse.logical_and %arg0, %arg1
// CHECK: %[[OR:.*]] = mfuse.logical_or %[[AND]], %arg0
// CHECK: return %[[OR]]
func.func @test_logical_and_non_bool_not_clustered(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.logical_and %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.logical_or %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Select operation with comparison result as condition
// Select can cluster with arithmetic operations when the condition is a tensor<i1>
// CHECK-LABEL: func @test_select_operation
// CHECK-SAME: %arg0: tensor<4x4xi1>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg1, %arg2, %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xi1>):
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[SELECT:.*]] = mfuse.select %[[ARG4]], %[[MUL]], %[[ARG2]]
// CHECK: mfuse.yield %[[SELECT]]
// CHECK: return %[[FUSED]]
func.func @test_select_operation(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.mul %arg1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.select %arg0, %0, %arg1 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Select operation with comparison-generated condition
// Comparison (gt) generates a boolean tensor that is used as select condition
// CHECK-LABEL: func @test_complex_logical_select
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[GT:.*]] = mfuse.gt %[[ARG2]], %[[ARG3]]
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG4]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG3]], %[[ARG4]]
// CHECK: %[[SELECT:.*]] = mfuse.select %[[GT]], %[[ADD]], %[[MUL]]
// CHECK: mfuse.yield %[[SELECT]]
// CHECK: return %[[FUSED]]
func.func @test_complex_logical_select(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.gt %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  %1 = mfuse.add %arg0, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mfuse.mul %arg1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %3 = mfuse.select %0, %1, %2 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %3 : tensor<4x4xf32>
}

// Test scenario: Combining comparison operations with logical_and and select.
// CHECK-LABEL: func @test_logical_comparison_mix
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>, %[[ARG5:.*]]: tensor<4x4xf32>):
// CHECK: %[[GT:.*]] = mfuse.gt %[[ARG3]], %[[ARG4]]
// CHECK: %[[LT:.*]] = mfuse.lt %[[ARG4]], %[[ARG5]]
// CHECK: %[[AND:.*]] = mfuse.logical_and %[[GT]], %[[LT]]
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG3]], %[[ARG5]]
// CHECK: %[[SELECT:.*]] = mfuse.select %[[AND]], %[[ADD]], %[[ARG4]]
// CHECK: mfuse.yield %[[SELECT]]
// CHECK: return %[[FUSED]]
func.func @test_logical_comparison_mix(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.gt %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  %1 = mfuse.lt %arg1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  %2 = mfuse.logical_and %0, %1 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  %3 = mfuse.add %arg0, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %4 = mfuse.select %2, %3, %arg1 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %4 : tensor<4x4xf32>
}
}
