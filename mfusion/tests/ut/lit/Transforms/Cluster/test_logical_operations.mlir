// RUN: mfusion-opt %s --muse-dvm-cluster | FileCheck %s

module {
// Test logical operations clustering: logical_and, logical_or, logical_not, select

// Test logical_and and logical_or operations
// NOTE: These operations do NOT cluster because:
// 1. muse.logical_not/logical_and/logical_or are not in the DVM check function list
// 2. Default DVM check only supports float types (f32/f16/bf16), not boolean (i1) output
// 3. These operations output tensor<i1> which fails the IsFloatType() check in DvmSupportChecker
// CHECK-LABEL: func @test_logical_and_or
// CHECK-SAME: %arg0: tensor<4x4xi1>
// CHECK-SAME: %arg1: tensor<4x4xi1>
// CHECK: %[[AND:.*]] = muse.logical_and %arg0, %arg1
// CHECK-SAME: : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
// CHECK: %[[OR:.*]] = muse.logical_or %[[AND]], %arg0
// CHECK-SAME: : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
// CHECK: return %[[OR]]
func.func @test_logical_and_or(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xi1>) -> tensor<4x4xi1> {
  %0 = muse.logical_and %arg0, %arg1 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  %1 = muse.logical_or %0, %arg0 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  return %1 : tensor<4x4xi1>
}

// Test scenario: Select operation with comparison result as condition
// Select can cluster with arithmetic operations when the condition is a tensor<i1>
// CHECK-LABEL: func @test_select_operation
// CHECK-SAME: %arg0: tensor<4x4xi1>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = muse.fused %arg1, %arg2, %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xi1>):
// CHECK: %[[MUL:.*]] = muse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[SELECT:.*]] = muse.select %[[ARG4]], %[[MUL]], %[[ARG2]]
// CHECK: muse.yield %[[SELECT]]
// CHECK: return %[[FUSED]]
func.func @test_select_operation(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.mul %arg1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.select %arg0, %0, %arg1 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Select operation with comparison-generated condition
// Comparison (gt) generates a boolean tensor that is used as select condition
// CHECK-LABEL: func @test_complex_logical_select
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[GT:.*]] = muse.gt %[[ARG2]], %[[ARG3]]
// CHECK: %[[ADD:.*]] = muse.add %[[ARG2]], %[[ARG4]]
// CHECK: %[[MUL:.*]] = muse.mul %[[ARG3]], %[[ARG4]]
// CHECK: %[[SELECT:.*]] = muse.select %[[GT]], %[[ADD]], %[[MUL]]
// CHECK: muse.yield %[[SELECT]]
// CHECK: return %[[FUSED]]
func.func @test_complex_logical_select(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.gt %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  %1 = muse.add %arg0, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = muse.mul %arg1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %3 = muse.select %0, %1, %2 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %3 : tensor<4x4xf32>
}

// Test scenario: Combining comparison operations with logical_and and select
// Comparison results are combined with logical_and (not clustered) and used in select (clustered)
// CHECK-LABEL: func @test_logical_comparison_mix
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[GT:.*]] = muse.gt %arg0, %arg1
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK: %[[LT:.*]] = muse.lt %arg1, %arg2
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK: %[[AND:.*]] = muse.logical_and %[[GT]], %[[LT]]
// CHECK-SAME: : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg2, %[[AND]], %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xi1>, %[[ARG5:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = muse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[SELECT:.*]] = muse.select %[[ARG4]], %[[ADD]], %[[ARG5]]
// CHECK: muse.yield %[[SELECT]]
// CHECK: return %[[FUSED]]
func.func @test_logical_comparison_mix(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.gt %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  %1 = muse.lt %arg1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  %2 = muse.logical_and %0, %1 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  %3 = muse.add %arg0, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %4 = muse.select %2, %3, %arg1 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %4 : tensor<4x4xf32>
}
}
