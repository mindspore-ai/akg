// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// Test logical operations clustering: logical_and, logical_or, logical_not, select

// Test logical_and and logical_or operations
// NOTE: These operations do NOT cluster because:
// 1. mfuse.logical_not/logical_and/logical_or are not in the DVM check function list
// 2. Default DVM check only supports float types (f32/f16/bf16), not boolean (i1) output
// 3. These operations output tensor<i1> which fails the IsFloatType() check in DvmSupportChecker
// CHECK-LABEL: func @test_logical_and_or
// CHECK-SAME: %arg0: tensor<4x4xi1>
// CHECK-SAME: %arg1: tensor<4x4xi1>
// CHECK: %[[AND:.*]] = mfuse.logical_and %arg0, %arg1
// CHECK-SAME: : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
// CHECK: %[[OR:.*]] = mfuse.logical_or %[[AND]], %arg0
// CHECK-SAME: : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
// CHECK: return %[[OR]]
func.func @test_logical_and_or(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xi1>) -> tensor<4x4xi1> {
  %0 = mfuse.logical_and %arg0, %arg1 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  %1 = mfuse.logical_or %0, %arg0 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  return %1 : tensor<4x4xi1>
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

// Test scenario: Combining comparison operations with logical_and and select
// Comparison results are combined with logical_and (not clustered) and used in select (clustered)
// CHECK-LABEL: func @test_logical_comparison_mix
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[GT:.*]] = mfuse.gt %arg0, %arg1
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK: %[[LT:.*]] = mfuse.lt %arg1, %arg2
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK: %[[AND:.*]] = mfuse.logical_and %[[GT]], %[[LT]]
// CHECK-SAME: : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg2, %[[AND]], %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xi1>, %[[ARG5:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[SELECT:.*]] = mfuse.select %[[ARG4]], %[[ADD]], %[[ARG5]]
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
