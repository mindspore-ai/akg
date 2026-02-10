// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// Test extreme value and abs operations: maximum, minimum, abs

// Test scenario: Maximum and minimum operations in sequence
// CHECK-LABEL: func @test_maximum_minimum_chain
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MAX:.*]] = mfuse.maximum %[[ADD]], %[[ARG4]]
// CHECK: %[[MIN:.*]] = mfuse.minimum %[[MAX]], %[[ARG3]]
// CHECK: mfuse.yield %[[MIN]]
// CHECK: return %[[FUSED]]
func.func @test_maximum_minimum_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.maximum %0, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mfuse.minimum %1, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// Test scenario: Absolute value operation after subtraction
// CHECK-LABEL: func @test_abs_chain
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[SUB:.*]] = mfuse.sub %[[ARG2]], %[[ARG3]]
// CHECK: %[[ABS:.*]] = mfuse.abs %[[SUB]]
// CHECK: mfuse.yield %[[ABS]]
// CHECK: return %[[FUSED]]
func.func @test_abs_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.sub %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.abs %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Combining abs, maximum operations
// CHECK-LABEL: func @test_abs_maximum_combo
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[ABS:.*]] = mfuse.abs %[[MUL]]
// CHECK: %[[MAX:.*]] = mfuse.maximum %[[ABS]], %[[ARG4]]
// CHECK: mfuse.yield %[[MAX]]
// CHECK: return %[[FUSED]]
func.func @test_abs_maximum_combo(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.mul %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.abs %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mfuse.maximum %1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// Test scenario: Multiple outputs from extreme value operations (max and min from same input)
// CHECK-LABEL: func @test_multiple_maximum_outputs
// CHECK: %[[FUSED:.*]]:2 = mfuse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MAX1:.*]] = mfuse.maximum %[[ADD]], %[[ARG4]]
// CHECK: %[[MIN1:.*]] = mfuse.minimum %[[ADD]], %[[ARG4]]
// CHECK: mfuse.yield %[[MAX1]], %[[MIN1]]
// CHECK: return %[[FUSED]]#0, %[[FUSED]]#1
func.func @test_multiple_maximum_outputs(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.maximum %0, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mfuse.minimum %0, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1, %2 : tensor<4x4xf32>, tensor<4x4xf32>
}

// Test scenario: Combining negation and absolute value operations
// CHECK-LABEL: func @test_abs_neg_combo
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>):
// CHECK: %[[NEG:.*]] = mfuse.neg %[[ARG2]]
// CHECK: %[[ABS:.*]] = mfuse.abs %[[NEG]]
// CHECK: mfuse.yield %[[ABS]]
// CHECK: return %[[FUSED]]
func.func @test_abs_neg_combo(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.neg %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.abs %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
}
