// RUN: mfusion-opt %s --muse-dvm-cluster | FileCheck %s

module {
// Test extreme value and abs operations: maximum, minimum, abs

// Test scenario: Maximum and minimum operations in sequence
// CHECK-LABEL: func @test_maximum_minimum_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = muse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MAX:.*]] = muse.maximum %[[ADD]], %[[ARG4]]
// CHECK: %[[MIN:.*]] = muse.minimum %[[MAX]], %[[ARG3]]
// CHECK: muse.yield %[[MIN]]
// CHECK: return %[[FUSED]]
func.func @test_maximum_minimum_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.maximum %0, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = muse.minimum %1, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// Test scenario: Absolute value operation after subtraction
// CHECK-LABEL: func @test_abs_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[SUB:.*]] = muse.sub %[[ARG2]], %[[ARG3]]
// CHECK: %[[ABS:.*]] = muse.abs %[[SUB]]
// CHECK: muse.yield %[[ABS]]
// CHECK: return %[[FUSED]]
func.func @test_abs_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.sub %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.abs %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Combining abs, maximum operations
// CHECK-LABEL: func @test_abs_maximum_combo
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[MUL:.*]] = muse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[ABS:.*]] = muse.abs %[[MUL]]
// CHECK: %[[MAX:.*]] = muse.maximum %[[ABS]], %[[ARG4]]
// CHECK: muse.yield %[[MAX]]
// CHECK: return %[[FUSED]]
func.func @test_abs_maximum_combo(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.mul %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.abs %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = muse.maximum %1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// Test scenario: Multiple outputs from extreme value operations (max and min from same input)
// CHECK-LABEL: func @test_multiple_maximum_outputs
// CHECK: %[[FUSED:.*]]:2 = muse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = muse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MAX1:.*]] = muse.maximum %[[ADD]], %[[ARG4]]
// CHECK: %[[MIN1:.*]] = muse.minimum %[[ADD]], %[[ARG4]]
// CHECK: muse.yield %[[MAX1]], %[[MIN1]]
// CHECK: return %[[FUSED]]#0, %[[FUSED]]#1
func.func @test_multiple_maximum_outputs(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0 = muse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.maximum %0, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = muse.minimum %0, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1, %2 : tensor<4x4xf32>, tensor<4x4xf32>
}

// Test scenario: Combining negation and absolute value operations
// CHECK-LABEL: func @test_abs_neg_combo
// CHECK: %[[FUSED:.*]] = muse.fused %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>):
// CHECK: %[[NEG:.*]] = muse.neg %[[ARG2]]
// CHECK: %[[ABS:.*]] = muse.abs %[[NEG]]
// CHECK: muse.yield %[[ABS]]
// CHECK: return %[[FUSED]]
func.func @test_abs_neg_combo(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.neg %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.abs %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
}
