// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// Test rounding operations: floor, ceil, trunc

// Test scenario: Floor operation after division
// CHECK-LABEL: func @test_floor_chain
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[DIV:.*]] = mfuse.div %[[ARG2]], %[[ARG3]]
// CHECK: %[[FLOOR:.*]] = mfuse.floor %[[DIV]]
// CHECK: mfuse.yield %[[FLOOR]]
// CHECK: return %[[FUSED]]
func.func @test_floor_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.div %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.floor %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Ceil operation after multiplication
// CHECK-LABEL: func @test_ceil_chain
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[CEIL:.*]] = mfuse.ceil %[[MUL]]
// CHECK: mfuse.yield %[[CEIL]]
// CHECK: return %[[FUSED]]
func.func @test_ceil_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.mul %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.ceil %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Truncate operation after subtraction
// CHECK-LABEL: func @test_trunc_chain
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[SUB:.*]] = mfuse.sub %[[ARG2]], %[[ARG3]]
// CHECK: %[[TRUNC:.*]] = mfuse.trunc %[[SUB]]
// CHECK: mfuse.yield %[[TRUNC]]
// CHECK: return %[[FUSED]]
func.func @test_trunc_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.sub %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.trunc %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Combining floor and ceil operations in sequence
// CHECK-LABEL: func @test_floor_ceil_combo
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[FLOOR:.*]] = mfuse.floor %[[ADD]]
// CHECK: %[[CEIL:.*]] = mfuse.ceil %[[FLOOR]]
// CHECK: mfuse.yield %[[CEIL]]
// CHECK: return %[[FUSED]]
func.func @test_floor_ceil_combo(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.floor %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mfuse.ceil %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// Test scenario: Multiple rounding operations (floor, ceil, trunc) producing multiple outputs
// CHECK-LABEL: func @test_all_rounding_ops
// CHECK: %[[FUSED:.*]]:3 = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[DIV:.*]] = mfuse.div %[[ARG2]], %[[ARG3]]
// CHECK: %[[FLOOR:.*]] = mfuse.floor %[[DIV]]
// CHECK: %[[CEIL:.*]] = mfuse.ceil %[[DIV]]
// CHECK: %[[TRUNC:.*]] = mfuse.trunc %[[DIV]]
// CHECK: mfuse.yield %[[FLOOR]], %[[CEIL]], %[[TRUNC]]
// CHECK: return %[[FUSED]]#0, %[[FUSED]]#1, %[[FUSED]]#2
func.func @test_all_rounding_ops(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  %0 = mfuse.div %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.floor %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mfuse.ceil %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %3 = mfuse.trunc %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1, %2, %3 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}
}
