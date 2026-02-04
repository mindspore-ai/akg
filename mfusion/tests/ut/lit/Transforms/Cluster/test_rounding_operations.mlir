// RUN: mfusion-opt %s --muse-dvm-cluster | FileCheck %s

module {
// Test rounding operations: floor, ceil, trunc

// Test scenario: Floor operation after division
// CHECK-LABEL: func @test_floor_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[DIV:.*]] = muse.div %[[ARG2]], %[[ARG3]]
// CHECK: %[[FLOOR:.*]] = muse.floor %[[DIV]]
// CHECK: muse.yield %[[FLOOR]]
// CHECK: return %[[FUSED]]
func.func @test_floor_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.div %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.floor %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Ceil operation after multiplication
// CHECK-LABEL: func @test_ceil_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[MUL:.*]] = muse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[CEIL:.*]] = muse.ceil %[[MUL]]
// CHECK: muse.yield %[[CEIL]]
// CHECK: return %[[FUSED]]
func.func @test_ceil_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.mul %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.ceil %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Truncate operation after subtraction
// CHECK-LABEL: func @test_trunc_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[SUB:.*]] = muse.sub %[[ARG2]], %[[ARG3]]
// CHECK: %[[TRUNC:.*]] = muse.trunc %[[SUB]]
// CHECK: muse.yield %[[TRUNC]]
// CHECK: return %[[FUSED]]
func.func @test_trunc_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.sub %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.trunc %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Combining floor and ceil operations in sequence
// CHECK-LABEL: func @test_floor_ceil_combo
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = muse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[FLOOR:.*]] = muse.floor %[[ADD]]
// CHECK: %[[CEIL:.*]] = muse.ceil %[[FLOOR]]
// CHECK: muse.yield %[[CEIL]]
// CHECK: return %[[FUSED]]
func.func @test_floor_ceil_combo(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.floor %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = muse.ceil %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// Test scenario: Multiple rounding operations (floor, ceil, trunc) producing multiple outputs
// CHECK-LABEL: func @test_all_rounding_ops
// CHECK: %[[FUSED:.*]]:3 = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[DIV:.*]] = muse.div %[[ARG2]], %[[ARG3]]
// CHECK: %[[FLOOR:.*]] = muse.floor %[[DIV]]
// CHECK: %[[CEIL:.*]] = muse.ceil %[[DIV]]
// CHECK: %[[TRUNC:.*]] = muse.trunc %[[DIV]]
// CHECK: muse.yield %[[FLOOR]], %[[CEIL]], %[[TRUNC]]
// CHECK: return %[[FUSED]]#0, %[[FUSED]]#1, %[[FUSED]]#2
func.func @test_all_rounding_ops(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  %0 = muse.div %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.floor %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = muse.ceil %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %3 = muse.trunc %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1, %2, %3 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}
}
