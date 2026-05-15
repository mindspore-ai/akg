// RUN: mfusion-opt %s --split | FileCheck %s

module {
// Test floor operation chain
// CHECK-LABEL: func @test_floor_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[DIV:.*]] = mfuse.div %[[ARG2]], %[[ARG3]]
// CHECK: %[[FLOOR:.*]] = mfuse.floor %[[DIV]]
// CHECK: mfuse.yield %[[FLOOR]]
// CHECK: return %[[FUSED]]
func.func @test_floor_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.div %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.floor %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test ceil operation chain
// CHECK-LABEL: func @test_ceil_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[CEIL:.*]] = mfuse.ceil %[[MUL]]
// CHECK: mfuse.yield %[[CEIL]]
// CHECK: return %[[FUSED]]
func.func @test_ceil_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.mul %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.ceil %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test trunc operation chain
// CHECK-LABEL: func @test_trunc_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[SUB:.*]] = mfuse.sub %[[ARG2]], %[[ARG3]]
// CHECK: %[[TRUNC:.*]] = mfuse.trunc %[[SUB]]
// CHECK: mfuse.yield %[[TRUNC]]
// CHECK: return %[[FUSED]]
func.func @test_trunc_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.sub %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.trunc %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test floor and ceil combination
// CHECK-LABEL: func @test_floor_ceil_combo
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[FLOOR:.*]] = mfuse.floor %[[ADD]]
// CHECK: %[[CEIL:.*]] = mfuse.ceil %[[FLOOR]]
// CHECK: mfuse.yield %[[CEIL]]
// CHECK: return %[[FUSED]]
func.func @test_floor_ceil_combo(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.floor %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.ceil %2 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %3 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test all rounding operations
// CHECK-LABEL: func @test_all_rounding_ops
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]]:3 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[DIV:.*]] = mfuse.div %[[ARG2]], %[[ARG3]]
// CHECK: %[[FLOOR:.*]] = mfuse.floor %[[DIV]]
// CHECK: %[[CEIL:.*]] = mfuse.ceil %[[DIV]]
// CHECK: %[[TRUNC:.*]] = mfuse.trunc %[[DIV]]
// CHECK: mfuse.yield %[[FLOOR]], %[[CEIL]], %[[TRUNC]]
// CHECK: return %[[FUSED]]#0, %[[FUSED]]#1, %[[FUSED]]#2
func.func @test_all_rounding_ops(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  %0:3 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.div %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.floor %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.ceil %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = mfuse.trunc %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2, %3, %4 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
  }
  return %0#0, %0#1, %0#2 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}
}
