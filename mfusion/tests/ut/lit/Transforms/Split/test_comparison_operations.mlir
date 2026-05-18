// RUN: mfusion-opt %s --split | FileCheck %s

module {
// Test eq comparison
// CHECK-LABEL: func @test_eq_comparison
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[EQ:.*]] = mfuse.eq %[[ADD]], %[[ARG2]]
// CHECK: mfuse.yield %[[EQ]]
// CHECK: return %[[FUSED]]
func.func @test_eq_comparison(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.eq %1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    mfuse.yield %2 : tensor<4x4xi1>
  }
  return %0 : tensor<4x4xi1>
}

// Test gt comparison chain
// CHECK-LABEL: func @test_gt_ge_comparison_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[GT:.*]] = mfuse.gt %[[MUL]], %[[ARG2]]
// CHECK: mfuse.yield %[[GT]]
// CHECK: return %[[FUSED]]
func.func @test_gt_ge_comparison_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.mul %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.gt %1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    mfuse.yield %2 : tensor<4x4xi1>
  }
  return %0 : tensor<4x4xi1>
}

// Test lt comparison chain
// CHECK-LABEL: func @test_lt_le_comparison_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[SUB:.*]] = mfuse.sub %[[ARG2]], %[[ARG3]]
// CHECK: %[[LT:.*]] = mfuse.lt %[[SUB]], %[[ARG2]]
// CHECK: mfuse.yield %[[LT]]
// CHECK: return %[[FUSED]]
func.func @test_lt_le_comparison_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.sub %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.lt %1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    mfuse.yield %2 : tensor<4x4xi1>
  }
  return %0 : tensor<4x4xi1>
}

// Test ne comparison chain
// CHECK-LABEL: func @test_ne_comparison_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[DIV:.*]] = mfuse.div %[[ARG2]], %[[ARG3]]
// CHECK: %[[NE:.*]] = mfuse.ne %[[DIV]], %[[ARG2]]
// CHECK: mfuse.yield %[[NE]]
// CHECK: return %[[FUSED]]
func.func @test_ne_comparison_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.div %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.ne %1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    mfuse.yield %2 : tensor<4x4xi1>
  }
  return %0 : tensor<4x4xi1>
}

// Test multiple comparisons
// CHECK-LABEL: func @test_multiple_comparisons
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED1:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG3]], %[[ARG4]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG3]]
// CHECK: %[[GT:.*]] = mfuse.gt %[[MUL]], %[[ARG4]]
// CHECK: mfuse.yield %[[GT]]
// CHECK: %[[FUSED2:.*]] = mfuse.fused %arg1, %arg2 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[SUB:.*]] = mfuse.sub %[[ARG3]], %[[ARG4]]
// CHECK: %[[LT:.*]] = mfuse.lt %[[SUB]], %[[ARG4]]
// CHECK: mfuse.yield %[[LT]]
// CHECK: return %[[FUSED1]], %[[FUSED2]]
func.func @test_multiple_comparisons(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> (tensor<4x4xi1>, tensor<4x4xi1>) {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1> {
  ^bb0(%arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>):
    %2 = mfuse.add %arg3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.mul %2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = mfuse.gt %3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    mfuse.yield %4 : tensor<4x4xi1>
  }
  %1 = mfuse.fused %arg1, %arg2 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1> {
  ^bb0(%arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>):
    %2 = mfuse.sub %arg3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.lt %2, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    mfuse.yield %3 : tensor<4x4xi1>
  }
  return %0, %1 : tensor<4x4xi1>, tensor<4x4xi1>
}
}
