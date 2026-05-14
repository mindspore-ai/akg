// RUN: mfusion-opt %s --split | FileCheck %s

module {
// Test maximum and minimum chain
// CHECK-LABEL: func @test_maximum_minimum_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>, %[[ARG5:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG3]], %[[ARG4]]
// CHECK: %[[MAX:.*]] = mfuse.maximum %[[ADD]], %[[ARG5]]
// CHECK: %[[MIN:.*]] = mfuse.minimum %[[MAX]], %[[ARG4]]
// CHECK: mfuse.yield %[[MIN]]
// CHECK: return %[[FUSED]]
func.func @test_maximum_minimum_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>, %arg5: tensor<4x4xf32>):
    %1 = mfuse.add %arg3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.maximum %1, %arg5 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.minimum %2, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %3 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test abs chain
// CHECK-LABEL: func @test_abs_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[SUB:.*]] = mfuse.sub %[[ARG2]], %[[ARG3]]
// CHECK: %[[ABS:.*]] = mfuse.abs %[[SUB]]
// CHECK: mfuse.yield %[[ABS]]
// CHECK: return %[[FUSED]]
func.func @test_abs_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.sub %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.abs %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test abs and maximum combo
// CHECK-LABEL: func @test_abs_maximum_combo
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>, %[[ARG5:.*]]: tensor<4x4xf32>):
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG3]], %[[ARG4]]
// CHECK: %[[ABS:.*]] = mfuse.abs %[[MUL]]
// CHECK: %[[MAX:.*]] = mfuse.maximum %[[ABS]], %[[ARG5]]
// CHECK: mfuse.yield %[[MAX]]
// CHECK: return %[[FUSED]]
func.func @test_abs_maximum_combo(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>, %arg5: tensor<4x4xf32>):
    %1 = mfuse.mul %arg3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.abs %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.maximum %2, %arg5 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %3 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test multiple maximum outputs
// CHECK-LABEL: func @test_multiple_maximum_outputs
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]]:2 = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>, %[[ARG5:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG3]], %[[ARG4]]
// CHECK: %[[MAX:.*]] = mfuse.maximum %[[ADD]], %[[ARG5]]
// CHECK: %[[MIN:.*]] = mfuse.minimum %[[ADD]], %[[ARG5]]
// CHECK: mfuse.yield %[[MAX]], %[[MIN]]
// CHECK: return %[[FUSED]]#0, %[[FUSED]]#1
func.func @test_multiple_maximum_outputs(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0:2 = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  ^bb0(%arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>, %arg5: tensor<4x4xf32>):
    %1 = mfuse.add %arg3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.maximum %1, %arg5 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.minimum %1, %arg5 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2, %3 : tensor<4x4xf32>, tensor<4x4xf32>
  }
  return %0#0, %0#1 : tensor<4x4xf32>, tensor<4x4xf32>
}

// Test abs and neg combo
// CHECK-LABEL: func @test_abs_neg_combo
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG1:.*]]: tensor<4x4xf32>):
// CHECK: %[[NEG:.*]] = mfuse.neg %[[ARG1]]
// CHECK: %[[ABS:.*]] = mfuse.abs %[[NEG]]
// CHECK: mfuse.yield %[[ABS]]
// CHECK: return %[[FUSED]]
func.func @test_abs_neg_combo(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg1: tensor<4x4xf32>):
    %1 = mfuse.neg %arg1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.abs %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}
}
