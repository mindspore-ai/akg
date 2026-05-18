// RUN: mfusion-opt %s --split | FileCheck %s

module {
// Test multi-output fusion splitting
// CHECK-LABEL: func @test_multi_output_split
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]]:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG2]]
// CHECK: %[[SUB:.*]] = mfuse.sub %[[ADD]], %[[ARG3]]
// CHECK: mfuse.yield %[[MUL]], %[[SUB]]
// CHECK: return %[[FUSED]]#0, %[[FUSED]]#1
func.func @test_multi_output_split(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.mul %1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.sub %1, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2, %3 : tensor<4x4xf32>, tensor<4x4xf32>
  }
  return %0#0, %0#1 : tensor<4x4xf32>, tensor<4x4xf32>
}
}
