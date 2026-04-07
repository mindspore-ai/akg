// RUN: mfusion-opt %s --split | FileCheck %s

module {
// CHECK-LABEL: func @test_exp_reduce_sum_mul_sub_split
// CHECK-SAME: %[[ARG0:.*]]: tensor<4x4xf32>, %[[ARG1:.*]]: tensor<4x4xf32>
// CHECK: %[[REDUCE_SUM:.*]] = mfuse.reduce_sum %[[ARG1]] {dimensions = [1], keepdim = true} : (tensor<4x4xf32>) -> tensor<4x1xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %[[ARG0]], %[[REDUCE_SUM]], %[[ARG1]] {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x1xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x1xf32>, %[[ARG4:.*]]: tensor<4x4xf32>):
// CHECK: %[[EXP:.*]] = mfuse.exp %[[ARG2]] : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: %[[MUL:.*]] = mfuse.mul %[[EXP]], %[[ARG3]] : (tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
// CHECK: %[[SUB:.*]] = mfuse.sub %[[MUL]], %[[ARG4]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: mfuse.yield %[[SUB]] : tensor<4x4xf32>
// CHECK: return %[[FUSED]]
func.func @test_exp_reduce_sum_mul_sub_split(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>):
    %1 = mfuse.exp %arg3 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.reduce_sum %arg4 {dimensions = [1], keepdim = true} : (tensor<4x4xf32>) -> tensor<4x1xf32>
    %3 = mfuse.mul %1, %2 : (tensor<4x4xf32>, tensor<4x1xf32>) -> tensor<4x4xf32>
    %4 = mfuse.sub %3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %4 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}
}
