// RUN: mfusion-opt %s --split | FileCheck %s

module {
// CHECK-LABEL: func @test_split_with_full
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG1:.*]]: tensor<4x4xf32>):
// CHECK: %[[FULL:.*]] = mfuse.full 1.000000e+00 : tensor<4x4xf32>
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG1]], %[[FULL]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: mfuse.yield %[[ADD]] : tensor<4x4xf32>
// CHECK: return %[[FUSED]] : tensor<4x4xf32>
func.func @test_split_with_full(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg1: tensor<4x4xf32>):
    %1 = mfuse.full 1.000000e+00 : tensor<4x4xf32>
    %2 = mfuse.add %arg1, %1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}
}
