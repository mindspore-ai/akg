// RUN: mfusion-opt %s --split | FileCheck %s

module {
// Test split with reshape operations
// CHECK-LABEL: func @test_split_with_reshape
// CHECK-SAME: %arg0: tensor<8xf32>
// CHECK: %[[FUSED1:.*]] = mfuse.fused %arg0
// CHECK: mfuse.add
// CHECK: mfuse.mul
// CHECK: mfuse.yield
// CHECK: %[[RESHAPE:.*]] = mfuse.reshape %[[FUSED1]]
// CHECK: %[[FUSED2:.*]]:2 = mfuse.fused %[[RESHAPE]]
// CHECK: mfuse.add
// CHECK: mfuse.add
// CHECK: mfuse.mul
// CHECK: mfuse.yield
// CHECK: return %[[FUSED2]]#0, %[[FUSED2]]#1
func.func @test_split_with_reshape(%arg0: tensor<8xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<8xf32>) -> tensor<8xf32> {
  ^bb0(%arg1: tensor<8xf32>):
    %3 = mfuse.add %arg1, %arg1 : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    %4 = mfuse.mul %3, %3 : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    mfuse.yield %4 : tensor<8xf32>
  }
  %1 = mfuse.reshape %0 : (tensor<8xf32>) -> tensor<2x4xf32>
  %2:2 = mfuse.fused %1 {fusion_type = "dvm"} : (tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  ^bb0(%arg1: tensor<2x4xf32>):
    %3 = mfuse.add %arg1, %arg1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %4 = mfuse.add %3, %3 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %5 = mfuse.mul %4, %4 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    mfuse.yield %3, %5 : tensor<2x4xf32>, tensor<2x4xf32>
  }
  return %2#0, %2#1 : tensor<2x4xf32>, tensor<2x4xf32>
}

// CHECK-LABEL: func @test_prune_passthrough_result_after_input_reshape_hoist
// CHECK-SAME: %arg0: tensor<8xf32>
// CHECK: %[[RESHAPE:.*]] = mfuse.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %[[RESHAPE]] {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<2x4xf32>) -> tensor<1x4xf32>
// CHECK: ^bb0(%[[IN:.*]]: tensor<2x4xf32>):
// CHECK: %[[SUM:.*]] = mfuse.reduce_sum %[[IN]] {dimensions = [0], keepdim = true} : (tensor<2x4xf32>) -> tensor<1x4xf32>
// CHECK: mfuse.yield %[[SUM]] : tensor<1x4xf32>
// CHECK: %[[ADD:.*]] = mfuse.add %[[RESHAPE]], %[[RESHAPE]] : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK: return %[[ADD]], %[[FUSED]] : tensor<2x4xf32>, tensor<1x4xf32>
func.func @test_prune_passthrough_result_after_input_reshape_hoist(%arg0: tensor<8xf32>)
    -> (tensor<2x4xf32>, tensor<1x4xf32>) {
  %0:2 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<8xf32>) -> (tensor<2x4xf32>, tensor<1x4xf32>) {
  ^bb0(%arg1: tensor<8xf32>):
    %1 = mfuse.reshape %arg1 : (tensor<8xf32>) -> tensor<2x4xf32>
    %2 = mfuse.reduce_sum %1 {dimensions = [0], keepdim = true} : (tensor<2x4xf32>) -> tensor<1x4xf32>
    mfuse.yield %1, %2 : tensor<2x4xf32>, tensor<1x4xf32>
  }
  %3 = mfuse.add %0#0, %0#0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %3, %0#1 : tensor<2x4xf32>, tensor<1x4xf32>
}

}
