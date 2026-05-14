// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
  // Test case 1: Reshape with single dimension 1 inserted (unsqueeze-like)
  // CHECK-LABEL: func @reshape_single_dim_one_inserted
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0
  // CHECK-SAME: {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x3xf32>) -> tensor<2x1x3xf32>
  // CHECK: ^bb0(%[[A:.*]]: tensor<2x3xf32>):
  // CHECK: %[[RS:.*]] = mfuse.reshape %[[A]]
  // CHECK: %[[ADD:.*]] = mfuse.add %[[RS]], %[[RS]]
  // CHECK: mfuse.yield %[[ADD]]
  // CHECK: return %[[FUSED]]
  func.func @reshape_single_dim_one_inserted(
      %arg0: tensor<2x3xf32>) -> tensor<2x1x3xf32> {
    %0 = mfuse.reshape %arg0 : (tensor<2x3xf32>) -> tensor<2x1x3xf32>
    %1 = mfuse.add %0, %0 : (tensor<2x1x3xf32>, tensor<2x1x3xf32>) -> tensor<2x1x3xf32>
    return %1 : tensor<2x1x3xf32>
  }

  // Test case 2: Reshape with multiple trailing dimensions 1 removed (squeeze-like)
  // CHECK-LABEL: func @reshape_multiple_ones_removed
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0
  // CHECK-SAME: {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<1x2x3x1x1xf32>) -> tensor<2x3xf32>
  // CHECK: ^bb0(%[[A:.*]]: tensor<1x2x3x1x1xf32>):
  // CHECK: %[[RS:.*]] = mfuse.reshape %[[A]]
  // CHECK: %[[ADD:.*]] = mfuse.add %[[RS]], %[[RS]]
  // CHECK: mfuse.yield %[[ADD]]
  // CHECK: return %[[FUSED]]
  func.func @reshape_multiple_ones_removed(
      %arg0: tensor<1x2x3x1x1xf32>) -> tensor<2x3xf32> {
    %0 = mfuse.reshape %arg0 : (tensor<1x2x3x1x1xf32>) -> tensor<2x3xf32>
    %1 = mfuse.add %0, %0 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }

  // Test case 3: Reshape with one dimension 1 removed (squeeze-like)
  // CHECK-LABEL: func @reshape_single_one_removed
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0
  // CHECK-SAME: {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x1x3xf32>) -> tensor<2x3xf32>
  // CHECK: ^bb0(%[[A:.*]]: tensor<2x1x3xf32>):
  // CHECK: %[[RS:.*]] = mfuse.reshape %[[A]]
  // CHECK: %[[ADD:.*]] = mfuse.add %[[RS]], %[[RS]]
  // CHECK: mfuse.yield %[[ADD]]
  // CHECK: return %[[FUSED]]
  func.func @reshape_single_one_removed(
      %arg0: tensor<2x1x3xf32>) -> tensor<2x3xf32> {
    %0 = mfuse.reshape %arg0 : (tensor<2x1x3xf32>) -> tensor<2x3xf32>
    %1 = mfuse.add %0, %0 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }

  // Test case 4: Reshape after mul with dimension 1 inserted
  // CHECK-LABEL: func @reshape_after_mul_with_one_inserted
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
  // CHECK-SAME: {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3x1xf32>
  // CHECK: ^bb0(%[[A:.*]]: tensor<2x3xf32>, %[[B:.*]]: tensor<2x3xf32>):
  // CHECK: %[[MM:.*]] = mfuse.mul %[[A]], %[[B]]
  // CHECK: %[[RS:.*]] = mfuse.reshape %[[MM]]
  // CHECK: %[[ADD:.*]] = mfuse.add %[[RS]], %[[RS]]
  // CHECK: mfuse.yield %[[ADD]]
  // CHECK: return %[[FUSED]]
  func.func @reshape_after_mul_with_one_inserted(
      %arg0: tensor<2x3xf32>,
      %arg1: tensor<2x3xf32>) -> tensor<2x3x1xf32> {
    %0 = mfuse.mul %arg0, %arg1 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %1 = mfuse.reshape %0 : (tensor<2x3xf32>) -> tensor<2x3x1xf32>
    %2 = mfuse.add %1, %1 : (tensor<2x3x1xf32>, tensor<2x3x1xf32>) -> tensor<2x3x1xf32>
    return %2 : tensor<2x3x1xf32>
  }
}
