// RUN: mfusion-opt %s --mfuse-decompose-matmul-with-bias-for-dvm-cluster | FileCheck %s

module {
  // CHECK-LABEL: func.func @decompose_2d_f32
  // CHECK-SAME: %[[A:.*]]: tensor<4096x16xf32>
  // CHECK-SAME: %[[B:.*]]: tensor<16x16xf32>
  // CHECK-SAME: %[[BIAS:.*]]: tensor<16xf32>
  // CHECK-NOT: mfuse.matmul_with_bias
  // CHECK: %[[MM:.*]] = mfuse.matmul %[[A]], %[[B]]
  // CHECK-SAME: : (tensor<4096x16xf32>, tensor<16x16xf32>) -> tensor<4096x16xf32>
  // CHECK: %[[ADD:.*]] = mfuse.add %[[MM]], %[[BIAS]]
  // CHECK-SAME: : (tensor<4096x16xf32>, tensor<16xf32>) -> tensor<4096x16xf32>
  // CHECK: return %[[ADD]]
  func.func @decompose_2d_f32(
      %arg0: tensor<4096x16xf32>,
      %arg1: tensor<16x16xf32>,
      %bias: tensor<16xf32>) -> tensor<4096x16xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %bias
        : (tensor<4096x16xf32>, tensor<16x16xf32>, tensor<16xf32>) -> tensor<4096x16xf32>
    return %0 : tensor<4096x16xf32>
  }

  // CHECK-LABEL: func.func @preserve_transpose_attrs
  // CHECK-SAME: %[[A:.*]]: tensor<16x4096xf32>
  // CHECK-SAME: %[[B:.*]]: tensor<16x16xf32>
  // CHECK-SAME: %[[BIAS:.*]]: tensor<16xf32>
  // CHECK-NOT: mfuse.matmul_with_bias
  // CHECK: %[[MM:.*]] = mfuse.matmul %[[A]], %[[B]] {trans_x1 = true, trans_x2 = true}
  // CHECK-SAME: : (tensor<16x4096xf32>, tensor<16x16xf32>) -> tensor<4096x16xf32>
  // CHECK: %[[ADD:.*]] = mfuse.add %[[MM]], %[[BIAS]]
  // CHECK: return %[[ADD]]
  func.func @preserve_transpose_attrs(
      %arg0: tensor<16x4096xf32>,
      %arg1: tensor<16x16xf32>,
      %bias: tensor<16xf32>) -> tensor<4096x16xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %bias {trans_x1 = true, trans_x2 = true}
        : (tensor<16x4096xf32>, tensor<16x16xf32>, tensor<16xf32>) -> tensor<4096x16xf32>
    return %0 : tensor<4096x16xf32>
  }

  // CHECK-LABEL: func.func @decompose_rank3
  // CHECK-SAME: %[[A:.*]]: tensor<2x4x8xf32>
  // CHECK-SAME: %[[B:.*]]: tensor<2x8x16xf32>
  // CHECK-SAME: %[[BIAS:.*]]: tensor<16xf32>
  // CHECK-NOT: mfuse.matmul_with_bias
  // CHECK-NOT: mfuse.reshape
  // CHECK: %[[MM:.*]] = mfuse.matmul %[[A]], %[[B]]
  // CHECK-SAME: : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
  // CHECK: %[[ADD:.*]] = mfuse.add %[[MM]], %[[BIAS]]
  // CHECK-SAME: : (tensor<2x4x16xf32>, tensor<16xf32>) -> tensor<2x4x16xf32>
  // CHECK: return %[[ADD]]
  func.func @decompose_rank3(
      %arg0: tensor<2x4x8xf32>,
      %arg1: tensor<2x8x16xf32>,
      %bias: tensor<16xf32>) -> tensor<2x4x16xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %bias
        : (tensor<2x4x8xf32>, tensor<2x8x16xf32>, tensor<16xf32>) -> tensor<2x4x16xf32>
    return %0 : tensor<2x4x16xf32>
  }
}

