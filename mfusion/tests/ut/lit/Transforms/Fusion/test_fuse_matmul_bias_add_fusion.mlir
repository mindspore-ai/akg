// RUN: mfusion-opt %s --fuse-matmul-bias-add | FileCheck %s

module {
  // MatMul (2D) + Add(bias 1D) -> MatmulWithBias. [2,4] @ [4,8] -> [2,8], bias [8].
  // CHECK-LABEL: func @fuse_matmul_bias_add
  // CHECK-SAME: (%[[A:.*]]: tensor<2x4xf32>, %[[B:.*]]: tensor<4x8xf32>, %[[BIAS:.*]]: tensor<8xf32>)
  func.func @fuse_matmul_bias_add(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %bias: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.add %0, %bias : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    // After fusion: matmul and add replaced by matmul_with_bias
    // CHECK-NOT: mfuse.matmul
    // CHECK-NOT: mfuse.add
    // CHECK: %[[R:.*]] = mfuse.matmul_with_bias %[[A]], %[[B]], %[[BIAS]]
    // CHECK: return %[[R]]
    return %1 : tensor<2x8xf32>
  }

  // Add(bias, matmul) with bias on LHS: same fusion (commutative).
  // CHECK-LABEL: func @fuse_matmul_bias_add_bias_left
  func.func @fuse_matmul_bias_add_bias_left(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %bias: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.add %bias, %0 : (tensor<8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
    // CHECK-NOT: mfuse.add
    // CHECK: mfuse.matmul_with_bias
    return %1 : tensor<2x8xf32>
  }

  // BatchMatmul + Add(bias 1D) -> MatmulWithBias. [2,4,8] @ [2,8,16] -> [2,4,16], bias [16].
  // CHECK-LABEL: func @fuse_batch_matmul_bias_add
  func.func @fuse_batch_matmul_bias_add(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>, %bias: tensor<16xf32>) -> tensor<2x4x16xf32> {
    %0 = mfuse.batch_matmul %arg0, %arg1 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    %1 = mfuse.add %0, %bias : (tensor<2x4x16xf32>, tensor<16xf32>) -> tensor<2x4x16xf32>
    // CHECK-NOT: mfuse.batch_matmul
    // CHECK-NOT: mfuse.add
    // CHECK: mfuse.matmul_with_bias
    return %1 : tensor<2x4x16xf32>
  }

  // No fusion: bias shape [4] does not match matmul output last dim 8.
  // CHECK-LABEL: func @no_fusion_bias_wrong_size
  func.func @no_fusion_bias_wrong_size(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %bias: tensor<4xf32>) -> tensor<2x8xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.add %0, %bias : (tensor<2x8xf32>, tensor<4xf32>) -> tensor<2x8xf32>
    // CHECK: mfuse.matmul
    // CHECK: mfuse.add
    // CHECK-NOT: mfuse.matmul_with_bias
    return %1 : tensor<2x8xf32>
  }

  // No fusion: bias is 2D [8,1], pass requires 1D (one of add inputs must have rank 1).
  // CHECK-LABEL: func @no_fusion_bias_not_1d
  func.func @no_fusion_bias_not_1d(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %bias: tensor<8x1xf32>) -> tensor<2x8xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.add %0, %bias : (tensor<2x8xf32>, tensor<8x1xf32>) -> tensor<2x8xf32>
    // CHECK: mfuse.matmul
    // CHECK: mfuse.add
    // CHECK-NOT: mfuse.matmul_with_bias
    return %1 : tensor<2x8xf32>
  }
}
