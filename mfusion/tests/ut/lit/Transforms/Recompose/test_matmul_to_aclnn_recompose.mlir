// RUN: mfusion-opt %s -recompose | FileCheck %s

module {
  // mfuse.matmul 2D x 2D -> aclnn.mm
  // CHECK-LABEL: func @matmul_2d_to_mm
  func.func @matmul_2d_to_mm(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
    // CHECK-NOT: mfuse.matmul
    // CHECK: mfuse.aclnn.mm
  }

  // mfuse.matmul ND (batch, rank>=3) -> aclnn.batch_matmul
  // CHECK-LABEL: func @matmul_nd_to_aclnn
  func.func @matmul_nd_to_aclnn(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    return %0 : tensor<2x4x16xf32>
    // CHECK-NOT: mfuse.matmul
    // CHECK: mfuse.aclnn.batch_matmul
  }

  // mfuse.matmul_with_bias 2D -> aclnn.mm + aclnn.add
  // CHECK-LABEL: func @matmul_with_bias_2d
  func.func @matmul_with_bias_2d(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>) -> tensor<4x16xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
    // CHECK-NOT: mfuse.matmul_with_bias
    // CHECK: mfuse.aclnn.mm
    // CHECK: mfuse.aclnn.add
  }

  // mfuse.matmul_with_bias ND (rank>=3) -> aclnn.batch_matmul + aclnn.add
  // CHECK-LABEL: func @matmul_with_bias_nd
  func.func @matmul_with_bias_nd(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>, %arg2: tensor<16xf32>) -> tensor<2x4x16xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>, tensor<16xf32>) -> tensor<2x4x16xf32>
    return %0 : tensor<2x4x16xf32>
    // CHECK-NOT: mfuse.matmul_with_bias
    // CHECK: mfuse.aclnn.batch_matmul
    // CHECK: mfuse.aclnn.add
  }

  // mfuse.matmul 2D (no trans) -> aclnn.mm (same as first test, but with trans attributes)
  // CHECK-LABEL: func @mm_2d_to_aclnn
  func.func @mm_2d_to_aclnn(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
    // CHECK-NOT: mfuse.matmul
    // CHECK: mfuse.aclnn.mm
  }

  // mfuse.matmul 2D with trans -> permute + aclnn.mm
  // CHECK-LABEL: func @mm_2d_trans_to_aclnn
  func.func @mm_2d_trans_to_aclnn(%arg0: tensor<8x4xf32>, %arg1: tensor<16x8xf32>) -> tensor<4x16xf32> {
    %0 = mfuse.matmul %arg0, %arg1 {trans_x1 = true, trans_x2 = true} : (tensor<8x4xf32>, tensor<16x8xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
    // CHECK-NOT: mfuse.matmul
    // CHECK: mfuse.permute
    // CHECK: mfuse.permute
    // CHECK: mfuse.aclnn.mm
  }

  // mfuse.matmul_with_bias 2D -> aclnn.mm + aclnn.add (same as matmul_with_bias_2d test)
  // CHECK-LABEL: func @mm_with_bias_2d_to_aclnn
  func.func @mm_with_bias_2d_to_aclnn(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>) -> tensor<4x16xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
    // CHECK-NOT: mfuse.matmul_with_bias
    // CHECK: mfuse.aclnn.mm
    // CHECK: mfuse.aclnn.add
  }

  // mfuse.matmul_with_bias with shape mismatch: bias [16] needs reshape to [1, 16] for broadcast
  // CHECK-LABEL: func @matmul_with_bias_shape_mismatch_2d
  func.func @matmul_with_bias_shape_mismatch_2d(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>, %arg2: tensor<16xf32>) -> tensor<4x16xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
    // CHECK-NOT: mfuse.matmul_with_bias
    // CHECK: mfuse.aclnn.mm
    // CHECK: mfuse.reshape
    // CHECK-SAME: (tensor<16xf32>) -> tensor<1x16xf32>
    // CHECK: mfuse.aclnn.add
  }

  // mfuse.matmul_with_bias ND with shape mismatch: bias [16] needs reshape to [1, 1, 16] for broadcast
  // CHECK-LABEL: func @matmul_with_bias_shape_mismatch_nd
  func.func @matmul_with_bias_shape_mismatch_nd(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>, %arg2: tensor<16xf32>) -> tensor<2x4x16xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>, tensor<16xf32>) -> tensor<2x4x16xf32>
    return %0 : tensor<2x4x16xf32>
    // CHECK-NOT: mfuse.matmul_with_bias
    // CHECK: mfuse.aclnn.batch_matmul
    // CHECK: mfuse.reshape
    // CHECK-SAME: (tensor<16xf32>) -> tensor<1x1x16xf32>
    // CHECK: mfuse.aclnn.add
  }
}
