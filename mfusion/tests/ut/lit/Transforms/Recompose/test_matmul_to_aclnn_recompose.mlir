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

  // mfuse.matmul 2D (no trans) -> aclnn.mm (same as first test, but with trans attributes)
  // CHECK-LABEL: func @mm_2d_to_aclnn
  func.func @mm_2d_to_aclnn(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
    // CHECK-NOT: mfuse.matmul
    // CHECK: mfuse.aclnn.mm
  }

  // mfuse.matmul 2D with trans -> aclnn.mm (transpose on op attrs; no permute)
  // CHECK-LABEL: func @mm_2d_trans_to_aclnn
  func.func @mm_2d_trans_to_aclnn(%arg0: tensor<8x4xf32>, %arg1: tensor<16x8xf32>) -> tensor<4x16xf32> {
    %0 = mfuse.matmul %arg0, %arg1 {trans_x1 = true, trans_x2 = true} : (tensor<8x4xf32>, tensor<16x8xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
    // CHECK-NOT: mfuse.matmul
    // CHECK-NOT: mfuse.permute
    // CHECK: mfuse.aclnn.mm{{.*}}trans_x1 = true{{.*}}trans_x2 = true
  }
}
