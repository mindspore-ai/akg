// RUN: mfusion-opt %s --fuse-batchmatmul-to-mul | FileCheck %s

module {
  // MatMul (3,1) @ (1,4) with no transpose: no reshape, direct mul.
  // CHECK-LABEL: func @matmul_k1_no_trans
  func.func @matmul_k1_no_trans(%arg0: tensor<3x1xf32>, %arg1: tensor<1x4xf32>) -> tensor<3x4xf32> {
    %0 = mfuse.matmul %arg0, %arg1 {trans_x1 = false, trans_x2 = false} : (tensor<3x1xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
    // CHECK-NOT: mfuse.reshape
    // CHECK: mfuse.mul {{.*}} : (tensor<3x1xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }

  // MatMul with trans: inputs (1,3) and (4,1); pass inserts reshape then mul.
  // CHECK-LABEL: func @matmul_k1_with_trans
  func.func @matmul_k1_with_trans(%arg0: tensor<1x3xf32>, %arg1: tensor<4x1xf32>) -> tensor<3x4xf32> {
    %0 = mfuse.matmul %arg0, %arg1 {trans_x1 = true, trans_x2 = true} : (tensor<1x3xf32>, tensor<4x1xf32>) -> tensor<3x4xf32>
    // CHECK: mfuse.reshape
    // CHECK: mfuse.reshape
    // CHECK: mfuse.mul
    return %0 : tensor<3x4xf32>
  }

  // BatchMatMul with transpose: (2,1,3) @ (2,4,1), trans_a/trans_b true -> reshape then mul.
  // CHECK-LABEL: func @batch_matmul_k1_with_trans
  func.func @batch_matmul_k1_with_trans(%arg0: tensor<2x1x3xf32>, %arg1: tensor<2x4x1xf32>) -> tensor<2x3x4xf32> {
    %0 = mfuse.batch_matmul %arg0, %arg1 {transpose_a = true, transpose_b = true} : (tensor<2x1x3xf32>, tensor<2x4x1xf32>) -> tensor<2x3x4xf32>
    // CHECK: mfuse.reshape
    // CHECK: mfuse.reshape
    // CHECK: mfuse.mul
    return %0 : tensor<2x3x4xf32>
  }

  // BatchMatMul (2,3,1) @ (2,1,4) no transpose: no reshape.
  // CHECK-LABEL: func @batch_matmul_k1_no_trans
  func.func @batch_matmul_k1_no_trans(%arg0: tensor<2x3x1xf32>, %arg1: tensor<2x1x4xf32>) -> tensor<2x3x4xf32> {
    %0 = mfuse.batch_matmul %arg0, %arg1 {transpose_a = false, transpose_b = false} : (tensor<2x3x1xf32>, tensor<2x1x4xf32>) -> tensor<2x3x4xf32>
    // CHECK-NOT: mfuse.reshape
    // CHECK: mfuse.mul {{.*}} : (tensor<2x3x1xf32>, tensor<2x1x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }
}
