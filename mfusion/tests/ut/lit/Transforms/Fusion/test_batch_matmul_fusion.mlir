// RUN: mfusion-opt %s --fuse-batch-matmul --canonicalize | FileCheck %s

module {
  // Mode 1: permute (swap last two dims) + matmul -> matmul with trans flag; permute eliminated.
  // CHECK-LABEL: func @transpose_elimination_matmul_one_permute
  func.func @transpose_elimination_matmul_one_permute(%arg0: tensor<4x2xf32>, %arg1: tensor<2x8xf32>) -> tensor<4x8xf32> {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<4x2xf32>) -> tensor<2x4xf32>
    %1 = mfuse.matmul %0, %arg1 : (tensor<2x4xf32>, tensor<2x8xf32>) -> tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
    // After pass: use permute input and set trans_x1=true (trans_x2=false omitted when default).
    // CHECK-NOT: mfuse.permute
    // CHECK: mfuse.matmul {{.*}} {trans_x1 = true}
  }

  // Mode 1: both inputs from permute (swap last two) -> matmul with both trans set.
  // CHECK-LABEL: func @transpose_elimination_matmul_both_permute
  func.func @transpose_elimination_matmul_both_permute(%arg0: tensor<4x2xf32>, %arg1: tensor<8x2xf32>) -> tensor<4x8xf32> {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<4x2xf32>) -> tensor<2x4xf32>
    %1 = mfuse.permute %arg1, [1, 0] : (tensor<8x2xf32>) -> tensor<2x8xf32>
    %2 = mfuse.matmul %0, %1 : (tensor<2x4xf32>, tensor<2x8xf32>) -> tensor<4x8xf32>
    return %2 : tensor<4x8xf32>
    // CHECK: mfuse.matmul {{.*}} {trans_x1 = true, trans_x2 = true}
  }

  // Mode 2: BatchMatMul with both inputs 2D -> MatMul.
  // CHECK-LABEL: func @batch_matmul_2d_to_matmul
  func.func @batch_matmul_2d_to_matmul(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %0 = mfuse.batch_matmul %arg0, %arg1 : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
    // After pass: batch_matmul 2D -> matmul
    // CHECK-NOT: mfuse.batch_matmul
    // CHECK: mfuse.matmul %arg0, %arg1
  }
}
