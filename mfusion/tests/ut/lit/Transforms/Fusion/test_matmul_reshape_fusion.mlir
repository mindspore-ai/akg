// RUN: mfusion-opt %s --fuse-matmul-reshape | FileCheck %s

module {
  // MatMul with N=1: second input [K, 1]. Pass adds reshape for second input (same shape).
  // CHECK-LABEL: func @matmul_n1_reshape_second_input
  func.func @matmul_n1_reshape_second_input(%arg0: tensor<2x4xf32>, %arg1: tensor<4x1xf32>) -> tensor<2x1xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x1xf32>) -> tensor<2x1xf32>
    // CHECK: mfuse.reshape
    // CHECK: mfuse.matmul
    return %0 : tensor<2x1xf32>
  }

  // MatmulWithBias with N=1: same, add reshape for second input.
  // CHECK-LABEL: func @matmul_with_bias_n1_reshape
  func.func @matmul_with_bias_n1_reshape(%arg0: tensor<2x4xf32>, %arg1: tensor<4x1xf32>, %arg2: tensor<1xf32>) -> tensor<2x1xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<2x4xf32>, tensor<4x1xf32>, tensor<1xf32>) -> tensor<2x1xf32>
    // CHECK: mfuse.reshape
    // CHECK: mfuse.matmul_with_bias
    return %0 : tensor<2x1xf32>
  }

  // N != 1: no fusion (no reshape added); matmul unchanged.
  // CHECK-LABEL: func @matmul_n_not_one
  func.func @matmul_n_not_one(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    // CHECK: mfuse.matmul {{.*}} : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}
