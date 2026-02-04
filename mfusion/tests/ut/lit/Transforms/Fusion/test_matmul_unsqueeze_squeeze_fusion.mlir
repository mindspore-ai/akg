// RUN: mfusion-opt %s --fuse-matmul-unsqueeze-squeeze | FileCheck %s

module {
  // 1D x 2D: [K] x [K, M] -> [M]; pass inserts reshape to [1,K] and [K,M], matmul [1,M], reshape to [M]
  // CHECK-LABEL: func @one_d_two_d
  func.func @one_d_two_d(%arg0: tensor<4xf32>, %arg1: tensor<4x8xf32>) -> tensor<8xf32> {
    %0 = muse.matmul %arg0, %arg1 : (tensor<4xf32>, tensor<4x8xf32>) -> tensor<8xf32>
    // Pass inserts reshape(unsqueeze) before 1D input, matmul, reshape(squeeze) after
    // CHECK: muse.reshape
    // CHECK: muse.matmul
    // CHECK: muse.reshape
    return %0 : tensor<8xf32>
  }

  // 1D x 2D with bias: matmul_with_bias gets reshape before/after
  // CHECK-LABEL: func @one_d_two_d_mm_with_bias
  func.func @one_d_two_d_mm_with_bias(%arg0: tensor<4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
    %0 = muse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<4xf32>, tensor<4x8xf32>, tensor<8xf32>) -> tensor<8xf32>
    // CHECK: muse.reshape
    // CHECK: muse.matmul_with_bias
    // CHECK: muse.reshape
    return %0 : tensor<8xf32>
  }
}
