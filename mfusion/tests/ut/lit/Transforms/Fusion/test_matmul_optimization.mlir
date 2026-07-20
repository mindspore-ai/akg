// RUN: mfusion-opt %s --matmul-optimization | FileCheck %s

module {
  // Pass 2 (UnsqueezeSqueeze): normalize 1D inputs via reshape.
  // CHECK-LABEL: func @matmul_opt_unsqueeze
  func.func @matmul_opt_unsqueeze(%arg0: tensor<4xf32>, %arg1: tensor<4x8xf32>) -> tensor<8xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<4xf32>, tensor<4x8xf32>) -> tensor<8xf32>
    // CHECK: mfuse.reshape
    // CHECK: mfuse.matmul
    // CHECK: mfuse.reshape
    return %0 : tensor<8xf32>
  }

  // Pass 1+5 (Cast, BiasAdd): 2D inputs so UnsqueezeSqueeze is a no-op; cast then bias fuse.
  // CHECK-LABEL: func @matmul_opt_cast_bias
  func.func @matmul_opt_cast_bias(%arg0: tensor<2x4xf16>, %arg1: tensor<4x8xf16>, %bias: tensor<8xf32>)
      -> tensor<2x8xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf16>, tensor<4x8xf16>) -> tensor<2x8xf16>
    %1 = mfuse.cast %0 : (tensor<2x8xf16>) -> tensor<2x8xf32>
    %2 = mfuse.add %1, %bias : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    // CHECK-NOT: mfuse.cast
    // CHECK-NOT: mfuse.add
    // CHECK: mfuse.matmul_with_bias
    return %2 : tensor<2x8xf32>
  }

  // Pass 4 (K1ToMul): K=1 matmul -> mul.
  // CHECK-LABEL: func @matmul_opt_k1_to_mul
  func.func @matmul_opt_k1_to_mul(%arg0: tensor<3x1xf32>, %arg1: tensor<1x4xf32>) -> tensor<3x4xf32> {
    %0 = mfuse.matmul %arg0, %arg1 {trans_x1 = false, trans_x2 = false}
        : (tensor<3x1xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
    // CHECK-NOT: mfuse.matmul
    // CHECK: mfuse.mul
    return %0 : tensor<3x4xf32>
  }

  // Pass 6 (ReshapeBiasAdd): matmul -> reshape -> add.
  // CHECK-LABEL: func @matmul_opt_reshape_bias
  func.func @matmul_opt_reshape_bias(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %bias: tensor<8xf32>)
      -> tensor<2x4x2xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.reshape %0 : (tensor<2x8xf32>) -> tensor<2x4x2xf32>
    %2 = mfuse.add %1, %bias : (tensor<2x4x2xf32>, tensor<8xf32>) -> tensor<2x4x2xf32>
    // CHECK-NOT: mfuse.add
    // CHECK: mfuse.matmul_with_bias
    // CHECK: mfuse.reshape
    return %2 : tensor<2x4x2xf32>
  }
}
