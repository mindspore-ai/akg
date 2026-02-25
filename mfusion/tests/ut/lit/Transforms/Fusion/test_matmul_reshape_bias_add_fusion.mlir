// RUN: mfusion-opt %s --fuse-matmul-reshape-bias-add | FileCheck %s

module {
  // MatMul -> Reshape -> Add(bias) => MatMul(with bias) -> Reshape
  // CHECK-LABEL: func @fuse_reshape_bias_add
  func.func @fuse_reshape_bias_add(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %bias: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.reshape %0 : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %2 = mfuse.add %1, %bias : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    // After fusion: matmul_with_bias then reshape, no separate add
    // CHECK-NOT: mfuse.add
    // CHECK: mfuse.matmul_with_bias
    // CHECK: mfuse.reshape
    return %2 : tensor<2x8xf32>
  }

  // Reshape between matmul and add: MatMul [2,8] -> Reshape [2,4,2] -> Add(bias [8])
  // CHECK-LABEL: func @fuse_matmul_reshape_bias
  func.func @fuse_matmul_reshape_bias(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %bias: tensor<8xf32>) -> tensor<2x4x2xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.reshape %0 : (tensor<2x8xf32>) -> tensor<2x4x2xf32>
    %2 = mfuse.add %1, %bias : (tensor<2x4x2xf32>, tensor<8xf32>) -> tensor<2x4x2xf32>
    // CHECK: mfuse.matmul_with_bias
    // CHECK: mfuse.reshape
    return %2 : tensor<2x4x2xf32>
  }

  // MatMul -> Reshape -> Add(bias) => MatMulWithBias -> Reshape (2D case)
  // CHECK-LABEL: func @fuse_mm_reshape_bias_add
  func.func @fuse_mm_reshape_bias_add(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %bias: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.reshape %0 : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %2 = mfuse.add %1, %bias : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    // CHECK-NOT: mfuse.add
    // CHECK: mfuse.matmul_with_bias
    // CHECK: mfuse.reshape
    return %2 : tensor<2x8xf32>
  }

  // MatmulWithBias -> Reshape -> Add(extra bias) => MatmulWithBias(combined bias) -> Reshape
  // Combined bias = old_bias + add_bias (one mfuse.add for the two bias values)
  // CHECK-LABEL: func @fuse_mm_with_bias_reshape_bias_add
  func.func @fuse_mm_with_bias_reshape_bias_add(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %bias0: tensor<8xf32>, %bias1: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %bias0 : (tensor<2x4xf32>, tensor<4x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.reshape %0 : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %2 = mfuse.add %1, %bias1 : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    // CHECK: mfuse.add
    // CHECK: mfuse.matmul_with_bias
    // CHECK: mfuse.reshape
    return %2 : tensor<2x8xf32>
  }

  // MatmulWithBias -> Reshape -> Add(extra bias) => MatmulWithBias(combined bias) -> Reshape
  // CHECK-LABEL: func @fuse_matmul_with_bias_reshape_bias_add
  func.func @fuse_matmul_with_bias_reshape_bias_add(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %bias0: tensor<8xf32>, %bias1: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %bias0 : (tensor<2x4xf32>, tensor<4x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.reshape %0 : (tensor<2x8xf32>) -> tensor<2x8xf32>
    %2 = mfuse.add %1, %bias1 : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    // CHECK: mfuse.add
    // CHECK: mfuse.matmul_with_bias
    // CHECK: mfuse.reshape
    return %2 : tensor<2x8xf32>
  }
}
