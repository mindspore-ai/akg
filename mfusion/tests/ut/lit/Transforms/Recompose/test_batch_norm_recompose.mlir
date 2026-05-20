// RUN: mfusion-opt %s -recompose | FileCheck %s
// Recompose leaves mfuse.aclnn.batch_norm unchanged when no meta lowering applies.

module {
  // CHECK-LABEL: func @batch_norm_recompose_identity
  func.func @batch_norm_recompose_identity(%x: tensor<4x16x32x32xf32>, %mean: tensor<16xf32>, %var: tensor<16xf32>,
      %gamma: tensor<16xf32>, %beta: tensor<16xf32>) -> tensor<4x16x32x32xf32> {
    %0 = mfuse.aclnn.batch_norm %x, %gamma, %beta, %mean, %var {training = false, momentum = 0.000000e+00 : f64,
        epsilon = 1.000000e-05 : f64, cudnn_enable = false}
        : (tensor<4x16x32x32xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<4x16x32x32xf32>
    return %0 : tensor<4x16x32x32xf32>
    // CHECK: mfuse.aclnn.batch_norm
    // CHECK-DAG: epsilon = 1.000000e-05
    // CHECK-DAG: momentum = 0.000000e+00
  }
}
