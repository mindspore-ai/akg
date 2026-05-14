// RUN: mfusion-opt %s --fuse-conv-batchnorm | FileCheck %s

module {
  // CHECK-LABEL: func.func @fuse_conv2d_bn
  func.func @fuse_conv2d_bn(%x: tensor<1x1x4x4xf32>) -> tensor<1x2x2x2xf32> {
    %w = mfuse.constant dense<1.000000e+00> : tensor<2x1x3x3xf32>
    %mean = mfuse.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
    %var = mfuse.constant dense<[4.000000e+00, 9.000000e+00]> : tensor<2xf32>
    %gamma = mfuse.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
    %beta = mfuse.constant dense<[5.000000e-01, 1.500000e+00]> : tensor<2xf32>
    %conv = mfuse.aclnn.conv2d %x, %w {stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false, output_padding = [0, 0], groups = 1 : i64} : (tensor<1x1x4x4xf32>, tensor<2x1x3x3xf32>) -> tensor<1x2x2x2xf32>
    %bn = mfuse.aclnn.batch_norm %conv, %gamma, %beta, %mean, %var {training = false, momentum = 0.000000e+00 : f64,
        epsilon = 1.000000e-05 : f64, cudnn_enable = false}
      : (tensor<1x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
    // CHECK: %[[FW:.*]] = mfuse.constant dense<{{.*}}> : tensor<2x1x3x3xf32>
    // CHECK: %[[FB:.*]] = mfuse.constant dense<{{.*}}> : tensor<2xf32>
    // CHECK: %[[R:.*]] = mfuse.aclnn.conv2d_with_bias %{{.*}}, %[[FW]], %[[FB]]
    // CHECK-NOT: mfuse.aclnn.batch_norm
    return %bn : tensor<1x2x2x2xf32>
  }

  // CHECK-LABEL: func.func @not_fuse_nonconst_gamma
  func.func @not_fuse_nonconst_gamma(%x: tensor<1x1x4x4xf32>, %gamma: tensor<2xf32>) -> tensor<1x2x2x2xf32> {
    %w = mfuse.constant dense<1.000000e+00> : tensor<2x1x3x3xf32>
    %mean = mfuse.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
    %var = mfuse.constant dense<[4.000000e+00, 9.000000e+00]> : tensor<2xf32>
    %beta = mfuse.constant dense<[5.000000e-01, 1.500000e+00]> : tensor<2xf32>
    %conv = mfuse.aclnn.conv2d %x, %w {stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false, output_padding = [0, 0], groups = 1 : i64} : (tensor<1x1x4x4xf32>, tensor<2x1x3x3xf32>) -> tensor<1x2x2x2xf32>
    %bn = mfuse.aclnn.batch_norm %conv, %gamma, %beta, %mean, %var {training = false, momentum = 0.000000e+00 : f64,
        epsilon = 1.000000e-05 : f64, cudnn_enable = false}
      : (tensor<1x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
    // CHECK: mfuse.aclnn.batch_norm
    // CHECK-NOT: mfuse.aclnn.conv2d_with_bias
    return %bn : tensor<1x2x2x2xf32>
  }

  // CHECK-LABEL: func.func @not_fuse_transposed_conv
  func.func @not_fuse_transposed_conv(%x: tensor<1x1x4x4xf32>) -> tensor<1x2x6x6xf32> {
    %w = mfuse.constant dense<1.000000e+00> : tensor<2x1x3x3xf32>
    %mean = mfuse.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>
    %var = mfuse.constant dense<[4.000000e+00, 9.000000e+00]> : tensor<2xf32>
    %gamma = mfuse.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>
    %beta = mfuse.constant dense<[5.000000e-01, 1.500000e+00]> : tensor<2xf32>
    %conv = mfuse.aclnn.conv2d %x, %w {stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = true, output_padding = [0, 0], groups = 1 : i64} : (tensor<1x1x4x4xf32>, tensor<2x1x3x3xf32>) -> tensor<1x2x6x6xf32>
    %bn = mfuse.aclnn.batch_norm %conv, %gamma, %beta, %mean, %var {training = false, momentum = 0.000000e+00 : f64,
        epsilon = 1.000000e-05 : f64, cudnn_enable = false}
      : (tensor<1x2x6x6xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x6x6xf32>
    // CHECK: mfuse.aclnn.conv2d {{.*}} transposed = true
    // CHECK: mfuse.aclnn.batch_norm
    // CHECK-NOT: mfuse.aclnn.conv2d_with_bias
    return %bn : tensor<1x2x6x6xf32>
  }
}
