// RUN: mfusion-opt %s --fuse-biasadd-conv | FileCheck %s

module {
  // Conv2D (no bias) + Add(bias [C]) -> Conv2DWithBias. NCHW: out [1,4,3,3], bias [4].
  // CHECK-LABEL: func @fuse_biasadd_conv
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x2x4x4xf32>, %[[WEIGHT:.*]]: tensor<4x2x2x2xf32>, %[[BIAS:.*]]: tensor<4xf32>)
  func.func @fuse_biasadd_conv(%input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>) -> tensor<1x4x3x3xf32> {
    %conv = mfuse.conv2d %input, %weight : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias : (tensor<1x4x3x3xf32>, tensor<4xf32>) -> tensor<1x4x3x3xf32>
    // After fusion: conv2d and add replaced by conv2d_with_bias
    // CHECK-NOT: mfuse.conv2d
    // CHECK-NOT: mfuse.add
    // CHECK: %[[R:.*]] = mfuse.conv2d_with_bias %[[INPUT]], %[[WEIGHT]], %[[BIAS]]
    // CHECK: return %[[R]]
    return %result : tensor<1x4x3x3xf32>
  }

  // Add(conv, bias) with bias on LHS: same fusion (commutative).
  // CHECK-LABEL: func @fuse_biasadd_conv_bias_left
  func.func @fuse_biasadd_conv_bias_left(%input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>) -> tensor<1x4x3x3xf32> {
    %conv = mfuse.conv2d %input, %weight : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %bias, %conv : (tensor<4xf32>, tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32>
    // CHECK-NOT: mfuse.add
    // CHECK: mfuse.conv2d_with_bias
    return %result : tensor<1x4x3x3xf32>
  }

  // No fusion: bias shape [2] does not match output channels 4.
  // CHECK-LABEL: func @no_fusion_bias_wrong_size
  func.func @no_fusion_bias_wrong_size(%input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<2xf32>) -> tensor<1x4x3x3xf32> {
    %conv = mfuse.conv2d %input, %weight : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias : (tensor<1x4x3x3xf32>, tensor<2xf32>) -> tensor<1x4x3x3xf32>
    // CHECK: mfuse.conv2d
    // CHECK: mfuse.add
    // CHECK-NOT: mfuse.conv2d_with_bias
    return %result : tensor<1x4x3x3xf32>
  }

  // No fusion: bias is 2D [4,1], pass requires 1D [C].
  // CHECK-LABEL: func @no_fusion_bias_not_1d
  func.func @no_fusion_bias_not_1d(%input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4x1xf32>) -> tensor<1x4x3x3xf32> {
    %conv = mfuse.conv2d %input, %weight : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias : (tensor<1x4x3x3xf32>, tensor<4x1xf32>) -> tensor<1x4x3x3xf32>
    // CHECK: mfuse.conv2d
    // CHECK: mfuse.add
    // CHECK-NOT: mfuse.conv2d_with_bias
    return %result : tensor<1x4x3x3xf32>
  }

  // Conv2D + Add(conv, Reshape(bias_1d)): pass only accepts 1D bias (no Reshape); no fusion per biasadd_conv_fusion_pass (Conv2D has no Reshape-bias path).
  // CHECK-LABEL: func @fuse_biasadd_conv_bias_from_reshape
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x2x4x4xf32>, %[[WEIGHT:.*]]: tensor<4x2x2x2xf32>, %[[BIAS:.*]]: tensor<4xf32>)
  func.func @fuse_biasadd_conv_bias_from_reshape(%input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>) -> tensor<1x4x3x3xf32> {
    %bias_reshaped = mfuse.reshape %bias : (tensor<4xf32>) -> tensor<1x4x1x1xf32>
    %conv = mfuse.conv2d %input, %weight : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias_reshaped : (tensor<1x4x3x3xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x3x3xf32>
    // No fusion: bias is not 1D (Reshape output); pass requires 1D [C] only.
    // CHECK: mfuse.reshape
    // CHECK: mfuse.conv2d
    // CHECK: mfuse.add
    // CHECK-NOT: mfuse.conv2d_with_bias
    return %result : tensor<1x4x3x3xf32>
  }

  // No fusion: bias is Reshape output (not 1D); pass only accepts 1D [C] bias.
  // CHECK-LABEL: func @no_fusion_reshape_bias_wrong_size
  func.func @no_fusion_reshape_bias_wrong_size(%input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<2xf32>) -> tensor<1x4x3x3xf32> {
    %bias_reshaped = mfuse.reshape %bias : (tensor<2xf32>) -> tensor<1x2x1x1xf32>
    %conv = mfuse.conv2d %input, %weight : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias_reshaped : (tensor<1x4x3x3xf32>, tensor<1x2x1x1xf32>) -> tensor<1x4x3x3xf32>
    // CHECK: mfuse.reshape
    // CHECK: mfuse.conv2d
    // CHECK: mfuse.add
    // CHECK-NOT: mfuse.conv2d_with_bias
    return %result : tensor<1x4x3x3xf32>
  }
}
