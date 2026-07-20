// RUN: mfusion-opt %s -decompose="pattern-type=BEFORE_MANUAL_FUSION" --fuse-biasadd-conv | FileCheck %s

module {
  // Conv2D + 1D channel bias: decompose aclnn.add -> mfuse.add, then fuse to conv2d_with_bias.
  // CHECK-LABEL: func.func @conv_bias_decompose_and_fuse
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x2x4x4xf32>, %[[WEIGHT:.*]]: tensor<4x2x2x2xf32>,
  // CHECK-SAME:              %[[BIAS:.*]]: tensor<4xf32>)
  // CHECK-NOT: mfuse.aclnn.conv2d
  // CHECK-NOT: mfuse.add
  // CHECK-NOT: mfuse.aclnn.add
  // CHECK: %[[R:.*]] = mfuse.aclnn.conv2d_with_bias %[[INPUT]], %[[WEIGHT]], %[[BIAS]]
  // CHECK: return %[[R]]
  func.func @conv_bias_decompose_and_fuse(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x3xf32> {
    %alpha = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.aclnn.add %conv, %bias, %alpha
      : (tensor<1x4x3x3xf32>, tensor<4xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<1x4x3x3xf32>
    return %result : tensor<1x4x3x3xf32>
  }

  // Non-unit alpha: mul(bias, alpha) then fuse scaled bias into conv2d_with_bias.
  // CHECK-LABEL: func.func @conv_bias_decompose_and_fuse_scaled_alpha
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x2x4x4xf32>, %[[WEIGHT:.*]]: tensor<4x2x2x2xf32>,
  // CHECK-SAME:              %[[BIAS:.*]]: tensor<4xf32>)
  // CHECK-NOT: mfuse.aclnn.add
  // CHECK-NOT: mfuse.add
  // CHECK: %[[SCALED:.*]] = mfuse.mul %[[BIAS]], %{{.*}}
  // CHECK: %[[R:.*]] = mfuse.aclnn.conv2d_with_bias %[[INPUT]], %[[WEIGHT]], %[[SCALED]]
  // CHECK: return %[[R]]
  func.func @conv_bias_decompose_and_fuse_scaled_alpha(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x3xf32> {
    %alpha = mfuse.constant dense<2> : tensor<i64, {is_scalar = ""}>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.aclnn.add %conv, %bias, %alpha
      : (tensor<1x4x3x3xf32>, tensor<4xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<1x4x3x3xf32>
    return %result : tensor<1x4x3x3xf32>
  }
}
