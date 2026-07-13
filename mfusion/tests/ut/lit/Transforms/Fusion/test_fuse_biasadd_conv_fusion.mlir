// RUN: mfusion-opt %s --fuse-biasadd-conv | FileCheck %s

module {
  // Conv2D (no bias) + Add(bias [C]) -> Conv2DWithBias.
  // CHECK-LABEL: func @fuse_biasadd_conv
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x2x4x4xf32>, %[[WEIGHT:.*]]: tensor<4x2x2x2xf32>,
  // CHECK-SAME:              %[[BIAS:.*]]: tensor<4xf32>)
  // CHECK-NOT: mfuse.aclnn.conv2d
  // CHECK-NOT: mfuse.add
  // CHECK: %[[R:.*]] = mfuse.aclnn.conv2d_with_bias %[[INPUT]], %[[WEIGHT]], %[[BIAS]]
  // CHECK: return %[[R]]
  func.func @fuse_biasadd_conv(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x3xf32> {
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias : (tensor<1x4x3x3xf32>, tensor<4xf32>) -> tensor<1x4x3x3xf32>
    return %result : tensor<1x4x3x3xf32>
  }

  // Add(bias, conv): commutative, same fusion.
  // CHECK-LABEL: func @fuse_biasadd_conv_bias_left
  // CHECK-NOT: mfuse.add
  // CHECK: mfuse.aclnn.conv2d_with_bias
  func.func @fuse_biasadd_conv_bias_left(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x3xf32> {
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %bias, %conv : (tensor<4xf32>, tensor<1x4x3x3xf32>) -> tensor<1x4x3x3xf32>
    return %result : tensor<1x4x3x3xf32>
  }

  // No fusion: bias length mismatch.
  // CHECK-LABEL: func @no_fusion_bias_wrong_size
  // CHECK: mfuse.add
  // CHECK-NOT: mfuse.aclnn.conv2d_with_bias
  func.func @no_fusion_bias_wrong_size(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<2xf32>)
      -> tensor<1x4x3x3xf32> {
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias : (tensor<1x4x3x3xf32>, tensor<2xf32>) -> tensor<1x4x3x3xf32>
    return %result : tensor<1x4x3x3xf32>
  }

  // No fusion: bias is 2D [4,1], not canonical channel-broadcast reshape.
  // CHECK-LABEL: func @no_fusion_bias_not_1d
  // CHECK: mfuse.add
  // CHECK-NOT: mfuse.aclnn.conv2d_with_bias
  func.func @no_fusion_bias_not_1d(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4x1xf32>)
      -> tensor<1x4x3x3xf32> {
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias : (tensor<1x4x3x3xf32>, tensor<4x1xf32>) -> tensor<1x4x3x3xf32>
    return %result : tensor<1x4x3x3xf32>
  }

  // Reshape bias_1d -> [1,C,1,1] peeled to 1D for conv2d_with_bias.
  // CHECK-LABEL: func @fuse_biasadd_conv_bias_from_reshape
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x2x4x4xf32>, %[[WEIGHT:.*]]: tensor<4x2x2x2xf32>,
  // CHECK-SAME:              %[[BIAS:.*]]: tensor<4xf32>)
  // CHECK-NOT: mfuse.reshape
  // CHECK-NOT: mfuse.add
  // CHECK: %[[R:.*]] = mfuse.aclnn.conv2d_with_bias %[[INPUT]], %[[WEIGHT]], %[[BIAS]]
  // CHECK: return %[[R]]
  func.func @fuse_biasadd_conv_bias_from_reshape(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x3xf32> {
    %bias_reshaped = mfuse.reshape %bias : (tensor<4xf32>) -> tensor<1x4x1x1xf32>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias_reshaped
      : (tensor<1x4x3x3xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x3x3xf32>
    return %result : tensor<1x4x3x3xf32>
  }

  // Reshape bias_1d -> [C,1,1] peeled to 1D.
  // CHECK-LABEL: func @fuse_biasadd_conv_bias_from_reshape_c1x1
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x2x4x4xf32>, %[[WEIGHT:.*]]: tensor<4x2x2x2xf32>,
  // CHECK-SAME:              %[[BIAS:.*]]: tensor<4xf32>)
  // CHECK-NOT: mfuse.reshape
  // CHECK: mfuse.aclnn.conv2d_with_bias %[[INPUT]], %[[WEIGHT]], %[[BIAS]]
  func.func @fuse_biasadd_conv_bias_from_reshape_c1x1(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x3xf32> {
    %bias_reshaped = mfuse.reshape %bias : (tensor<4xf32>) -> tensor<4x1x1xf32>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias_reshaped
      : (tensor<1x4x3x3xf32>, tensor<4x1x1xf32>) -> tensor<1x4x3x3xf32>
    return %result : tensor<1x4x3x3xf32>
  }

  // Shared reshape has another user: no fusion.
  // CHECK-LABEL: func @keep_shared_bias_reshape
  // CHECK: mfuse.reshape
  // CHECK: mfuse.add
  // CHECK-NOT: mfuse.aclnn.conv2d_with_bias
  func.func @keep_shared_bias_reshape(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> (tensor<1x4x3x3xf32>, tensor<1x4x1x1xf32>) {
    %bias_reshaped = mfuse.reshape %bias : (tensor<4xf32>) -> tensor<1x4x1x1xf32>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias_reshaped
      : (tensor<1x4x3x3xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x3x3xf32>
    return %result, %bias_reshaped : tensor<1x4x3x3xf32>, tensor<1x4x1x1xf32>
  }

  // Mixed dtype: must NOT fuse.
  // CHECK-LABEL: func @no_fusion_mixed_dtype_bias
  // CHECK: mfuse.add
  // CHECK-NOT: mfuse.aclnn.conv2d_with_bias
  func.func @no_fusion_mixed_dtype_bias(
      %input: tensor<1x2x4x4xf16>, %weight: tensor<4x2x2x2xf16>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x3xf32> {
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf16>, tensor<4x2x2x2xf16>) -> tensor<1x4x3x3xf16>
    %result = mfuse.add %conv, %bias : (tensor<1x4x3x3xf16>, tensor<4xf32>) -> tensor<1x4x3x3xf32>
    return %result : tensor<1x4x3x3xf32>
  }

  // W == C: bare [C] ambiguous; do not channel-fuse.
  // CHECK-LABEL: func @no_fusion_ambiguous_w_equals_c
  // CHECK: mfuse.add
  // CHECK-NOT: mfuse.aclnn.conv2d_with_bias
  func.func @no_fusion_ambiguous_w_equals_c(
      %input: tensor<1x2x4x5xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x4xf32> {
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x5xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x4xf32>
    %result = mfuse.add %conv, %bias : (tensor<1x4x3x4xf32>, tensor<4xf32>) -> tensor<1x4x3x4xf32>
    return %result : tensor<1x4x3x4xf32>
  }

  // W == C with canonical [1,C,1,1] reshape: still fuse.
  // CHECK-LABEL: func @fuse_biasadd_conv_reshape_when_w_equals_c
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x2x4x5xf32>, %[[WEIGHT:.*]]: tensor<4x2x2x2xf32>,
  // CHECK-SAME:              %[[BIAS:.*]]: tensor<4xf32>)
  // CHECK-NOT: mfuse.reshape
  // CHECK-NOT: mfuse.add
  // CHECK: mfuse.aclnn.conv2d_with_bias %[[INPUT]], %[[WEIGHT]], %[[BIAS]]
  func.func @fuse_biasadd_conv_reshape_when_w_equals_c(
      %input: tensor<1x2x4x5xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x4xf32> {
    %bias_reshaped = mfuse.reshape %bias : (tensor<4xf32>) -> tensor<1x4x1x1xf32>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x5xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x4xf32>
    %result = mfuse.add %conv, %bias_reshaped
      : (tensor<1x4x3x4xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x3x4xf32>
    return %result : tensor<1x4x3x4xf32>
  }

  // Cast to dynamic/symbolic: must NOT fuse.
  // CHECK-LABEL: func @no_fusion_via_cast_to_dynamic
  // CHECK: mfuse.add
  // CHECK-NOT: mfuse.aclnn.conv2d_with_bias
  func.func @no_fusion_via_cast_to_dynamic(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>> {
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %conv_sym = builtin.unrealized_conversion_cast %conv
        : tensor<1x4x3x3xf32> to tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>>
    %result = mfuse.add %conv_sym, %bias
      : (tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>>, tensor<4xf32>)
        -> tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>>
    return %result : tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>>
  }

  // Fusion miss (conv multi-use): legalize bare [C] illegal Add -> reshape([1,C,1,1])+add.
  // CHECK-LABEL: func @legalize_bare_c_when_conv_multi_use
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x2x4x4xf32>, %[[WEIGHT:.*]]: tensor<4x2x2x2xf32>,
  // CHECK-SAME:              %[[BIAS:.*]]: tensor<4xf32>)
  // CHECK: mfuse.aclnn.conv2d
  // CHECK: %[[RSH:.*]] = mfuse.reshape %[[BIAS]] : (tensor<4xf32>) -> tensor<1x4x1x1xf32>
  // CHECK: mfuse.add %{{.*}}, %[[RSH]] : (tensor<1x4x3x3xf32>, tensor<1x4x1x1xf32>) -> tensor<1x4x3x3xf32>
  // CHECK-NOT: mfuse.aclnn.conv2d_with_bias
  // CHECK-NOT: mfuse.add %{{.*}}, %[[BIAS]] : (tensor<1x4x3x3xf32>, tensor<4xf32>)
  func.func @legalize_bare_c_when_conv_multi_use(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> (tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>) {
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %result = mfuse.add %conv, %bias : (tensor<1x4x3x3xf32>, tensor<4xf32>) -> tensor<1x4x3x3xf32>
    return %result, %conv : tensor<1x4x3x3xf32>, tensor<1x4x3x3xf32>
  }
}
