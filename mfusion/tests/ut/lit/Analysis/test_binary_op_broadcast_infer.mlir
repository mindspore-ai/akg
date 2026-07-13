// RUN: mfusion-opt %s -decompose="pattern-type=BEFORE_MANUAL_FUSION" --verify-each | FileCheck %s

module {
  // Positive: conv2d [N,C,H,W] + 1D bias [C] decomposes via ConvBiasAddInfer.
  // CHECK-LABEL: func.func @decompose_nchw_conv_plus_1d_bias
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x2x4x4xf32>, %[[WEIGHT:.*]]: tensor<4x2x2x2xf32>,
  // CHECK-SAME:              %[[BIAS:.*]]: tensor<4xf32>)
  // CHECK-NOT: mfuse.aclnn.add
  // CHECK: mfuse.aclnn.conv2d
  // CHECK: mfuse.add %{{.*}}, %[[BIAS]] : (tensor<1x4x3x3xf32>, tensor<4xf32>) -> tensor<1x4x3x3xf32>
  func.func @decompose_nchw_conv_plus_1d_bias(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x3xf32> {
    %alpha = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %0 = mfuse.aclnn.add %conv, %bias, %alpha
        : (tensor<1x4x3x3xf32>, tensor<4xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<1x4x3x3xf32>
    return %0 : tensor<1x4x3x3xf32>
  }

  // Positive: conv2d + bias with symbolic shape on conv output (via cast).
  // CHECK-LABEL: func.func @decompose_conv_dynamic_hw_plus_1d_bias
  // CHECK-NOT: mfuse.aclnn.add
  // CHECK: mfuse.add %{{.*}}, %{{.*}} : (tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>>, tensor<4xf32>) -> tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>>
  func.func @decompose_conv_dynamic_hw_plus_1d_bias(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>> {
    %alpha = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %conv_sym = builtin.unrealized_conversion_cast %conv
        : tensor<1x4x3x3xf32> to tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>>
    %0 = mfuse.aclnn.add %conv_sym, %bias, %alpha
        : (tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>>, tensor<4xf32>,
           tensor<i64, {is_scalar = ""}>) -> tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>>
    return %0 : tensor<1x4x?x?xf32, #mfuse.symshape<["1", "4", "s0", "s1"]>>
  }

  // Positive: matmul-style [M,K] + [K] uses trailing broadcast (not conv infer).
  // CHECK-LABEL: func.func @decompose_matmul_plus_1d_bias
  // CHECK-NOT: mfuse.aclnn.add
  // CHECK: mfuse.add {{.*}}, {{.*}} : (tensor<3x16xf32>, tensor<16xf32>) -> tensor<3x16xf32>
  func.func @decompose_matmul_plus_1d_bias(%matmul: tensor<3x16xf32>, %bias: tensor<16xf32>)
      -> tensor<3x16xf32> {
    %alpha = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %0 = mfuse.aclnn.add %matmul, %bias, %alpha
        : (tensor<3x16xf32>, tensor<16xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<3x16xf32>
    return %0 : tensor<3x16xf32>
  }

  // Positive: trailing broadcast preserves dynamic dims for tensor * scalar.
  // CHECK-LABEL: func.func @decompose_dynamic_tensor_times_scalar
  // CHECK: mfuse.mul {{.*}} : (tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, tensor<f32, {is_scalar = ""}>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  func.func @decompose_dynamic_tensor_times_scalar(%lhs: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>,
                                                   %scalar: tensor<f32, {is_scalar = ""}>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>> {
    %0 = mfuse.mul %lhs, %scalar : (tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, tensor<f32, {is_scalar = ""}>)
        -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
    return %0 : tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  }

  // Negative: [N,C,H,W] + [C] without conv2d producer -> no decompose.
  // CHECK-LABEL: func.func @keep_nchw_tensor_without_conv
  // CHECK: mfuse.aclnn.add
  // CHECK-NOT: mfuse.add
  func.func @keep_nchw_tensor_without_conv(%lhs: tensor<2x8x5x5xf32>, %bias: tensor<8xf32>)
      -> tensor<2x8x5x5xf32> {
    %alpha = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %0 = mfuse.aclnn.add %lhs, %bias, %alpha
        : (tensor<2x8x5x5xf32>, tensor<8xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<2x8x5x5xf32>
    return %0 : tensor<2x8x5x5xf32>
  }

  // Negative: conv2d output [1,C,H,W] with H==C is ambiguous for channel-bias infer.
  // CHECK-LABEL: func.func @keep_ambiguous_channel_dim_with_conv
  // CHECK: mfuse.aclnn.add
  // CHECK-NOT: mfuse.add
  func.func @keep_ambiguous_channel_dim_with_conv(
      %input: tensor<1x2x5x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x4x3xf32> {
    %alpha = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x5x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x4x3xf32>
    %0 = mfuse.aclnn.add %conv, %bias, %alpha
        : (tensor<1x4x4x3xf32>, tensor<4xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<1x4x4x3xf32>
    return %0 : tensor<1x4x4x3xf32>
  }

  // Negative: ConvBiasAddInfer is Add-only; unit-alpha Sub must not use channel-bias fallback.
  // CHECK-LABEL: func.func @keep_aclnn_sub_conv_minus_1d_bias
  // CHECK: mfuse.aclnn.sub
  // CHECK-NOT: mfuse.sub
  func.func @keep_aclnn_sub_conv_minus_1d_bias(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x3xf32> {
    %alpha = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %0 = mfuse.aclnn.sub %conv, %bias, %alpha
        : (tensor<1x4x3x3xf32>, tensor<4xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<1x4x3x3xf32>
    return %0 : tensor<1x4x3x3xf32>
  }

  // Negative: mixed dtype must not use ConvBiasAddInfer (keep aclnn.add).
  // CHECK-LABEL: func.func @keep_aclnn_add_mixed_dtype_conv_bias
  // CHECK: mfuse.aclnn.add
  // CHECK-NOT: mfuse.add
  func.func @keep_aclnn_add_mixed_dtype_conv_bias(
      %input: tensor<1x2x4x4xf16>, %weight: tensor<4x2x2x2xf16>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x3xf32> {
    %alpha = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf16>, tensor<4x2x2x2xf16>) -> tensor<1x4x3x3xf16>
    %0 = mfuse.aclnn.add %conv, %bias, %alpha
        : (tensor<1x4x3x3xf16>, tensor<4xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<1x4x3x3xf32>
    return %0 : tensor<1x4x3x3xf32>
  }

  // Negative: non-unit alpha with (bias, conv) must NOT use channel-bias
  // (alpha scales y=conv -> bias + alpha*conv, not conv + alpha*bias).
  // CHECK-LABEL: func.func @keep_aclnn_add_bias_left_scaled_alpha
  // CHECK: mfuse.aclnn.add
  // CHECK-NOT: mfuse.add
  // CHECK-NOT: mfuse.mul
  func.func @keep_aclnn_add_bias_left_scaled_alpha(
      %input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>, %bias: tensor<4xf32>)
      -> tensor<1x4x3x3xf32> {
    %alpha = mfuse.constant dense<2> : tensor<i64, {is_scalar = ""}>
    %conv = mfuse.aclnn.conv2d %input, %weight {
        stride = [1, 1], padding = [0, 0], dilation = [1, 1], transposed = false,
        output_padding = [0, 0], groups = 1 : i64}
      : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    %0 = mfuse.aclnn.add %bias, %conv, %alpha
        : (tensor<4xf32>, tensor<1x4x3x3xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<1x4x3x3xf32>
    return %0 : tensor<1x4x3x3xf32>
  }
}
