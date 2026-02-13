// RUN: mfusion-opt %s --fuse-conv2d-cast | FileCheck %s

module {
  // Conv2D (f16) + Cast (f32) -> single Conv2D with f32 output.
  // CHECK-LABEL: func @fuse_conv2d_cast
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<1x2x4x4xf16>, %[[WEIGHT:.*]]: tensor<4x2x2x2xf16>)
  func.func @fuse_conv2d_cast(%input: tensor<1x2x4x4xf16>, %weight: tensor<4x2x2x2xf16>) -> tensor<1x4x3x3xf32> {
    %conv = mfuse.conv2d %input, %weight : (tensor<1x2x4x4xf16>, tensor<4x2x2x2xf16>) -> tensor<1x4x3x3xf16>
    %result = mfuse.cast %conv {dtype = f32} : (tensor<1x4x3x3xf16>) -> tensor<1x4x3x3xf32>
    // After fusion: cast replaced by conv2d with f32 result; cast eliminated
    // CHECK-NOT: mfuse.cast
    // CHECK: %[[R:.*]] = mfuse.conv2d %[[INPUT]], %[[WEIGHT]] : (tensor<1x2x4x4xf16>, tensor<4x2x2x2xf16>) -> tensor<1x4x3x3xf32>
    // CHECK: return %[[R]]
    return %result : tensor<1x4x3x3xf32>
  }

  // Conv2D already f32: no f16->f32 cast, so no fusion.
  // CHECK-LABEL: func @no_fusion_conv_already_f32
  func.func @no_fusion_conv_already_f32(%input: tensor<1x2x4x4xf32>, %weight: tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32> {
    %conv = mfuse.conv2d %input, %weight : (tensor<1x2x4x4xf32>, tensor<4x2x2x2xf32>) -> tensor<1x4x3x3xf32>
    // CHECK: mfuse.conv2d
    return %conv : tensor<1x4x3x3xf32>
  }

  // Conv2D (f16) + Cast to f16 (not f32): fusion only for f16->f32, so no fusion.
  // CHECK-LABEL: func @no_fusion_cast_to_f16
  func.func @no_fusion_cast_to_f16(%input: tensor<1x2x4x4xf16>, %weight: tensor<4x2x2x2xf16>) -> tensor<1x4x3x3xf16> {
    %conv = mfuse.conv2d %input, %weight : (tensor<1x2x4x4xf16>, tensor<4x2x2x2xf16>) -> tensor<1x4x3x3xf16>
    %result = mfuse.cast %conv {dtype = f16} : (tensor<1x4x3x3xf16>) -> tensor<1x4x3x3xf16>
    // CHECK: mfuse.conv2d
    // CHECK: mfuse.cast
    return %result : tensor<1x4x3x3xf16>
  }

  // Conv2D -> middle op (reshape) -> Cast: no fusion (Cast is not direct user of Conv2D).
  // CHECK-LABEL: func @no_fusion_conv_middle_op_then_cast
  func.func @no_fusion_conv_middle_op_then_cast(%input: tensor<1x2x4x4xf16>, %weight: tensor<4x2x2x2xf16>) -> tensor<1x4x3x3xf32> {
    %conv = mfuse.conv2d %input, %weight : (tensor<1x2x4x4xf16>, tensor<4x2x2x2xf16>) -> tensor<1x4x3x3xf16>
    %mid = mfuse.reshape %conv : (tensor<1x4x3x3xf16>) -> tensor<1x4x3x3xf16>
    %result = mfuse.cast %mid {dtype = f32} : (tensor<1x4x3x3xf16>) -> tensor<1x4x3x3xf32>
    // CHECK: mfuse.conv2d
    // CHECK: mfuse.reshape
    // CHECK: mfuse.cast
    return %result : tensor<1x4x3x3xf32>
  }
}
