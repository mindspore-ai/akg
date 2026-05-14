// RUN: mfusion-opt %s --convert-mfuse-to-torch | FileCheck %s

// CHECK-LABEL: func.func @conv2d_no_bias_stride2
// CHECK: torch.aten.convolution
// CHECK-NOT: mfuse.aclnn.conv2d
func.func @conv2d_no_bias_stride2(%arg0: tensor<1x1x8x8xf32>, %arg1: tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32> {
  %0 = mfuse.aclnn.conv2d %arg0, %arg1 {dilation = [1, 1], groups = 1 : i64, output_padding = [0, 0], padding = [0, 0], stride = [2, 2], transposed = false} : (tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}

// CHECK-LABEL: func.func @conv2d_with_bias_stride2
// CHECK: torch.aten.convolution
// CHECK-NOT: mfuse.aclnn.conv2d_with_bias
func.func @conv2d_with_bias_stride2(%arg0: tensor<1x1x8x8xf32>, %arg1: tensor<1x1x3x3xf32>, %arg2: tensor<1xf32>)
    -> tensor<1x1x3x3xf32> {
  %0 = mfuse.aclnn.conv2d_with_bias %arg0, %arg1, %arg2 {dilation = [1, 1], groups = 1 : i64, output_padding = [0, 0], padding = [0, 0], stride = [2, 2], transposed = false} : (tensor<1x1x8x8xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>) -> tensor<1x1x3x3xf32>
  return %0 : tensor<1x1x3x3xf32>
}
