// RUN: mfusion-opt %s -decompose="pattern-type=BEFORE_MANUAL_FUSION" -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

module {
// CHECK-LABEL: @aclnn_add_test
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.mul
// CHECK: mfuse.add
func.func @aclnn_add_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %alpha = arith.constant dense<0.5> : tensor<f32>
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_add_test_with_f64_alpha
// CHECK: mfuse.aclnn.add
func.func @aclnn_add_test_with_f64_alpha(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %alpha = arith.constant dense<0.5> : tensor<f64>
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_add_alpha_one_with_f64_input
// CHECK: mfuse.aclnn.add
func.func @aclnn_add_alpha_one_with_f64_input(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf64>) -> tensor<4x4xf32> {
  %alpha = arith.constant dense<0.5> : tensor<f32>
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf64>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_add_alpha_one_test
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.add
// CHECK-NOT: mfuse.mul
func.func @aclnn_add_alpha_one_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Create alpha constant (1.0)
  %alpha = arith.constant dense<1.0> : tensor<f64>
  // AclnnAdd with alpha == 1
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_sub_test
// CHECK: mfuse.aclnn.sub
func.func @aclnn_sub_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Create alpha constant (0.5)
  %alpha = arith.constant dense<0.5> : tensor<f64>
  // AclnnSub with alpha != 1
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_sub_alpha_one_test
// CHECK-NOT: mfuse.aclnn.sub
// CHECK: mfuse.sub
// CHECK-NOT: mfuse.mul
func.func @aclnn_sub_alpha_one_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %alpha = arith.constant dense<1.0> : tensor<f64>
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_sub_alpha_with_cast
// CHECK-NOT: mfuse.aclnn.sub
// CHECK: mfuse.cast
// CHECK: mfuse.mul
// CHECK: mfuse.sub
func.func @aclnn_sub_alpha_with_cast(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf16>, %alpha: tensor<f32>) -> tensor<4x4xf32> {
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf16>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
}
