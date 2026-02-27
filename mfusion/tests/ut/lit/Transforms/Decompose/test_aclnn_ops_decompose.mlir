// RUN: mfusion-opt %s -decompose="pattern-type=BEFORE_MANUAL_FUSION" -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @test_aclnn_add(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Create alpha constant (0.5)
  %alpha = arith.constant dense<0.5> : tensor<f32>
  // AclnnAdd with alpha != 1
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.add
  // CHECK: mfuse.mul
  // CHECK: mfuse.add
}

func.func @test_aclnn_add_alpha_one(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Create alpha constant (1.0)
  %alpha = arith.constant dense<1.0> : tensor<f32>
  // AclnnAdd with alpha == 1
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.add
  // CHECK: mfuse.add
  // CHECK-NOT: mfuse.mul
}

func.func @test_aclnn_sub(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Create alpha constant (0.5)
  %alpha = arith.constant dense<0.5> : tensor<f32>
  // AclnnSub with alpha != 1
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.sub
  // CHECK: mfuse.mul
  // CHECK: mfuse.sub
}

func.func @test_aclnn_sub_alpha_one(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Create alpha constant (1.0)
  %alpha = arith.constant dense<1.0> : tensor<f32>
  // AclnnSub with alpha == 1
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.sub
  // CHECK: mfuse.sub
  // CHECK-NOT: mfuse.mul
}
