// RUN: mfusion-opt %s --recompose | FileCheck %s

module {
// Test basic add to aclnn.add conversion
// CHECK-LABEL: func @test_basic_add
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<i64>
// CHECK: %[[ACLNN_ADD:.*]] = mfuse.aclnn.add %arg0, %arg1, %[[CST]] : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<i64>) -> tensor<4x4xf32>
// CHECK: return %[[ACLNN_ADD]]
func.func @test_basic_add(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Test add with scalar operand
// CHECK-LABEL: func @test_add_with_scalar
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<f32>
// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<i64>
// CHECK: %[[ACLNN_ADD:.*]] = mfuse.aclnn.add %arg0, %arg1, %[[CST]] : (tensor<4x4xf32>, tensor<f32>, tensor<i64>) -> tensor<4x4xf32>
// CHECK: return %[[ACLNN_ADD]]
func.func @test_add_with_scalar(%arg0: tensor<4x4xf32>, %arg1: tensor<f32>) -> tensor<4x4xf32> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// Test add with mul scalar (x operand)
// CHECK-LABEL: func @test_add_with_mul_scalar_x
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<f32>
// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<i64>
// CHECK: %[[ACLNN_ADD:.*]] = mfuse.aclnn.add %0, %arg0, %[[CST]] : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<i64>) -> tensor<4x4xf32>
// CHECK: return %[[ACLNN_ADD]]
func.func @test_add_with_mul_scalar_x(%arg0: tensor<4x4xf32>, %arg1: tensor<f32>) -> tensor<4x4xf32> {
  %0 = mfuse.mul %arg0, %arg1 : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  %1 = mfuse.add %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test add with mul scalar (y operand)
// CHECK-LABEL: func @test_add_with_mul_scalar_y
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<f32>
// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<i64>
// CHECK: %[[ACLNN_ADD:.*]] = mfuse.aclnn.add %arg0, %0, %[[CST]] : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<i64>) -> tensor<4x4xf32>
// CHECK: return %[[ACLNN_ADD]]
func.func @test_add_with_mul_scalar_y(%arg0: tensor<4x4xf32>, %arg1: tensor<f32>) -> tensor<4x4xf32> {
  %0 = mfuse.mul %arg0, %arg1 : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  %1 = mfuse.add %arg0, %0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}
}
