// matmul_with_bias decompose: matmul_with_bias -> matmul + add
// RUN: mfusion-opt %s --decompose="pattern-type=AFTER_MANUAL_FUSION op-list=matmul_with_bias" | FileCheck %s

// CHECK-LABEL: func.func @test_matmul_with_bias_2d
// CHECK-NOT: mfuse.matmul_with_bias
// CHECK: %[[MATMUL:.*]] = mfuse.matmul %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
// CHECK: %[[ADD:.*]] = mfuse.add %[[MATMUL]], %arg2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK: return %[[ADD]]
func.func @test_matmul_with_bias_2d(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<2x3xf32>, tensor<3x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @test_matmul_with_bias_transpose
// CHECK-NOT: mfuse.matmul_with_bias
// CHECK: %[[MATMUL:.*]] = mfuse.matmul %arg0, %arg1 {trans_x1 = true} : (tensor<3x2xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
// CHECK: %[[ADD:.*]] = mfuse.add %[[MATMUL]], %arg2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK: return %[[ADD]]
func.func @test_matmul_with_bias_transpose(%arg0: tensor<3x2xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 {trans_x1 = true} : (tensor<3x2xf32>, tensor<3x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @test_matmul_with_bias_transpose_both
// CHECK-NOT: mfuse.matmul_with_bias
// CHECK: %[[MATMUL:.*]] = mfuse.matmul %arg0, %arg1 {trans_x1 = true, trans_x2 = true} : (tensor<3x2xf32>, tensor<4x3xf32>) -> tensor<2x4xf32>
// CHECK: %[[ADD:.*]] = mfuse.add %[[MATMUL]], %arg2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK: return %[[ADD]]
func.func @test_matmul_with_bias_transpose_both(%arg0: tensor<3x2xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 {trans_x1 = true, trans_x2 = true} : (tensor<3x2xf32>, tensor<4x3xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @test_matmul_with_bias_broadcast_bias
// CHECK-NOT: mfuse.matmul_with_bias
// CHECK: %[[MATMUL:.*]] = mfuse.matmul %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
// CHECK: %[[ADD:.*]] = mfuse.add %[[MATMUL]], %arg2 : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
// CHECK: return %[[ADD]]
func.func @test_matmul_with_bias_broadcast_bias(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4xf32>) -> tensor<2x4xf32> {
  %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @test_matmul_with_bias_batch
// CHECK-NOT: mfuse.matmul_with_bias
// CHECK: %[[MATMUL:.*]] = mfuse.matmul %arg0, %arg1 : (tensor<8x2x3xf32>, tensor<8x3x4xf32>) -> tensor<8x2x4xf32>
// CHECK: %[[ADD:.*]] = mfuse.add %[[MATMUL]], %arg2 : (tensor<8x2x4xf32>, tensor<4xf32>) -> tensor<8x2x4xf32>
// CHECK: return %[[ADD]]
func.func @test_matmul_with_bias_batch(%arg0: tensor<8x2x3xf32>, %arg1: tensor<8x3x4xf32>, %arg2: tensor<4xf32>) -> tensor<8x2x4xf32> {
  %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<8x2x3xf32>, tensor<8x3x4xf32>, tensor<4xf32>) -> tensor<8x2x4xf32>
  return %0 : tensor<8x2x4xf32>
}
