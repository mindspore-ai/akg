// RUN: mfusion-opt %s --muse-dvm-cluster | FileCheck %s

module {
// Test matmul operation clustering
// NOTE: matmul operation does NOT cluster with add in this test case because:
// 1. The matmul output shape (128x512) fails the MatMulShapeCheck constraints
// 2. In DVMCluster::MatMulShapeCheck: when c_shape.back() <= kMinDimSize (512)
//    AND c_shape.size() >= 2 AND c_shape[c_shape.size() - 2] <= kMinDimSize,
//    the matmul is rejected for clustering
// 3. Here output is 128x512, where both dimensions (128, 512) are <= 512, triggering the rejection
// CHECK-LABEL: func @test_matmul_cluster
// CHECK-SAME: %arg0: tensor<128x256xf16>
// CHECK-SAME: %arg1: tensor<256x512xf16>
// CHECK: %[[MATMUL:.*]] = muse.matmul %arg0, %arg1
// CHECK-SAME: : (tensor<128x256xf16>, tensor<256x512xf16>) -> tensor<128x512xf16>
// CHECK: %[[ADD:.*]] = muse.add %[[MATMUL]], %[[MATMUL]]
// CHECK-SAME: : (tensor<128x512xf16>, tensor<128x512xf16>) -> tensor<128x512xf16>
// CHECK: return %[[ADD]]
func.func @test_matmul_cluster(%arg0: tensor<128x256xf16>, %arg1: tensor<256x512xf16>) -> tensor<128x512xf16> {
  // Matmul operation (only supports f16/bf16)
  %0 = muse.matmul %arg0, %arg1 : (tensor<128x256xf16>, tensor<256x512xf16>) -> tensor<128x512xf16>
  // Add operation with matmul result
  %1 = muse.add %0, %0 : (tensor<128x512xf16>, tensor<128x512xf16>) -> tensor<128x512xf16>
  return %1 : tensor<128x512xf16>
}

// Test batch_matmul clustering - batch_matmul can cluster when shape constraints are satisfied
// CHECK-LABEL: func @test_batch_matmul_cluster
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<2x128x256xf16>, tensor<2x256x512xf16>) -> tensor<2x128x512xf16>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x128x256xf16>, %[[ARG3:.*]]: tensor<2x256x512xf16>):
// CHECK: %[[BMM:.*]] = muse.batch_matmul %[[ARG2]], %[[ARG3]]
// CHECK: %[[ADD:.*]] = muse.add %[[BMM]], %[[BMM]]
// CHECK: muse.yield %[[ADD]]
// CHECK: return %[[FUSED]]
func.func @test_batch_matmul_cluster(%arg0: tensor<2x128x256xf16>, %arg1: tensor<2x256x512xf16>) -> tensor<2x128x512xf16> {
  // Batch matmul operation
  %0 = muse.batch_matmul %arg0, %arg1 : (tensor<2x128x256xf16>, tensor<2x256x512xf16>) -> tensor<2x128x512xf16>
  // Add operation
  %1 = muse.add %0, %0 : (tensor<2x128x512xf16>, tensor<2x128x512xf16>) -> tensor<2x128x512xf16>
  return %1 : tensor<2x128x512xf16>
}

// Test batch_matmul with transpose_a attribute
// CHECK-LABEL: func @test_batch_matmul_transpose_a
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<2x256x128xf16>, tensor<2x256x512xf16>) -> tensor<2x128x512xf16>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x256x128xf16>, %[[ARG3:.*]]: tensor<2x256x512xf16>):
// CHECK: %[[BMM:.*]] = muse.batch_matmul %[[ARG2]], %[[ARG3]]
// CHECK-SAME: {transpose_a = true}
// CHECK: %[[ADD:.*]] = muse.add %[[BMM]], %[[BMM]]
// CHECK: muse.yield %[[ADD]]
// CHECK: return %[[FUSED]]
func.func @test_batch_matmul_transpose_a(%arg0: tensor<2x256x128xf16>, %arg1: tensor<2x256x512xf16>) -> tensor<2x128x512xf16> {
  // Batch matmul operation with transpose_a
  %0 = muse.batch_matmul %arg0, %arg1 {transpose_a = true} : (tensor<2x256x128xf16>, tensor<2x256x512xf16>) -> tensor<2x128x512xf16>
  // Add operation
  %1 = muse.add %0, %0 : (tensor<2x128x512xf16>, tensor<2x128x512xf16>) -> tensor<2x128x512xf16>
  return %1 : tensor<2x128x512xf16>
}

// Test batch_matmul with transpose_b attribute
// CHECK-LABEL: func @test_batch_matmul_transpose_b
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<2x128x256xf16>, tensor<2x512x256xf16>) -> tensor<2x128x512xf16>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x128x256xf16>, %[[ARG3:.*]]: tensor<2x512x256xf16>):
// CHECK: %[[BMM:.*]] = muse.batch_matmul %[[ARG2]], %[[ARG3]]
// CHECK-SAME: {transpose_b = true}
// CHECK: %[[ADD:.*]] = muse.add %[[BMM]], %[[BMM]]
// CHECK: muse.yield %[[ADD]]
// CHECK: return %[[FUSED]]
func.func @test_batch_matmul_transpose_b(%arg0: tensor<2x128x256xf16>, %arg1: tensor<2x512x256xf16>) -> tensor<2x128x512xf16> {
  // Batch matmul operation with transpose_b
  %0 = muse.batch_matmul %arg0, %arg1 {transpose_b = true} : (tensor<2x128x256xf16>, tensor<2x512x256xf16>) -> tensor<2x128x512xf16>
  // Add operation
  %1 = muse.add %0, %0 : (tensor<2x128x512xf16>, tensor<2x128x512xf16>) -> tensor<2x128x512xf16>
  return %1 : tensor<2x128x512xf16>
}

// Test batch_matmul with both transpose_a and transpose_b attributes
// CHECK-LABEL: func @test_batch_matmul_transpose_both
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<2x256x128xf16>, tensor<2x512x256xf16>) -> tensor<2x128x512xf16>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x256x128xf16>, %[[ARG3:.*]]: tensor<2x512x256xf16>):
// CHECK: %[[BMM:.*]] = muse.batch_matmul %[[ARG2]], %[[ARG3]]
// CHECK-SAME: {transpose_a = true, transpose_b = true}
// CHECK: %[[ADD:.*]] = muse.add %[[BMM]], %[[BMM]]
// CHECK: muse.yield %[[ADD]]
// CHECK: return %[[FUSED]]
func.func @test_batch_matmul_transpose_both(%arg0: tensor<2x256x128xf16>, %arg1: tensor<2x512x256xf16>) -> tensor<2x128x512xf16> {
  // Batch matmul operation with both transpose_a and transpose_b
  %0 = muse.batch_matmul %arg0, %arg1 {transpose_a = true, transpose_b = true} : (tensor<2x256x128xf16>, tensor<2x512x256xf16>) -> tensor<2x128x512xf16>
  // Add operation
  %1 = muse.add %0, %0 : (tensor<2x128x512xf16>, tensor<2x128x512xf16>) -> tensor<2x128x512xf16>
  return %1 : tensor<2x128x512xf16>
}
}