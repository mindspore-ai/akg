// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// Test matmul operation clustering
// NOTE: matmul operation does NOT cluster with add in this test case because:
// 1. This is treated as a 2D mm: (N x K) * (K x M)
// 2. DVMCluster::matMulShapeCheck rejects matmul when the static output's last two dims are both <= 256
// 3. Here the output is 128x256, so the matmul is rejected
// CHECK-LABEL: func @test_matmul_cluster
// CHECK-SAME: %arg0: tensor<128x256xf16>
// CHECK-SAME: %arg1: tensor<256x256xf16>
// CHECK: %[[MATMUL:.*]] = mfuse.matmul %arg0, %arg1
// CHECK-SAME: : (tensor<128x256xf16>, tensor<256x256xf16>) -> tensor<128x256xf16>
// CHECK: %[[ADD:.*]] = mfuse.add %[[MATMUL]], %[[MATMUL]]
// CHECK-SAME: : (tensor<128x256xf16>, tensor<128x256xf16>) -> tensor<128x256xf16>
// CHECK: return %[[ADD]]
func.func @test_matmul_cluster(%arg0: tensor<128x256xf16>, %arg1: tensor<256x256xf16>) -> tensor<128x256xf16> {
  // Matmul operation (only supports f16/bf16)
  %0 = mfuse.matmul %arg0, %arg1 : (tensor<128x256xf16>, tensor<256x256xf16>) -> tensor<128x256xf16>
  // Add operation with matmul result
  %1 = mfuse.add %0, %0 : (tensor<128x256xf16>, tensor<128x256xf16>) -> tensor<128x256xf16>
  return %1 : tensor<128x256xf16>
}

// Test 3D matmul clustering - BMM-form matmul can cluster when batch dimensions match
// CHECK-LABEL: func @test_bmm_form_matmul_cluster
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<2x128x256xf16>, tensor<2x256x512xf16>) -> tensor<2x128x512xf16>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x128x256xf16>, %[[ARG3:.*]]: tensor<2x256x512xf16>):
// CHECK: %[[BMM:.*]] = mfuse.matmul %[[ARG2]], %[[ARG3]]
// CHECK: %[[ADD:.*]] = mfuse.add %[[BMM]], %[[BMM]]
// CHECK: mfuse.yield %[[ADD]]
// CHECK: return %[[FUSED]]
func.func @test_bmm_form_matmul_cluster(%arg0: tensor<2x128x256xf16>, %arg1: tensor<2x256x512xf16>) -> tensor<2x128x512xf16> {
  %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x128x256xf16>, tensor<2x256x512xf16>) -> tensor<2x128x512xf16>
  // Add operation
  %1 = mfuse.add %0, %0 : (tensor<2x128x512xf16>, tensor<2x128x512xf16>) -> tensor<2x128x512xf16>
  return %1 : tensor<2x128x512xf16>
}

// Test 3D matmul with trans_x1 attribute
// CHECK-LABEL: func @test_bmm_form_matmul_transpose_a
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<2x256x128xf16>, tensor<2x256x512xf16>) -> tensor<2x128x512xf16>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x256x128xf16>, %[[ARG3:.*]]: tensor<2x256x512xf16>):
// CHECK: %[[BMM:.*]] = mfuse.matmul %[[ARG2]], %[[ARG3]]
// CHECK-SAME: {trans_x1 = true}
// CHECK: %[[ADD:.*]] = mfuse.add %[[BMM]], %[[BMM]]
// CHECK: mfuse.yield %[[ADD]]
// CHECK: return %[[FUSED]]
func.func @test_bmm_form_matmul_transpose_a(%arg0: tensor<2x256x128xf16>, %arg1: tensor<2x256x512xf16>) -> tensor<2x128x512xf16> {
  %0 = mfuse.matmul %arg0, %arg1 {trans_x1 = true} : (tensor<2x256x128xf16>, tensor<2x256x512xf16>) -> tensor<2x128x512xf16>
  // Add operation
  %1 = mfuse.add %0, %0 : (tensor<2x128x512xf16>, tensor<2x128x512xf16>) -> tensor<2x128x512xf16>
  return %1 : tensor<2x128x512xf16>
}

// Test 3D matmul with trans_x2 attribute
// CHECK-LABEL: func @test_bmm_form_matmul_transpose_b
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<2x128x256xf16>, tensor<2x512x256xf16>) -> tensor<2x128x512xf16>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x128x256xf16>, %[[ARG3:.*]]: tensor<2x512x256xf16>):
// CHECK: %[[BMM:.*]] = mfuse.matmul %[[ARG2]], %[[ARG3]]
// CHECK-SAME: {trans_x2 = true}
// CHECK: %[[ADD:.*]] = mfuse.add %[[BMM]], %[[BMM]]
// CHECK: mfuse.yield %[[ADD]]
// CHECK: return %[[FUSED]]
func.func @test_bmm_form_matmul_transpose_b(%arg0: tensor<2x128x256xf16>, %arg1: tensor<2x512x256xf16>) -> tensor<2x128x512xf16> {
  %0 = mfuse.matmul %arg0, %arg1 {trans_x2 = true} : (tensor<2x128x256xf16>, tensor<2x512x256xf16>) -> tensor<2x128x512xf16>
  // Add operation
  %1 = mfuse.add %0, %0 : (tensor<2x128x512xf16>, tensor<2x128x512xf16>) -> tensor<2x128x512xf16>
  return %1 : tensor<2x128x512xf16>
}

// Test 3D matmul with both trans_x1 and trans_x2 attributes
// CHECK-LABEL: func @test_bmm_form_matmul_transpose_both
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<2x256x128xf16>, tensor<2x512x256xf16>) -> tensor<2x128x512xf16>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x256x128xf16>, %[[ARG3:.*]]: tensor<2x512x256xf16>):
// CHECK: %[[BMM:.*]] = mfuse.matmul %[[ARG2]], %[[ARG3]]
// CHECK-SAME: {trans_x1 = true, trans_x2 = true}
// CHECK: %[[ADD:.*]] = mfuse.add %[[BMM]], %[[BMM]]
// CHECK: mfuse.yield %[[ADD]]
// CHECK: return %[[FUSED]]
func.func @test_bmm_form_matmul_transpose_both(%arg0: tensor<2x256x128xf16>, %arg1: tensor<2x512x256xf16>) -> tensor<2x128x512xf16> {
  %0 = mfuse.matmul %arg0, %arg1 {trans_x1 = true, trans_x2 = true} : (tensor<2x256x128xf16>, tensor<2x512x256xf16>) -> tensor<2x128x512xf16>
  // Add operation
  %1 = mfuse.add %0, %0 : (tensor<2x128x512xf16>, tensor<2x128x512xf16>) -> tensor<2x128x512xf16>
  return %1 : tensor<2x128x512xf16>
}
}
