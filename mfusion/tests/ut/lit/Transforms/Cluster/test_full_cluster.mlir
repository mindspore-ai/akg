// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// CHECK-LABEL: func.func @test_full_f32_can_cluster
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG1:.*]]: tensor<4x4xf32>):
// CHECK: %[[CST:.*]] = mfuse.constant
// CHECK: %[[FULL:.*]] = mfuse.full %[[CST]]
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG1]], %[[FULL]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: mfuse.yield %[[ADD]] : tensor<4x4xf32>
func.func @test_full_f32_can_cluster(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = mfuse.constant dense<1.0> : tensor<f32, {is_scalar = ""}>
  %0 = mfuse.full %cst : (tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
  %1 = mfuse.add %arg0, %0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_full_i64_cannot_cluster
// CHECK-NOT: mfuse.fused
// CHECK: %[[CST:.*]] = mfuse.constant dense<7.000000e+00> : tensor<f64, {is_scalar = ""}>
// CHECK: %[[FULL:.*]] = mfuse.full %[[CST]]
// CHECK: %[[CAST:.*]] = mfuse.cast %[[FULL]] : (tensor<4x4xsi64>) -> tensor<4x4xi32>
func.func @test_full_i64_cannot_cluster() -> tensor<4x4xi32> {
  %cst = mfuse.constant dense<7.000000e+00> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.full %cst : (tensor<f64, {is_scalar = ""}>) -> tensor<4x4xsi64>
  %1 = mfuse.cast %0 : (tensor<4x4xsi64>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}
}
