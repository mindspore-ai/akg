// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// CHECK-LABEL: func @test_skip_cluster_with_rank0_external_inputs
// CHECK: %[[CAST:.*]] = mfuse.cast %arg1 : (tensor<f64>) -> tensor<f32>
// CHECK-NOT: mfuse.fused
// CHECK-NOT: mfuse.reshape
// CHECK: %[[DIV:.*]] = mfuse.div %arg0, %[[CAST]] : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
// CHECK-NOT: mfuse.fused
// CHECK-NOT: mfuse.reshape
// CHECK: %[[SELECT:.*]] = mfuse.select %arg2, %arg3, %[[DIV]] : (tensor<4x4xi1>, tensor<f32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK-NOT: mfuse.fused
// CHECK: return %[[SELECT]]
func.func @test_skip_cluster_with_rank0_external_inputs(
    %arg0: tensor<4x4xf32>,
    %arg1: tensor<f64>,
    %arg2: tensor<4x4xi1>,
    %arg3: tensor<f32>) -> tensor<4x4xf32> {
  %0 = mfuse.cast %arg1 : (tensor<f64>) -> tensor<f32>
  %1 = mfuse.div %arg0, %0 : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  %2 = mfuse.select %arg2, %arg3, %1 : (tensor<4x4xi1>, tensor<f32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// CHECK-LABEL: func @test_skip_cluster_with_full_wrapping_external_rank0
// CHECK: %[[FULL:.*]] = mfuse.full
// CHECK-NOT: mfuse.fused
// CHECK: %[[MUL:.*]] = mfuse.mul %arg0, %[[FULL]]
// CHECK-NOT: mfuse.fused
// CHECK: return %[[MUL]]
func.func @test_skip_cluster_with_full_wrapping_external_rank0(
    %arg0: tensor<4x4xf32>,
    %arg1: tensor<f32>) -> tensor<4x4xf32> {
  %0 = mfuse.full %arg1 : (tensor<f32>) -> tensor<f32>
  %1 = mfuse.mul %arg0, %0 : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// CHECK-LABEL: func @test_cluster_with_rank0_constant_cast
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK: %[[CST:.*]] = mfuse.constant dense<8.000000e+00> : tensor<f64, {is_scalar = ""}>
// CHECK: mfuse.mul
// CHECK: mfuse.add
func.func @test_cluster_with_rank0_constant_cast(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.constant dense<8.0> : tensor<f64, {is_scalar = ""}>
  %1 = mfuse.mul %arg0, %0 : (tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  %2 = mfuse.add %1, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// CHECK-LABEL: func @test_cluster_with_rank0_constant_input
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK: mfuse.constant dense<2.000000e+00> : tensor<f64, {is_scalar = ""}>
// CHECK: mfuse.mul
// CHECK: mfuse.add
func.func @test_cluster_with_rank0_constant_input(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.constant dense<2.0> : tensor<f64, {is_scalar = ""}>
  %1 = mfuse.mul %arg0, %0 : (tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  %2 = mfuse.add %1, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// CHECK-LABEL: func @test_cluster_with_rank0_i64_constant_input
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK: mfuse.constant dense<2> : tensor<i64, {is_scalar = ""}>
// CHECK: mfuse.mul
// CHECK: mfuse.add
func.func @test_cluster_with_rank0_i64_constant_input(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = mfuse.constant dense<2> : tensor<i64, {is_scalar = ""}>
  %1 = mfuse.mul %arg0, %0 : (tensor<4x4xi32>, tensor<i64, {is_scalar = ""}>) -> tensor<4x4xi32>
  %2 = mfuse.add %1, %arg0 : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %2 : tensor<4x4xi32>
}

// CHECK-LABEL: func @test_no_cluster_with_rank0_i64_out_of_range
// CHECK-NOT: mfuse.fused
// CHECK: return
func.func @test_no_cluster_with_rank0_i64_out_of_range(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = mfuse.constant dense<2147483648> : tensor<i64, {is_scalar = ""}>
  %1 = mfuse.mul %arg0, %0 : (tensor<4x4xi32>, tensor<i64, {is_scalar = ""}>) -> tensor<4x4xi32>
  %2 = mfuse.add %1, %arg0 : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %2 : tensor<4x4xi32>
}

// CHECK-LABEL: func @test_no_cluster_with_rank0_f64_out_of_range
// CHECK-NOT: mfuse.fused
// CHECK: return
func.func @test_no_cluster_with_rank0_f64_out_of_range(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.constant dense<1.000000e+40> : tensor<f64, {is_scalar = ""}>
  %1 = mfuse.mul %arg0, %0 : (tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  %2 = mfuse.add %1, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}
}
