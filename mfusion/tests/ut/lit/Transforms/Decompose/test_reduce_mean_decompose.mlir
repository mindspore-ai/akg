// reduce_mean decompose: f32 -> sum+div; dtype-changing inputs cast first; f16/bf16 -> cast f32, sum+div, cast back.
// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=reduce_mean" | FileCheck %s

// CHECK-LABEL: func.func @test_reduce_mean_decompose
// CHECK-NOT: mfuse.reduce_mean
// CHECK-DAG: %[[CST:.*]] = mfuse.constant dense<4> : tensor<i64, {is_scalar = ""}>
// CHECK-DAG: %[[REDUCE:.*]] = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = true}
// CHECK: %[[DIV:.*]] = mfuse.div %[[REDUCE]], %[[CST]]
// CHECK: return %[[DIV]]
func.func @test_reduce_mean_decompose(%arg0: tensor<2x4xf32>) -> tensor<2x1xf32> {
  %0 = mfuse.reduce_mean %arg0 {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// CHECK-LABEL: func.func @test_reduce_mean_decompose_int_input_float_result
// CHECK-NOT: mfuse.reduce_mean
// CHECK-DAG: %[[CST:.*]] = mfuse.constant dense<4> : tensor<i64, {is_scalar = ""}>
// CHECK-DAG: %[[IN_F32:.*]] = mfuse.cast %arg0 : (tensor<2x4xi32>) -> tensor<2x4xf32>
// CHECK-DAG: %[[REDUCE:.*]] = mfuse.reduce_sum %[[IN_F32]] {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
// CHECK: %[[DIV:.*]] = mfuse.div %[[REDUCE]], %[[CST]]
// CHECK: return %[[DIV]]
func.func @test_reduce_mean_decompose_int_input_float_result(%arg0: tensor<2x4xi32>) -> tensor<2x1xf32> {
  %0 = mfuse.reduce_mean %arg0 {dimensions = [1], keepdim = true} : (tensor<2x4xi32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// CHECK-LABEL: func.func @test_reduce_mean_decompose_int_input_f16_result
// CHECK-NOT: mfuse.cast %arg0 : (tensor<2x4xi32>) -> tensor<2x4xf16>
// CHECK-DAG: %[[CST:.*]] = mfuse.constant dense<4> : tensor<i64, {is_scalar = ""}>
// CHECK-DAG: %[[IN_F32:.*]] = mfuse.cast %arg0 : (tensor<2x4xi32>) -> tensor<2x4xf32>
// CHECK-DAG: %[[REDUCE:.*]] = mfuse.reduce_sum %[[IN_F32]] {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
// CHECK: %[[DIV:.*]] = mfuse.div %[[REDUCE]], %[[CST]]
// CHECK: %[[OUT:.*]] = mfuse.cast %[[DIV]] : (tensor<2x1xf32>) -> tensor<2x1xf16>
// CHECK: return %[[OUT]]
func.func @test_reduce_mean_decompose_int_input_f16_result(%arg0: tensor<2x4xi32>) -> tensor<2x1xf16> {
  %0 = mfuse.reduce_mean %arg0 {dimensions = [1], keepdim = true} : (tensor<2x4xi32>) -> tensor<2x1xf16>
  return %0 : tensor<2x1xf16>
}

// CHECK-LABEL: func.func @test_reduce_mean_decompose_scalar
// CHECK-NOT: mfuse.reduce_mean
// CHECK-DAG: %[[REDUCE:.*]] = mfuse.reduce_sum %arg0 {dimensions = [], keepdim = false} : (tensor<f32>) -> tensor<f32>
// CHECK: return %[[REDUCE]]
func.func @test_reduce_mean_decompose_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = mfuse.reduce_mean %arg0 {dimensions = [], keepdim = false} : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

func.func @test_reduce_mean_f16_decompose_cast(%arg0: tensor<2x4xf16>) -> tensor<2x1xf16> {
  %0 = mfuse.reduce_mean %arg0 {dimensions = [1], keepdim = true} : (tensor<2x4xf16>) -> tensor<2x1xf16>
  return %0 : tensor<2x1xf16>

  // CHECK-LABEL: func.func @test_reduce_mean_f16_decompose_cast
  // CHECK-NOT: mfuse.reduce_mean
  // CHECK-DAG: %[[CST:.*]] = mfuse.constant dense<4> : tensor<i64, {is_scalar = ""}>
  // CHECK-DAG: %[[IN_F32:.*]] = mfuse.cast %arg0 : (tensor<2x4xf16>) -> tensor<2x4xf32>
  // CHECK-DAG: %[[REDUCE:.*]] = mfuse.reduce_sum %[[IN_F32]] {dimensions = [1], keepdim = true}
  // CHECK-SAME: (tensor<2x4xf32>) -> tensor<2x1xf32>
  // CHECK: %[[DIV:.*]] = mfuse.div %[[REDUCE]], %[[CST]]
  // CHECK-SAME: (tensor<2x1xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<2x1xf32>
  // CHECK: %[[OUT:.*]] = mfuse.cast %[[DIV]] : (tensor<2x1xf32>) -> tensor<2x1xf16>
  // CHECK: return %[[OUT]]
}

func.func @test_reduce_mean_bf16_decompose_cast(%arg0: tensor<2x4xbf16>) -> tensor<2x1xbf16> {
  %0 = mfuse.reduce_mean %arg0 {dimensions = [1], keepdim = true} : (tensor<2x4xbf16>) -> tensor<2x1xbf16>
  return %0 : tensor<2x1xbf16>

  // CHECK-LABEL: func.func @test_reduce_mean_bf16_decompose_cast
  // CHECK-NOT: mfuse.reduce_mean
  // CHECK-DAG: %[[CST:.*]] = mfuse.constant dense<4> : tensor<i64, {is_scalar = ""}>
  // CHECK-DAG: %[[IN_F32:.*]] = mfuse.cast %arg0 : (tensor<2x4xbf16>) -> tensor<2x4xf32>
  // CHECK-DAG: %[[REDUCE:.*]] = mfuse.reduce_sum %[[IN_F32]] {dimensions = [1], keepdim = true}
  // CHECK-SAME: (tensor<2x4xf32>) -> tensor<2x1xf32>
  // CHECK: %[[DIV:.*]] = mfuse.div %[[REDUCE]], %[[CST]]
  // CHECK-SAME: (tensor<2x1xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<2x1xf32>
  // CHECK: %[[OUT:.*]] = mfuse.cast %[[DIV]] : (tensor<2x1xf32>) -> tensor<2x1xbf16>
  // CHECK: return %[[OUT]]
}
