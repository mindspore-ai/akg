// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=reduce_mean" | FileCheck %s

func.func @test_reduce_mean_decompose(%arg0: tensor<2x4xf32>) -> tensor<2x1xf32> {
  %0 = mfuse.reduce_mean %arg0 {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>

  // CHECK-LABEL: func.func @test_reduce_mean_decompose
  // CHECK-NOT: mfuse.reduce_mean
  // CHECK-DAG: %[[CST:.*]] = mfuse.constant dense<4.000000e+00>
  // CHECK-DAG: %[[REDUCE:.*]] = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = true}
  // CHECK: %[[DIV:.*]] = mfuse.div %[[REDUCE]], %[[CST]]
  // CHECK: return %[[DIV]]
}

func.func @test_reduce_mean_decompose_int_input_float_result(%arg0: tensor<2x4xi32>) -> tensor<2x1xf32> {
  %0 = mfuse.reduce_mean %arg0 {dimensions = [1], keepdim = true} : (tensor<2x4xi32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>

  // CHECK-LABEL: func.func @test_reduce_mean_decompose_int_input_float_result
  // CHECK-NOT: mfuse.reduce_mean
  // CHECK-DAG: %[[CST:.*]] = mfuse.constant dense<4.000000e+00>
  // CHECK-DAG: %[[REDUCE:.*]] = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = true} : (tensor<2x4xi32>) -> tensor<2x1xf32>
  // CHECK: %[[DIV:.*]] = mfuse.div %[[REDUCE]], %[[CST]]
  // CHECK: return %[[DIV]]
}

func.func @test_reduce_mean_decompose_scalar(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = mfuse.reduce_mean %arg0 {dimensions = [], keepdim = false} : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>

  // CHECK-LABEL: func.func @test_reduce_mean_decompose_scalar
  // CHECK-NOT: mfuse.reduce_mean
  // CHECK-DAG: %[[CST:.*]] = mfuse.constant dense<1.000000e+00> : tensor<f32, {is_scalar = ""}>
  // CHECK-DAG: %[[REDUCE:.*]] = mfuse.reduce_sum %arg0 {dimensions = [], keepdim = false} : (tensor<f32>) -> tensor<f32>
  // CHECK: %[[DIV:.*]] = mfuse.div %[[REDUCE]], %[[CST]]
  // CHECK: return %[[DIV]]
}
