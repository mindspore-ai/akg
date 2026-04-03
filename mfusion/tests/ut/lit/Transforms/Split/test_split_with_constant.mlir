// RUN: mfusion-opt %s --split | FileCheck %s

module {
// Test splitting with arithmetic constant operations
// CHECK-LABEL: func @test_split_with_scalar_constant
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG1:.*]]: tensor<4x4xf32>):
// CHECK: %[[CST:.*]] = mfuse.constant dense<2.000000e+00> : tensor<f32, {is_scalar = ""}>
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG1]], %[[CST]]
// CHECK: mfuse.yield %[[MUL]]
// CHECK: return %[[FUSED]]
func.func @test_split_with_scalar_constant(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg1: tensor<4x4xf32>):
    %cst = mfuse.constant dense<2.000000e+00> : tensor<f32, {is_scalar = ""}>
    %1 = mfuse.mul %arg1, %cst : (tensor<4x4xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
    mfuse.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test split with tensor constant
// CHECK-LABEL: func @test_split_with_tensor_constant
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK: mfuse.fused
// CHECK: ^bb0(%[[ARG1:.*]]: tensor<1x4xf32>, %[[ARG2:.*]]: tensor<4x4xf32>):
// CHECK: mfuse.broadcast_to %[[ARG1]] : (tensor<1x4xf32>) -> tensor<4x4xf32>
// CHECK: mfuse.mul %[[ARG2]],
// CHECK: mfuse.yield
// CHECK: return
func.func @test_split_with_tensor_constant(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %cst = mfuse.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]]> : tensor<1x4xf32>
  %0 = mfuse.fused %cst, %arg0 {fusion_type = "dvm"} : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg1: tensor<1x4xf32>, %arg2: tensor<4x4xf32>):
    %1 = mfuse.broadcast_to %arg1 : (tensor<1x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.mul %arg2, %1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test no split with bool constant
// CHECK-LABEL: func @test_split_with_bool_constant
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[CST:.*]] = mfuse.constant dense<true> : tensor<i1, {is_scalar = ""}>
// CHECK: %[[GT:.*]] = mfuse.gt %arg0, %arg0
// CHECK: %[[SELECT:.*]] = mfuse.select %[[CST]], %arg0, %arg1
// CHECK: return %[[SELECT]]
func.func @test_split_with_bool_constant(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %cst = mfuse.constant dense<true> : tensor<i1, {is_scalar = ""}>
    %1 = mfuse.gt %arg2, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %2 = mfuse.select %cst, %arg2, %arg3 : (tensor<i1, {is_scalar = ""}>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test no split with infinity constant
// CHECK-LABEL: func @test_split_with_infinity_constant
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK: %[[CST:.*]] = mfuse.constant dense<0x7F800000> : tensor<f32, {is_scalar = ""}>
// CHECK: %[[MUL:.*]] = mfuse.mul %arg0, %[[CST]]
// CHECK: return %[[MUL]]
func.func @test_split_with_infinity_constant(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg1: tensor<4x4xf32>):
    %cst = mfuse.constant dense<0x7F800000> : tensor<f32, {is_scalar = ""}>
    %1 = mfuse.mul %arg1, %cst : (tensor<4x4xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
    mfuse.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test split with multiple constants
// CHECK-LABEL: func @test_split_with_multiple_constants
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>):
// CHECK: %[[CST:.*]] = mfuse.constant dense<2.000000e+00> : tensor<f32, {is_scalar = ""}>
// CHECK: %[[CST_0:.*]] = mfuse.constant dense<3.000000e+00> : tensor<f32, {is_scalar = ""}>
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG2]], %[[CST]]
// CHECK: %[[ADD:.*]] = mfuse.add %[[MUL]], %[[CST_0]]
// CHECK: mfuse.yield %[[ADD]]
// CHECK: return %[[FUSED]]
func.func @test_split_with_multiple_constants(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>):
    %cst = mfuse.constant dense<2.000000e+00> : tensor<f32, {is_scalar = ""}>
    %cst_0 = mfuse.constant dense<3.000000e+00> : tensor<f32, {is_scalar = ""}>
    %1 = mfuse.mul %arg2, %cst : (tensor<4x4xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
    %2 = mfuse.add %1, %cst_0 : (tensor<4x4xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}
}
