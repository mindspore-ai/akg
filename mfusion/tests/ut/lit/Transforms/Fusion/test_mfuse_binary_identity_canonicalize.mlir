// RUN: mfusion-opt %s --canonicalize | FileCheck %s

module {
  // CHECK-LABEL: func.func @do_not_fold_float_add_zero_rhs
  // CHECK: %[[ADD:.*]] = mfuse.add
  // CHECK: return %[[ADD]]
  func.func @do_not_fold_float_add_zero_rhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %zero = mfuse.constant dense<0.0> : tensor<f32, {is_scalar = ""}>
    %0 = mfuse.add %arg0, %zero : (tensor<4x4xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK-LABEL: func.func @do_not_fold_float_add_zero_lhs
  // CHECK: %[[ADD:.*]] = mfuse.add
  // CHECK: return %[[ADD]]
  func.func @do_not_fold_float_add_zero_lhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %zero = mfuse.constant dense<0.0> : tensor<f64, {is_scalar = ""}>
    %0 = mfuse.add %zero, %arg0 : (tensor<f64, {is_scalar = ""}>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK-LABEL: func.func @do_not_fold_float_sub_zero_rhs
  // CHECK: %[[SUB:.*]] = mfuse.sub
  // CHECK: return %[[SUB]]
  func.func @do_not_fold_float_sub_zero_rhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %zero = mfuse.constant dense<0.0> : tensor<f32, {is_scalar = ""}>
    %0 = mfuse.sub %arg0, %zero : (tensor<4x4xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK-LABEL: func.func @fold_int_add_zero_rhs
  // CHECK-NOT: mfuse.add
  // CHECK: return %arg0
  func.func @fold_int_add_zero_rhs(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    %zero = mfuse.constant dense<0> : tensor<i32, {is_scalar = ""}>
    %0 = mfuse.add %arg0, %zero : (tensor<4x4xi32>, tensor<i32, {is_scalar = ""}>) -> tensor<4x4xi32>
    return %0 : tensor<4x4xi32>
  }

  // CHECK-LABEL: func.func @fold_int_add_zero_lhs
  // CHECK-NOT: mfuse.add
  // CHECK: return %arg0
  func.func @fold_int_add_zero_lhs(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    %zero = mfuse.constant dense<0> : tensor<i64, {is_scalar = ""}>
    %0 = mfuse.add %zero, %arg0 : (tensor<i64, {is_scalar = ""}>, tensor<4x4xi32>) -> tensor<4x4xi32>
    return %0 : tensor<4x4xi32>
  }

  // CHECK-LABEL: func.func @fold_int_sub_zero_rhs
  // CHECK-NOT: mfuse.sub
  // CHECK: return %arg0
  func.func @fold_int_sub_zero_rhs(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    %zero = mfuse.constant dense<0> : tensor<i32, {is_scalar = ""}>
    %0 = mfuse.sub %arg0, %zero : (tensor<4x4xi32>, tensor<i32, {is_scalar = ""}>) -> tensor<4x4xi32>
    return %0 : tensor<4x4xi32>
  }

  // CHECK-LABEL: func.func @fold_mul_one_rhs
  // CHECK-NOT: mfuse.mul
  // CHECK: return %arg0
  func.func @fold_mul_one_rhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %one = mfuse.constant dense<1.0> : tensor<f32, {is_scalar = ""}>
    %0 = mfuse.mul %arg0, %one : (tensor<4x4xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK-LABEL: func.func @fold_mul_one_lhs
  // CHECK-NOT: mfuse.mul
  // CHECK: return %arg0
  func.func @fold_mul_one_lhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %one = mfuse.constant dense<1.0> : tensor<f32, {is_scalar = ""}>
    %0 = mfuse.mul %one, %arg0 : (tensor<f32, {is_scalar = ""}>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK-LABEL: func.func @fold_div_one_rhs
  // CHECK-NOT: mfuse.div
  // CHECK: return %arg0
  func.func @fold_div_one_rhs(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %one = mfuse.constant dense<1.0> : tensor<f32, {is_scalar = ""}>
    %0 = mfuse.div %arg0, %one : (tensor<4x4xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK-LABEL: func.func @do_not_fold_add_broadcast_zero
  // CHECK: %[[ADD:.*]] = mfuse.add
  // CHECK: return %[[ADD]]
  func.func @do_not_fold_add_broadcast_zero(%arg0: tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32> {
    %zero = mfuse.constant dense<0.0> : tensor<4x4xf32>
    %0 = mfuse.add %zero, %arg0 : (tensor<4x4xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
