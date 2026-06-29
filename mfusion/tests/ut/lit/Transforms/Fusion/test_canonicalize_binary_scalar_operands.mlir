// RUN: mfusion-opt %s --mfuse-canonicalize-binary-scalar-operands | FileCheck %s

module {
  // CHECK-LABEL: func.func @canonicalize_lhs_num_to_tensor_sub
  // CHECK: %[[C:.*]] = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
  // CHECK-NOT: mfuse.num_to_tensor
  // CHECK: %[[R:.*]] = mfuse.sub %[[C]], %arg0 : (tensor<i64, {is_scalar = ""}>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: return %[[R]] : tensor<4x4xf32>
  func.func @canonicalize_lhs_num_to_tensor_sub(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c1 = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %n = mfuse.num_to_tensor %c1 : (tensor<i64, {is_scalar = ""}>) -> tensor<i64>
    %0 = mfuse.sub %n, %arg0 : (tensor<i64>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK-LABEL: func.func @canonicalize_rhs_num_to_tensor_div
  // CHECK: %[[C:.*]] = mfuse.constant dense<2.000000e+00> : tensor<f64, {is_scalar = ""}>
  // CHECK-NOT: mfuse.num_to_tensor
  // CHECK: %[[R:.*]] = mfuse.div %arg0, %[[C]] : (tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  // CHECK: return %[[R]] : tensor<4x4xf32>
  func.func @canonicalize_rhs_num_to_tensor_div(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c2 = mfuse.constant dense<2.0> : tensor<f64, {is_scalar = ""}>
    %n = mfuse.num_to_tensor %c2 : (tensor<f64, {is_scalar = ""}>) -> tensor<f64>
    %0 = mfuse.div %arg0, %n : (tensor<4x4xf32>, tensor<f64>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // CHECK-LABEL: func.func @canonicalize_commutative_lhs_num_to_tensor_add
  // CHECK: %[[C:.*]] = mfuse.constant dense<3> : tensor<i64, {is_scalar = ""}>
  // CHECK-NOT: mfuse.num_to_tensor
  // CHECK: %[[R:.*]] = mfuse.add %arg0, %[[C]] : (tensor<4x4xi32>, tensor<i64, {is_scalar = ""}>) -> tensor<4x4xi32>
  // CHECK: return %[[R]] : tensor<4x4xi32>
  func.func @canonicalize_commutative_lhs_num_to_tensor_add(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
    %c3 = mfuse.constant dense<3> : tensor<i64, {is_scalar = ""}>
    %n = mfuse.num_to_tensor %c3 : (tensor<i64, {is_scalar = ""}>) -> tensor<i64>
    %0 = mfuse.add %n, %arg0 : (tensor<i64>, tensor<4x4xi32>) -> tensor<4x4xi32>
    return %0 : tensor<4x4xi32>
  }

  // CHECK-LABEL: func.func @keep_scalar_scalar_unmodified
  // CHECK: mfuse.num_to_tensor
  // CHECK: mfuse.num_to_tensor
  // CHECK: mfuse.add
  func.func @keep_scalar_scalar_unmodified() -> tensor<i64> {
    %c1 = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %c2 = mfuse.constant dense<2> : tensor<i64, {is_scalar = ""}>
    %n1 = mfuse.num_to_tensor %c1 : (tensor<i64, {is_scalar = ""}>) -> tensor<i64>
    %n2 = mfuse.num_to_tensor %c2 : (tensor<i64, {is_scalar = ""}>) -> tensor<i64>
    %0 = mfuse.add %n1, %n2 : (tensor<i64>, tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
  }
}
