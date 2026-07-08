// RUN: mfusion-opt %s --convert-mfuse-to-dvm | FileCheck %s

module {
  // CHECK-LABEL: func @main_lhs_i64_scalar_sub_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[SUB:.*]] = dvm.binary_scalar Sub 1, %[[LOADA]] : i32, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[SUB]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.constant
  // CHECK-NOT: mfuse.sub
  func.func @main_lhs_i64_scalar_sub_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %c1 = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %0 = mfuse.sub %c1, %a : (tensor<i64, {is_scalar = ""}>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_lhs_f64_scalar_div_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[DIV:.*]] = dvm.binary_scalar Div 1.000000e+00, %[[LOADA]] : f32, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[DIV]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.constant
  // CHECK-NOT: mfuse.div
  func.func @main_lhs_f64_scalar_div_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %c1 = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
    %0 = mfuse.div %c1, %a : (tensor<f64, {is_scalar = ""}>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
