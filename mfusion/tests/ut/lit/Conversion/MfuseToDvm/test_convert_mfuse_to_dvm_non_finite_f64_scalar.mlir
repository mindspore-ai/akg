// RUN: mfusion-opt %s --convert-mfuse-to-dvm | FileCheck %s

module {
  // CHECK-LABEL: func @main_non_finite_f64_scalar_ne
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[NE:.*]] = dvm.binary_scalar NotEqual %[[LOADA]], 0xFF800000 : tensor<2xf32>, f32 -> tensor<2xi1>
  // CHECK: %[[STORE:.*]] = dvm.store %[[NE]] : tensor<2xi1> -> tensor<2xi1>
  // CHECK: return %[[STORE]] : tensor<2xi1>
  // CHECK-NOT: mfuse.constant
  // CHECK-NOT: mfuse.ne
  func.func @main_non_finite_f64_scalar_ne(%a: tensor<2xf32>) -> tensor<2xi1>
      attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.constant dense<0xFFF0000000000000> : tensor<f64, {is_scalar = ""}>
    %1 = mfuse.ne %a, %0 : (tensor<2xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<2xi1>
    return %1 : tensor<2xi1>
  }
}
