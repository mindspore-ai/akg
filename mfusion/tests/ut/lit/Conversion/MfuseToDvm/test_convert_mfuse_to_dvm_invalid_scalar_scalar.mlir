// RUN: not mfusion-opt %s --convert-mfuse-to-dvm 2>&1 | FileCheck %s

module {
  func.func @main_invalid_scalar_scalar() -> tensor<f32, {is_scalar = ""}> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    // CHECK: DVM binary scalar conversion does not support scalar-scalar operands
    %0 = mfuse.constant dense<2.000000e+00> : tensor<f32, {is_scalar = ""}>
    %1 = mfuse.constant dense<3.000000e+00> : tensor<f32, {is_scalar = ""}>
    %2 = mfuse.add %0, %1 : (tensor<f32, {is_scalar = ""}>, tensor<f32, {is_scalar = ""}>) -> tensor<f32, {is_scalar = ""}>
    return %2 : tensor<f32, {is_scalar = ""}>
  }
}
