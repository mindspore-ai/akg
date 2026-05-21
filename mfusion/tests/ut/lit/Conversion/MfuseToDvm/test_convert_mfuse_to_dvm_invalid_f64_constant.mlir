// RUN: not mfusion-opt %s --convert-mfuse-to-dvm 2>&1 | FileCheck %s

module {
  func.func @main_invalid_f64(%arg0: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    // CHECK: cannot convert f64 scalar constant to f32 for DVM
    %0 = mfuse.constant dense<1.000000e+40> : tensor<f64, {is_scalar = ""}>
    %1 = mfuse.mul %arg0, %0 : (tensor<2xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}
