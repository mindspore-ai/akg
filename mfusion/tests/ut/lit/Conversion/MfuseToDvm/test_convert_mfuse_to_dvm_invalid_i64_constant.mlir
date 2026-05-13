// RUN: not mfusion-opt %s --convert-mfuse-to-dvm 2>&1 | FileCheck %s

module {
  func.func @main_invalid_i64(%arg0: tensor<2xi32>) -> tensor<2xi32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    // CHECK: cannot convert i64 scalar constant to i32 for DVM
    %0 = mfuse.constant dense<2147483648> : tensor<i64, {is_scalar = ""}>
    %1 = mfuse.mul %arg0, %0 : (tensor<2xi32>, tensor<i64, {is_scalar = ""}>) -> tensor<2xi32>
    return %1 : tensor<2xi32>
  }
}
