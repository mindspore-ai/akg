// RUN: not mfusion-opt %s --convert-mfuse-to-dvm 2>&1 | FileCheck %s

module {
  func.func @main_invalid_full_bool() -> tensor<2xi1> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    // CHECK: unsupported DVM broadcast scalar constant element type: 'i1'
    %0 = mfuse.constant dense<true> : tensor<i1, {is_scalar = ""}>
    %1 = mfuse.full %0 : (tensor<i1, {is_scalar = ""}>) -> tensor<2xi1>
    return %1 : tensor<2xi1>
  }
}
