// RUN: not mfusion-opt %s --convert-mfuse-to-dvm 2>&1 | FileCheck %s

module {
  func.func @main_invalid_bool(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    // CHECK: unsupported DVM scalar constant element type: 'i1'
    %0 = mfuse.constant dense<true> : tensor<i1, {is_scalar = ""}>
    %1 = mfuse.select %0, %arg0, %arg1 : (tensor<i1, {is_scalar = ""}>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}
