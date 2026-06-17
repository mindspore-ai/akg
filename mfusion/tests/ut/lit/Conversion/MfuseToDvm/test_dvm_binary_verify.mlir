// RUN: mfusion-opt %s --verify-diagnostics

module {
  func.func @invalid_binary_mixed_element_type(%arg0: tensor<2xf32>, %arg1: tensor<2xi32>) {
    // expected-error @+1 {{dvm.binary operands must have the same element type, got 'f32' vs 'i32'}}
    %0 = dvm.binary Add %arg0, %arg1 : tensor<2xf32>, tensor<2xi32> -> tensor<2xf32>
    return
  }
}
