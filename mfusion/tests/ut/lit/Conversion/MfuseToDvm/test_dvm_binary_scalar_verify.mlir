// RUN: mfusion-opt %s --verify-diagnostics

module {
  func.func @invalid_scalar_i1(%arg0: tensor<2xf32>) {
    // expected-error @+1 {{dvm.binary_scalar unsupported scalar type: 'i1'}}
    %0 = dvm.binary_scalar Add %arg0, 1 : tensor<2xf32>, i1 -> tensor<2xf32>
    return
  }
}
