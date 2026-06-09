// RUN: mfusion-opt %s --verify-diagnostics

module {
  func.func @invalid_scalar_f64(%arg0: tensor<2xf32>) {
    // expected-error @+1 {{dvm.binary_scalar unsupported scalar type: 'f64'}}
    %0 = dvm.binary_scalar Add %arg0, 1.000000e+00 : tensor<2xf32>, f64 -> tensor<2xf32>
    return
  }
}
