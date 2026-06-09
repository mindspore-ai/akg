// RUN: mfusion-opt %s --verify-diagnostics

module {
  func.func @valid_constants() {
    %0 = dvm.constant dense<1.000000e+00> : tensor<f32>
    %1 = dvm.constant dense<1> : tensor<i32>
    %2 = dvm.constant dense<1.000000e+00> : tensor<f16>
    %3 = dvm.constant dense<1.000000e+00> : tensor<bf16>
    return
  }

  func.func @invalid_f64() {
    // expected-error @+1 {{dvm.constant unsupported element type: 'f64'}}
    %0 = dvm.constant dense<1.000000e+00> : tensor<f64>
    return
  }
}
