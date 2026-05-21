// RUN: mfusion-opt %s --verify-diagnostics

module {
  func.func @invalid_i64() {
    // expected-error @+1 {{dvm.constant unsupported element type: 'i64'}}
    %0 = dvm.constant dense<1> : tensor<i64>
    return
  }
}
