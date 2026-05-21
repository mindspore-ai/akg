// RUN: mfusion-opt %s --verify-diagnostics

module {
  func.func @invalid_bool() {
    // expected-error @+1 {{dvm.constant unsupported element type: 'i1'}}
    %0 = dvm.constant dense<true> : tensor<i1>
    return
  }
}
