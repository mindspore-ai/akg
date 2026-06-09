// RUN: mfusion-opt %s --verify-diagnostics

module {
  func.func @invalid_rank() {
    // expected-error @+1 {{dvm.constant only supports scalar tensor (rank=0), got rank 1}}
    %0 = dvm.constant dense<[1, 2]> : tensor<2xi32>
    return
  }
}
