// RUN: mfusion-opt %s --verify-diagnostics

func.func @reject_negative_perm_dim(%arg0: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  // expected-error @+1 {{'mfuse.permute' op perm dimensions must be non-negative, got -1 at index 0}}
  %0 = mfuse.permute %arg0, [-1, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>
}
