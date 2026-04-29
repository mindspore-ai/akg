// RUN: mfusion-opt %s --convert-torch-to-mfuse | FileCheck %s

// CHECK-LABEL: func.func @unsupported_splat_literal_ui16
// CHECK: torch.vtensor.literal(dense<1> : tensor<2xui16>) : !torch.vtensor<[2],ui16>
func.func @unsupported_splat_literal_ui16() -> !torch.vtensor<[2],ui16> {
  %0 = torch.vtensor.literal(dense<1> : tensor<2xui16>) : !torch.vtensor<[2],ui16>
  return %0 : !torch.vtensor<[2],ui16>
}
