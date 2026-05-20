// RUN: mfusion-opt %s --copy-fused-subgraphs | FileCheck %s

module {
  // CHECK-LABEL: func @main_mul_fused_0
  // CHECK: mfusion.copied_subgraph = "main_mul_fused_0_"
  // CHECK: mfusion.outlined
  func.func @main_mul_fused_0(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined} {
    %0 = mfuse.mul %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_mul_fused_0_
  // CHECK-NOT: mfusion.outlined
  // CHECK: mfuse.mul
}
