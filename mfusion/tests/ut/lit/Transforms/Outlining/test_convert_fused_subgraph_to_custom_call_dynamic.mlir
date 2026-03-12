// RUN: mfusion-opt %s --convert-fused-subgraph-to-custom-call | FileCheck %s

module {
  func.func @main_mul_fused_0(%a: tensor<2x?xf32>, %b: tensor<2x?xf32>) -> tensor<2x?xf32>
      attributes {mfusion.outlined, mfusion.fusion_type = "dvm", mfusion.copied_subgraph = "main_mul_fused_0_"} {
    %0 = dvm.load %a : tensor<2x?xf32> -> tensor<2x?xf32>
    %1 = dvm.load %b : tensor<2x?xf32> -> tensor<2x?xf32>
    %2 = dvm.binary Mul %0, %1 : tensor<2x?xf32>, tensor<2x?xf32> -> tensor<2x?xf32>
    %3 = dvm.store %2 : tensor<2x?xf32> -> tensor<2x?xf32>
    return %3 : tensor<2x?xf32>
  }

  // CHECK-LABEL: func @main
  // CHECK: mfuse.dvm_call
  // CHECK-SAME: is_dynamic = true
  // CHECK-SAME: subgraph = "main_mul_fused_0_"
  // CHECK: subgraph_mlir =
  func.func @main(%a: tensor<2x?xf32>, %b: tensor<2x?xf32>) -> tensor<2x?xf32> {
    %0 = call @main_mul_fused_0(%a, %b) : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
    return %0 : tensor<2x?xf32>
  }

  // CHECK-NOT: func @main_mul_fused_0
}
