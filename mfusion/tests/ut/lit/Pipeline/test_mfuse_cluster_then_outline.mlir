// RUN: mfusion-opt %s --mfuse-dvm-cluster --outline-mfuse-fused-subgraphs | FileCheck %s

module {
  // CHECK-LABEL: func.func private @main_fused_0
  // CHECK-DAG: mfusion.fusion_type = "dvm"
  // CHECK-DAG: mfusion.outlined
  // CHECK: mfuse.add
  // CHECK: mfuse.mul

  // CHECK-LABEL: func @main
  // CHECK: %[[CALL:.*]] = call @main_fused_0(%[[A:.*]], %[[B:.*]])
  // CHECK: return %[[CALL]]
  func.func @main(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = mfuse.add %a, %b : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = mfuse.mul %0, %a : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
  // CHECK-NOT: mfuse.fused
}
