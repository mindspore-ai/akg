// RUN: mfusion-opt %s --outline-mfuse-fused-subgraphs | FileCheck %s

module {
  // CHECK-LABEL: func.func private @main_fused_1
  // CHECK-DAG: mfusion.fusion_type = "akg"
  // CHECK-DAG: mfusion.outlined
  // CHECK: mfuse.add

  // CHECK-LABEL: func.func private @main_fused_0
  // CHECK-DAG: mfusion.fusion_type = "dvm"
  // CHECK-DAG: mfusion.outlined
  // CHECK: mfuse.mul

  // CHECK-LABEL: func @main
  // CHECK: %[[CALL0:.*]] = call @main_fused_0(%[[A:.*]], %[[B:.*]])
  // CHECK: %[[CALL1:.*]] = call @main_fused_1(%[[CALL0]], %[[C:.*]])
  // CHECK: return %[[CALL1]]
  func.func @main(%a: tensor<2xf32>, %b: tensor<2xf32>, %c: tensor<2xf32>) -> tensor<2xf32> {
    %0 = mfuse.fused %a, %b {fusion_type = "dvm"} :
        (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32> {
      ^bb0(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>):
        %1 = mfuse.mul %arg0, %arg1 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        mfuse.yield %1 : tensor<2xf32>
    }
    %2 = mfuse.fused %0, %c {fusion_type = "akg"} :
        (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32> {
      ^bb0(%arg2: tensor<2xf32>, %arg3: tensor<2xf32>):
        %3 = mfuse.add %arg2, %arg3 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        mfuse.yield %3 : tensor<2xf32>
    }
    return %2 : tensor<2xf32>
  }
  // CHECK-NOT: mfuse.fused
}
