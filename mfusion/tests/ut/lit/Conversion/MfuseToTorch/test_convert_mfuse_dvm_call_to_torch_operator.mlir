// RUN: mfusion-opt %s --convert-mfuse-to-torch | FileCheck %s

module {
  // CHECK-LABEL: func @main
  // CHECK-NOT: mfusion.subgraph =
  // CHECK-DAG: %[[SUBGRAPH:.*]] = torch.constant.str "main_mul_fused_0_"
  // CHECK: torch.operator "torch.mfusion.dvm_call__i1_o1"(%{{.*}}, %[[SUBGRAPH]])
  // CHECK: mfusion.is_dynamic = false
  // CHECK: mfusion.subgraph_mlir =
  func.func @main(%a: tensor<2xf32>) -> tensor<2xf32> {
    %0 = mfuse.dvm_call %a {is_dynamic = false, subgraph_mlir = "module { func.func @entry(%arg0: tensor<2xf32>) -> tensor<2xf32> { return %arg0 : tensor<2xf32> } }", subgraph = "main_mul_fused_0_"} : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
