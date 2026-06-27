// RUN: mfusion-opt %s --mfuse-fold-redundant-matmul-transpose --convert-mfuse-to-torch="kernel-generator=dvm" --reconcile-unrealized-casts --canonicalize | FileCheck %s

module {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<4096x16xf32>, %arg1: tensor<16x64xf32>) -> tensor<4096x64xf32> {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<4096x16xf32>) -> tensor<16x4096xf32>
    %1 = mfuse.aclnn.mm %0, %arg1 {trans_x1 = true} : (tensor<16x4096xf32>, tensor<16x64xf32>) -> tensor<4096x64xf32>
    return %1 : tensor<4096x64xf32>
  }

  // CHECK-NOT: torch.aten.permute
  // CHECK: torch.aten.mm %arg0, %arg1
  // CHECK-NOT: dvm_trans

  // CHECK-LABEL: func.func private @copied_dvm
  func.func private @copied_dvm(%arg0: tensor<4096x16xf32>, %arg1: tensor<16x64xf32>)
      -> tensor<4096x64xf32> attributes {mfusion.fusion_type = "dvm"} {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<4096x16xf32>) -> tensor<16x4096xf32>
    %1 = mfuse.aclnn.mm %0, %arg1 {trans_x1 = true} : (tensor<16x4096xf32>, tensor<16x64xf32>) -> tensor<4096x64xf32>
    return %1 : tensor<4096x64xf32>
  }

  // CHECK: torch.aten.permute %arg0
  // CHECK: torch.aten.mm{{.*}}dvm_trans_a = true
}
