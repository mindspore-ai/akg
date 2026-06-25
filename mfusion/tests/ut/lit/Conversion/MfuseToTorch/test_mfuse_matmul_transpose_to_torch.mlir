// RUN: mfusion-opt %s --convert-mfuse-to-torch="kernel-generator=dvm" --canonicalize 2>&1 | FileCheck %s

// CHECK-NOT: still has transpose flags after cleanup

module {
  // CHECK-LABEL: func.func @main(
  func.func @main(%arg0: tensor<2x4xf32>, %arg1: tensor<8x4xf32>) -> tensor<2x8xf32> {
    %0 = mfuse.aclnn.mm %arg0, %arg1 {trans_x2 = true} : (tensor<2x4xf32>, tensor<8x4xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }

  // CHECK-DAG: %[[MAIN_C1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[MAIN_C0:.*]] = torch.constant.int 0
  // CHECK: %[[MAIN_PERM:.*]] = torch.prim.ListConstruct %[[MAIN_C1]], %[[MAIN_C0]]
  // CHECK: %[[MAIN_RHS:.*]] = torch.aten.permute %arg1, %[[MAIN_PERM]]
  // CHECK: torch.aten.mm %arg0, %[[MAIN_RHS]]
  // CHECK-NOT: dvm_trans

  // CHECK-LABEL: func.func @main_paired_conversion_only(
  func.func @main_paired_conversion_only(%arg0: tensor<4096x16xf32>, %arg1: tensor<16x64xf32>)
      -> tensor<4096x64xf32> {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<4096x16xf32>) -> tensor<16x4096xf32>
    %1 = mfuse.aclnn.mm %0, %arg1 {trans_x1 = true} : (tensor<16x4096xf32>, tensor<16x64xf32>) -> tensor<4096x64xf32>
    return %1 : tensor<4096x64xf32>
  }

  // CHECK: %[[PAIR_FIRST:.*]] = torch.aten.permute %arg0
  // CHECK: %[[PAIR_SECOND:.*]] = torch.aten.permute %[[PAIR_FIRST]]
  // CHECK: torch.aten.mm %[[PAIR_SECOND]], %arg1
  // CHECK-NOT: dvm_trans

  // CHECK-LABEL: func.func @copied_dvm(
  func.func @copied_dvm(%arg0: tensor<2x4xf32>, %arg1: tensor<8x4xf32>)
      -> tensor<2x8xf32> attributes {mfusion.fusion_type = "dvm"} {
    %0 = mfuse.aclnn.mm %arg0, %arg1 {trans_x2 = true} : (tensor<2x4xf32>, tensor<8x4xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }

  // CHECK-NOT: torch.aten.permute
  // CHECK: torch.aten.mm{{.*}}dvm_trans_b = true

  // CHECK-LABEL: func.func private @outlined_dvm(
  func.func private @outlined_dvm(%arg0: tensor<2x4xf32>, %arg1: tensor<8x4xf32>)
      -> tensor<2x8xf32> attributes {mfusion.fusion_type = "dvm", mfusion.outlined} {
    %0 = mfuse.aclnn.mm %arg0, %arg1 {trans_x2 = true} : (tensor<2x4xf32>, tensor<8x4xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }

  // CHECK: %[[OUTLINED_RHS:.*]] = torch.aten.permute %arg1
  // CHECK: torch.aten.mm %arg0, %[[OUTLINED_RHS]]
  // CHECK-NOT: dvm_trans

  // CHECK-LABEL: func.func @copied_akg(
  func.func @copied_akg(%arg0: tensor<2x4xf32>, %arg1: tensor<8x4xf32>)
      -> tensor<2x8xf32> attributes {mfusion.fusion_type = "akg"} {
    %0 = mfuse.aclnn.mm %arg0, %arg1 {trans_x2 = true} : (tensor<2x4xf32>, tensor<8x4xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }

  // CHECK: %[[AKG_RHS:.*]] = torch.aten.permute %arg1
  // CHECK: torch.aten.mm %arg0, %[[AKG_RHS]]
  // CHECK-NOT: dvm_trans
}

// CHECK-NOT: still has transpose flags after cleanup
