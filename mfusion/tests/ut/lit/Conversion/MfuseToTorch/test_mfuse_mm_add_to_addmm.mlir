// RUN: mfusion-opt %s --convert-mfuse-to-torch="kernel-generator=dvm" --canonicalize | FileCheck %s
// Post-recompose path: legal 2D aclnn.mm+add / matmul_with_bias → aten.addmm;
// illegal / ND / DVM-copied keep mm+add.

module {
  // Primary inductor path after recompose: aclnn.mm + 1D bias.
  // CHECK-LABEL: func.func @aclnn_mm_bias_to_addmm
  // CHECK: torch.aten.addmm
  // CHECK-NOT: torch.aten.add.Tensor
  func.func @aclnn_mm_bias_to_addmm(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<8xf32>)
      -> tensor<2x8xf32> {
    %0 = mfuse.aclnn.mm %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.add %0, %arg2 : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @matmul_with_bias_to_addmm
  // CHECK: torch.aten.addmm
  // CHECK-NOT: torch.aten.add.Tensor
  func.func @matmul_with_bias_to_addmm(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<8xf32>)
      -> tensor<2x8xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2
        : (tensor<2x4xf32>, tensor<4x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @illegal_bias_keeps_mm_add
  // CHECK-NOT: torch.aten.addmm
  // CHECK: torch.aten.mm
  // CHECK: torch.aten.add.Tensor
  func.func @illegal_bias_keeps_mm_add(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4xf32>)
      -> tensor<2x8xf32> {
    %0 = mfuse.aclnn.mm %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.add %0, %arg2 : (tensor<2x8xf32>, tensor<4xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @dtype_mismatch_keeps_mm_add
  // CHECK-NOT: torch.aten.addmm
  // CHECK: torch.aten.mm
  // CHECK: torch.aten.add.Tensor
  func.func @dtype_mismatch_keeps_mm_add(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<8xbf16>)
      -> tensor<2x8xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2
        : (tensor<2x4xf32>, tensor<4x8xf32>, tensor<8xbf16>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @aclnn_mm_row_bias_to_addmm
  // CHECK: torch.aten.addmm
  // CHECK-NOT: torch.aten.add.Tensor
  func.func @aclnn_mm_row_bias_to_addmm(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<1x8xf32>)
      -> tensor<2x8xf32> {
    %0 = mfuse.aclnn.mm %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.add %0, %arg2 : (tensor<2x8xf32>, tensor<1x8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @matmul_with_bias_trans_to_addmm
  // CHECK: torch.aten.addmm
  // CHECK-NOT: torch.aten.add.Tensor
  func.func @matmul_with_bias_trans_to_addmm(%arg0: tensor<4x2xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<8xf32>)
      -> tensor<2x8xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2 {trans_x1 = true}
        : (tensor<4x2xf32>, tensor<4x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }

  // CHECK-LABEL: func.func @matmul_with_bias_nd_keeps_mm_add
  // CHECK-NOT: torch.aten.addmm
  // CHECK: torch.aten.matmul
  // CHECK: torch.aten.add.Tensor
  func.func @matmul_with_bias_nd_keeps_mm_add(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>,
                                               %arg2: tensor<16xf32>) -> tensor<2x4x16xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2
        : (tensor<2x4x8xf32>, tensor<2x8x16xf32>, tensor<16xf32>) -> tensor<2x4x16xf32>
    return %0 : tensor<2x4x16xf32>
  }

  // CHECK-LABEL: func.func @dvm_copied_keeps_mm_add
  // CHECK-NOT: torch.aten.addmm
  // CHECK: torch.aten.mm
  // CHECK: torch.aten.add.Tensor
  func.func @dvm_copied_keeps_mm_add(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<8xf32>)
      -> tensor<2x8xf32> attributes {mfusion.fusion_type = "dvm"} {
    %0 = mfuse.aclnn.mm %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.add %0, %arg2 : (tensor<2x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    return %1 : tensor<2x8xf32>
  }
}
