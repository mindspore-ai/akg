// RUN: mfusion-opt %s -decompose="pattern-type=BEFORE_MANUAL_FUSION" | FileCheck %s

module {
  // convert-aclnn-matmul-to-matmul: mfuse.aclnn.mm -> mfuse.matmul (trans flags forwarded)
  // CHECK-LABEL: func @mm_to_matmul_2d
  func.func @mm_to_matmul_2d(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %0 = mfuse.aclnn.mm %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    // After pass: aclnn.mm replaced by matmul
    // CHECK-NOT: mfuse.aclnn.mm
    // CHECK: mfuse.matmul
    return %0 : tensor<2x8xf32>
  }
  // CHECK: return

  // aclnn.mm with trans_x1/trans_x2 -> mfuse.matmul preserves attrs
  // CHECK-LABEL: func @mm_to_matmul_2d_trans
  func.func @mm_to_matmul_2d_trans(%arg0: tensor<8x4xf32>, %arg1: tensor<16x8xf32>) -> tensor<4x16xf32> {
    %0 = mfuse.aclnn.mm %arg0, %arg1 {trans_x1 = true, trans_x2 = true} : (tensor<8x4xf32>, tensor<16x8xf32>) -> tensor<4x16xf32>
    // CHECK-NOT: mfuse.aclnn.mm
    // CHECK: mfuse.matmul{{.*}}trans_x1 = true{{.*}}trans_x2 = true
    return %0 : tensor<4x16xf32>
  }
  // CHECK: return

  // convert-aclnn-matmul-to-matmul: mfuse.aclnn.matmul (batch/ND) -> mfuse.matmul
  // CHECK-LABEL: func @aclnn_matmul_to_matmul_nd
  func.func @aclnn_matmul_to_matmul_nd(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
    %0 = mfuse.aclnn.matmul %arg0, %arg1 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    // After pass: aclnn.matmul replaced by matmul
    // CHECK-NOT: mfuse.aclnn.matmul
    // CHECK: mfuse.matmul
    return %0 : tensor<2x4x16xf32>
  }
  // CHECK: return

  // Mixed: aclnn.mm and aclnn.matmul in one function; return %1 so only second matmul may appear (first can be DCE'd)
  // CHECK-LABEL: func @mm_and_matmul_to_matmul
  func.func @mm_and_matmul_to_matmul(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<2x4x8xf32>, %arg3: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
    %0 = mfuse.aclnn.mm %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = mfuse.aclnn.matmul %arg2, %arg3 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    // CHECK-NOT: mfuse.aclnn.mm
    // CHECK-NOT: mfuse.aclnn.matmul
    // CHECK: mfuse.matmul
    return %1 : tensor<2x4x16xf32>
  }
  // CHECK: return

  // convert-aclnn-matmul-to-matmul: mfuse.aclnn.batch_matmul -> mfuse.matmul (trans_x1=false, trans_x2=false)
  // CHECK-LABEL: func @aclnn_batch_matmul_to_matmul
  func.func @aclnn_batch_matmul_to_matmul(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
    %0 = mfuse.aclnn.batch_matmul %arg0, %arg1 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    // After pass: aclnn.batch_matmul replaced by matmul
    // CHECK-NOT: mfuse.aclnn.batch_matmul
    // CHECK: mfuse.matmul
    return %0 : tensor<2x4x16xf32>
  }
  // CHECK: return

}
