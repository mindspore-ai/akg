// RUN: mfusion-opt %s -convert-aclnn-matmul-to-matmul | FileCheck %s

module {
  // convert-aclnn-matmul-to-matmul: muse.aclnn.mm -> muse.matmul (trans_x1=false, trans_x2=false)
  // CHECK-LABEL: func @mm_to_matmul_2d
  func.func @mm_to_matmul_2d(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>) -> tensor<2x8xf32> {
    %0 = muse.aclnn.mm %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    // After pass: aclnn.mm replaced by matmul
    // CHECK-NOT: muse.aclnn.mm
    // CHECK: muse.matmul
    return %0 : tensor<2x8xf32>
  }
  // CHECK: return

  // convert-aclnn-matmul-to-matmul: muse.aclnn.matmul (batch/ND) -> muse.matmul
  // CHECK-LABEL: func @aclnn_matmul_to_matmul_nd
  func.func @aclnn_matmul_to_matmul_nd(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
    %0 = muse.aclnn.matmul %arg0, %arg1 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    // After pass: aclnn.matmul replaced by matmul
    // CHECK-NOT: muse.aclnn.matmul
    // CHECK: muse.matmul
    return %0 : tensor<2x4x16xf32>
  }
  // CHECK: return

  // Mixed: aclnn.mm and aclnn.matmul in one function; return %1 so only second matmul may appear (first can be DCE'd)
  // CHECK-LABEL: func @mm_and_matmul_to_matmul
  func.func @mm_and_matmul_to_matmul(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<2x4x8xf32>, %arg3: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
    %0 = muse.aclnn.mm %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %1 = muse.aclnn.matmul %arg2, %arg3 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    // CHECK-NOT: muse.aclnn.mm
    // CHECK-NOT: muse.aclnn.matmul
    // CHECK: muse.matmul
    return %1 : tensor<2x4x16xf32>
  }
  // CHECK: return

  // aclnn.mm + aclnn.add(bias, alpha=1) -> muse.matmul_with_bias
  // CHECK-LABEL: func @mm_add_to_matmul_with_bias
  func.func @mm_add_to_matmul_with_bias(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = muse.aclnn.mm %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    %cst = arith.constant dense<1.000000e+00> : tensor<f32>
    %1 = muse.aclnn.add %0, %arg2, %cst : (tensor<2x8xf32>, tensor<8xf32>, tensor<f32>) -> tensor<2x8xf32>
    // After pass: mm+add fused to matmul_with_bias
    // CHECK-NOT: muse.aclnn.mm
    // CHECK-NOT: muse.aclnn.add
    // CHECK: muse.matmul_with_bias
    return %1 : tensor<2x8xf32>
  }

  // aclnn.matmul (ND) + aclnn.add(bias, alpha=1) -> muse.matmul_with_bias
  // CHECK-LABEL: func @aclnn_matmul_add_to_matmul_with_bias
  func.func @aclnn_matmul_add_to_matmul_with_bias(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>, %arg2: tensor<16xf32>) -> tensor<2x4x16xf32> {
    %0 = muse.aclnn.matmul %arg0, %arg1 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    %cst = arith.constant dense<1.000000e+00> : tensor<f32>
    %1 = muse.aclnn.add %0, %arg2, %cst : (tensor<2x4x16xf32>, tensor<16xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    // CHECK-NOT: muse.aclnn.matmul
    // CHECK-NOT: muse.aclnn.add
    // CHECK: muse.matmul_with_bias
    return %1 : tensor<2x4x16xf32>
  }

  // convert-aclnn-matmul-to-matmul: muse.aclnn.batch_matmul -> muse.matmul (trans_x1=false, trans_x2=false)
  // CHECK-LABEL: func @aclnn_batch_matmul_to_matmul
  func.func @aclnn_batch_matmul_to_matmul(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> {
    %0 = muse.aclnn.batch_matmul %arg0, %arg1 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    // After pass: aclnn.batch_matmul replaced by matmul
    // CHECK-NOT: muse.aclnn.batch_matmul
    // CHECK: muse.matmul
    return %0 : tensor<2x4x16xf32>
  }
  // CHECK: return

  // aclnn.batch_matmul + aclnn.add(bias, alpha=1) -> muse.matmul_with_bias
  // CHECK-LABEL: func @aclnn_batch_matmul_add_to_matmul_with_bias
  func.func @aclnn_batch_matmul_add_to_matmul_with_bias(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x8x16xf32>, %arg2: tensor<16xf32>) -> tensor<2x4x16xf32> {
    %0 = muse.aclnn.batch_matmul %arg0, %arg1 : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    %cst = arith.constant dense<1.000000e+00> : tensor<f32>
    %1 = muse.aclnn.add %0, %arg2, %cst : (tensor<2x4x16xf32>, tensor<16xf32>, tensor<f32>) -> tensor<2x4x16xf32>
    // CHECK-NOT: muse.aclnn.batch_matmul
    // CHECK-NOT: muse.aclnn.add
    // CHECK: muse.matmul_with_bias
    return %1 : tensor<2x4x16xf32>
  }
}
