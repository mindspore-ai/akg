// RUN: mfusion-opt %s --fuse-matmul-transpose-weight | FileCheck %s

module {
  // When inner axis (last dim * elem_size) is not 512-byte aligned, pass inserts permute and sets trans.
  // 128 * 4 = 512 -> aligned for f32. Use non-aligned shape to trigger: e.g. 100 * 4 = 400.
  // CHECK-LABEL: func @transpose_weight_one_input
  func.func @transpose_weight_one_input(%arg0: tensor<2x100xf32>, %arg1: tensor<100x8xf32>) -> tensor<2x8xf32> {
    %0 = muse.matmul %arg0, %arg1 : (tensor<2x100xf32>, tensor<100x8xf32>) -> tensor<2x8xf32>
    // Both inputs are unaligned (100*4=400, 8*4=32), so permute both
    // CHECK: muse.permute
    // CHECK: muse.permute
    // CHECK: muse.matmul {{.*}} {trans_x1 = true, trans_x2 = true}
    return %0 : tensor<2x8xf32>
  }

  // CHECK-LABEL: func @transpose_weight_both_inputs
  func.func @transpose_weight_both_inputs(%arg0: tensor<2x100xf32>, %arg1: tensor<100x100xf32>) -> tensor<2x100xf32> {
    %0 = muse.matmul %arg0, %arg1 : (tensor<2x100xf32>, tensor<100x100xf32>) -> tensor<2x100xf32>
    // If both unaligned: permute on both
    // CHECK: muse.permute
    // CHECK: muse.permute
    // CHECK: muse.matmul
    return %0 : tensor<2x100xf32>
  }

  // matmul_with_bias with unaligned inner axis -> permute + matmul_with_bias
  // CHECK-LABEL: func @transpose_weight_mm_with_bias
  func.func @transpose_weight_mm_with_bias(%arg0: tensor<2x100xf32>, %arg1: tensor<100x8xf32>, %arg2: tensor<8xf32>) -> tensor<2x8xf32> {
    %0 = muse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<2x100xf32>, tensor<100x8xf32>, tensor<8xf32>) -> tensor<2x8xf32>
    // CHECK: muse.permute
    // CHECK: muse.permute
    // CHECK: muse.matmul_with_bias {{.*}} {trans_x1 = true, trans_x2 = true}
    return %0 : tensor<2x8xf32>
  }
}
