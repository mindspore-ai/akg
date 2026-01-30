// RUN: mfusion-opt %s --fuse-matmul-cast | FileCheck %s

module {
  // CHECK-LABEL: func @matmul_cast_fuse
  // CHECK-SAME: (%[[A:.*]]: tensor<2x4xf16>, %[[B:.*]]: tensor<4x8xf16>)
  func.func @matmul_cast_fuse(%arg0: tensor<2x4xf16>, %arg1: tensor<4x8xf16>) -> tensor<2x8xf32> {
    %0 = muse.matmul %arg0, %arg1 : (tensor<2x4xf16>, tensor<4x8xf16>) -> tensor<2x8xf16>
    %1 = muse.cast %0 {dtype = f32} : (tensor<2x8xf16>) -> tensor<2x8xf32>
    // After fusion: matmul + cast replaced by single matmul with f32 result
    // CHECK-NOT: muse.cast
    // CHECK: %[[R:.*]] = muse.matmul %[[A]], %[[B]] : (tensor<2x4xf16>, tensor<4x8xf16>) -> tensor<2x8xf32>
    // CHECK: return %[[R]]
    return %1 : tensor<2x8xf32>
  }

  // CHECK-LABEL: func @mm_cast_fuse
  func.func @mm_cast_fuse(%arg0: tensor<2x4xf16>, %arg1: tensor<4x8xf16>) -> tensor<2x8xf32> {
    %0 = muse.matmul %arg0, %arg1 : (tensor<2x4xf16>, tensor<4x8xf16>) -> tensor<2x8xf16>
    %1 = muse.cast %0 {dtype = f32} : (tensor<2x8xf16>) -> tensor<2x8xf32>
    // CHECK-NOT: muse.cast
    // CHECK: muse.matmul
    return %1 : tensor<2x8xf32>
  }

  // CHECK-LABEL: func @mm_with_bias_cast_fuse
  func.func @mm_with_bias_cast_fuse(%arg0: tensor<2x4xf16>, %arg1: tensor<4x8xf16>, %arg2: tensor<8xf16>) -> tensor<2x8xf32> {
    %0 = muse.matmul_with_bias %arg0, %arg1, %arg2 : (tensor<2x4xf16>, tensor<4x8xf16>, tensor<8xf16>) -> tensor<2x8xf16>
    %1 = muse.cast %0 {dtype = f32} : (tensor<2x8xf16>) -> tensor<2x8xf32>
    // CHECK-NOT: muse.cast
    // CHECK: muse.matmul_with_bias
    return %1 : tensor<2x8xf32>
  }
}
