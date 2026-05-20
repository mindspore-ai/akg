// RUN: mfusion-opt %s --fuse-matmul-transpose-weight | FileCheck %s

// Regression: fuse-matmul-transpose-weight creates a new mfuse.permute for
// alignment and must propagate symshape encoding to the permuted dynamic result.

module {
  // CHECK-LABEL: func.func @transpose_weight_dynamic
  // CHECK: %[[SELF_PERM:.*]] = mfuse.permute %arg0, [1, 0] : (tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>) -> tensor<16x?xf16, #mfuse.symshape<["16", "s0"]>>
  // CHECK: mfuse.matmul %[[SELF_PERM]], {{.*}} {trans_x1 = true, trans_x2 = true}
  func.func @transpose_weight_dynamic(%arg0: tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>, %arg1: tensor<32x16xf16>) -> tensor<?x32xf16, #mfuse.symshape<["s0", "32"]>> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>}} {
    %0 = mfuse.permute %arg1, [1, 0] : (tensor<32x16xf16>) -> tensor<16x32xf16>
    %1 = mfuse.matmul %arg0, %0 : (tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>, tensor<16x32xf16>) -> tensor<?x32xf16, #mfuse.symshape<["s0", "32"]>>
    return %1 : tensor<?x32xf16, #mfuse.symshape<["s0", "32"]>>
  }
}
