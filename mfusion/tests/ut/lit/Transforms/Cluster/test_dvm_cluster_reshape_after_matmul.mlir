// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
  // Regression for DVMCluster::reshapeOpCheck: must inspect the *producer* of the reshape input
  // (a matmul-class op), not compare the reshape op's name to mfuse.matmul / batch_matmul / grouped_matmul.
  // Before the fix, mfuse.reshape was never clusterable here; 3D mfuse.matmul -> reshape -> add would not fuse.
  //
  // CHECK-LABEL: func @reshape_after_bmm_form_matmul_fuses_in_dvm_cluster
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
  // CHECK-SAME: {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x128x256xf16>, tensor<2x256x512xf16>) -> tensor<131072xf16>
  // CHECK: ^bb0(%[[A:.*]]: tensor<2x128x256xf16>, %[[B:.*]]: tensor<2x256x512xf16>):
  // CHECK: %[[BMM:.*]] = mfuse.matmul %[[A]], %[[B]]
  // CHECK: %[[RS:.*]] = mfuse.reshape %[[BMM]]
  // CHECK: %[[ADD:.*]] = mfuse.add %[[RS]], %[[RS]]
  // CHECK: mfuse.yield %[[ADD]]
  // CHECK: return %[[FUSED]]
  func.func @reshape_after_bmm_form_matmul_fuses_in_dvm_cluster(
      %arg0: tensor<2x128x256xf16>,
      %arg1: tensor<2x256x512xf16>) -> tensor<131072xf16> {
    %0 = mfuse.matmul %arg0, %arg1
        : (tensor<2x128x256xf16>, tensor<2x256x512xf16>) -> tensor<2x128x512xf16>
    %1 = mfuse.reshape %0 : (tensor<2x128x512xf16>) -> tensor<131072xf16>
    %2 = mfuse.add %1, %1 : (tensor<131072xf16>, tensor<131072xf16>) -> tensor<131072xf16>
    return %2 : tensor<131072xf16>
  }
}
