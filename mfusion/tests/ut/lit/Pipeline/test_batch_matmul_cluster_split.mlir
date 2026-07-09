// RUN: mfusion-opt %s --fuse-batch-matmul --mfuse-dvm-cluster --split --canonicalize | FileCheck %s

module {
  // CHECK-LABEL: func @test_add_permute_broadcast_matmul_cluster_split
  // CHECK-NOT: mfuse.permute
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg1, %arg2 {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<640x640xbf16>, tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
  // CHECK: ^bb0(%[[ARG3:.*]]: tensor<640x640xbf16>, %[[ARG4:.*]]: tensor<640x640xbf16>):
  // CHECK: %[[ADD:.*]] = mfuse.add %[[ARG3]], %[[ARG4]]
  // CHECK: %[[BCAST:.*]] = mfuse.broadcast_to %[[ADD]]
  // CHECK: mfuse.yield %[[BCAST]]
  // CHECK: %[[MATMUL:.*]] = mfuse.matmul %arg0, %[[FUSED]] {trans_x2 = true}
  // CHECK: return %[[MATMUL]]
  func.func @test_add_permute_broadcast_matmul_cluster_split(
      %arg0: tensor<1x4096x640xbf16>, %arg1: tensor<640x640xbf16>, %arg2: tensor<640x640xbf16>)
      -> tensor<1x4096x640xbf16> {
    %0 = mfuse.add %arg1, %arg2 : (tensor<640x640xbf16>, tensor<640x640xbf16>) -> tensor<640x640xbf16>
    %1 = mfuse.permute %0, [1, 0] : (tensor<640x640xbf16>) -> tensor<640x640xbf16>
    %2 = mfuse.broadcast_to %1 : (tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
    %3 = mfuse.matmul %arg0, %2 : (tensor<1x4096x640xbf16>, tensor<1x640x640xbf16>) -> tensor<1x4096x640xbf16>
    return %3 : tensor<1x4096x640xbf16>
  }
}
