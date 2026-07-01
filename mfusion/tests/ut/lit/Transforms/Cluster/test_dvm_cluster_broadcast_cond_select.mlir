// RUN: mfusion-opt %s --fuse-safe-softmax-dvm --mfuse-dvm-cluster | FileCheck %s
//
// Broadcast-cond-select behaviour across the safe-softmax pipeline boundary.
//
// Key fact established by this test: fuse-safe-softmax-dvm folds ANY matching
// broadcast-cond-select chain (safe-softmax-tagged OR a generic untagged one)
// into a mfuse.fused DVM island *before* the cluster pass runs. Therefore the
// cluster pass never sees the inner select, and the old selectOpCheck branch
// that rejected untagged broadcast-cond-selects is unreachable for this
// pipeline. The select is fused by the fusion pass, not rejected by cluster.

module {
  // Pipeline INACTIVE (no safe-softmax anchor): a generic broadcast select is
  // NOT folded by fuse-safe-softmax-dvm, so the cluster pass sees it and folds
  // it (with its neg neighbour) into a generic DVM cluster.
  // CHECK-LABEL: func @test_broadcast_select_clusters_with_neighbor
  // CHECK: %[[FUSED:.*]] = mfuse.fused
  // CHECK-SAME: {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x2x1xi1>, tensor<2x2x8xf32>, tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
  // CHECK: ^bb0(
  // CHECK: mfuse.logical_not
  // CHECK: mfuse.select
  // CHECK: mfuse.neg
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  // CHECK-NOT: mfusion.dvm_fuse_kind = "safe_softmax"
  func.func @test_broadcast_select_clusters_with_neighbor(%mask: tensor<2x2x1xi1>, %a: tensor<2x2x8xf32>,
      %b: tensor<2x2x8xf32>) -> tensor<2x2x8xf32> {
    %not = mfuse.logical_not %mask : (tensor<2x2x1xi1>) -> tensor<2x2x1xi1>
    %sel = mfuse.select %not, %a, %b : (tensor<2x2x1xi1>, tensor<2x2x8xf32>, tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    %out = mfuse.neg %sel : (tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    return %out : tensor<2x2x8xf32>
  }

  // Pipeline ACTIVE (safe-softmax anchor present): the same generic broadcast
  // select is folded by fuse-safe-softmax-dvm into a mfuse.fused DVM island
  // (NOT a safe_softmax island), because the fusion pass runs before cluster.
  // CHECK-LABEL: func @test_untagged_broadcast_select_fused_before_cluster
  // The whole chain is folded into one DVM fused island.
  // CHECK: mfuse.fused{{.*}}fusion_type = "dvm"
  // It is NOT a safe_softmax island.
  // CHECK-NOT: mfusion.dvm_fuse_kind = "safe_softmax"
  // Inside the island the original ops are preserved.
  // CHECK: mfuse.select
  // CHECK: mfuse.neg
  //
  // Safe-softmax anchor: activates the pipeline for the module; minimal check.
  // CHECK-LABEL: func @safe_softmax_pipeline_anchor
  // CHECK: mfuse.fused{{.*}}fusion_type = "dvm"
  // CHECK-NOT: torch.aten._softmax
  func.func @test_untagged_broadcast_select_fused_before_cluster(%mask: tensor<2x2x1xi1>, %a: tensor<2x2x8xf32>,
      %b: tensor<2x2x8xf32>) -> tensor<2x2x8xf32> {
    %not = mfuse.logical_not %mask : (tensor<2x2x1xi1>) -> tensor<2x2x1xi1>
    %sel = mfuse.select %not, %a, %b : (tensor<2x2x1xi1>, tensor<2x2x8xf32>, tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    %out = mfuse.neg %sel : (tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    return %out : tensor<2x2x8xf32>
  }

  func.func @safe_softmax_pipeline_anchor(%scores: tensor<2x2x8xf32>, %mask: tensor<2x2x1xi1>)
      -> tensor<2x2x8xf32> {
    %scores_torch = builtin.unrealized_conversion_cast %scores
        : tensor<2x2x8xf32> to !torch.vtensor<[2,2,8],f32>
    %dim = torch.constant.int 2
    %half = torch.constant.bool false
    %softmax_torch = torch.aten._softmax %scores_torch, %dim, %half
        : !torch.vtensor<[2,2,8],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,2,8],f32>
    %softmax = builtin.unrealized_conversion_cast %softmax_torch
        : !torch.vtensor<[2,2,8],f32> to tensor<2x2x8xf32>
    %cst = mfuse.constant dense<0.000000e+00> : tensor<f32, {is_scalar = ""}>
    %zero = mfuse.full %cst : (tensor<f32, {is_scalar = ""}>) -> tensor<2x2x8xf32>
    %not = mfuse.logical_not %mask : (tensor<2x2x1xi1>) -> tensor<2x2x1xi1>
    %sel = mfuse.select %not, %zero, %softmax
        : (tensor<2x2x1xi1>, tensor<2x2x8xf32>, tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    return %sel : tensor<2x2x8xf32>
  }
}
