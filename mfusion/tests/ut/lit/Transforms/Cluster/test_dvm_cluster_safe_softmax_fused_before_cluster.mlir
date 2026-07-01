// RUN: mfusion-opt %s --fuse-safe-softmax-dvm --mfuse-dvm-cluster | FileCheck %s
//
// Behavioural guard for selectOpCheck in DVMCluster.cc.
//
// Observed (bc34abd build) and confirmed below: fuse-safe-softmax-dvm folds
// the whole safe-softmax candidate — including the broadcast-cond-select —
// into a single mfuse.fused DVM island. The select therefore lives INSIDE the
// fusedop body and is invisible to DVMCluster's base-op whitelist, so the
// cluster sees the fusedop as a black box and does not reopen it.
//
// For the residual untagged case: when the safe-softmax pipeline is active,
// fuse-safe-softmax-dvm folds EVERY matching broadcast-cond-select chain (not
// only safe-softmax-tagged ones) into a mfuse.fused DVM island *before* the
// cluster pass runs. So the cluster never sees the inner select, and the old
// selectOpCheck branch that rejected untagged broadcast-cond-selects from the
// generic DVM cluster is unreachable for this pipeline — those selects are
// fused by the fusion pass, not rejected by the cluster pass.
//
// This test pins that behaviour:
//  * the safe-softmax select is folded into a mfuse.fused (tagged member) and
//    the cluster does not touch the inside;
//  * an untagged generic broadcast-cond-select is ALSO folded into a
//    mfuse.fused (DVM island, but without the safe_softmax kind) by the fusion
//    pass — it is not left standalone.

module {
  // Safe-softmax candidate: fused _softmax + broadcast select with zero branch.
  // Observed: fuse folds the whole candidate into a mfuse.fused DVM island tagged on
  // the wrapper. The cluster sees the fusedop as a black box and does not reopen it.
  // CHECK-LABEL: func @test_safe_softmax_select_keeps_tag_after_cluster
  // CHECK: mfuse.fused{{.*}}fusion_type = "dvm"
  // CHECK-SAME: mfusion.dvm_fuse_kind = "safe_softmax"
  // CHECK: mfuse.select
  // CHECK: mfuse.yield
  // CHECK: return %0
  func.func @test_safe_softmax_select_keeps_tag_after_cluster(%scores: tensor<2x2x8xf32>,
      %mask: tensor<2x2x1xi1>) -> tensor<2x2x8xf32> {
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

  // Untagged broadcast-cond-select must still be kept out of the generic DVM
  // cluster when the safe-softmax pipeline is active. In reality
  // fuse-safe-softmax-dvm folds ANY matching broadcast-cond-select chain
  // (this generic one included) into a mfuse.fused DVM island *before* the
  // cluster pass runs, so the cluster never sees the inner select. The select
  // ends up fused, not rejected/standalone.
  // CHECK-LABEL: func @test_generic_select_not_clustered_when_active
  // The whole chain is folded into one DVM fused island.
  // CHECK: mfuse.fused{{.*}}fusion_type = "dvm"
  // It is NOT a safe_softmax island.
  // CHECK-NOT: mfusion.dvm_fuse_kind = "safe_softmax"
  // CHECK: mfuse.select
  // CHECK: mfuse.neg
  func.func @test_generic_select_not_clustered_when_active(%mask: tensor<2x2x1xi1>,
      %a: tensor<2x2x8xf32>, %b: tensor<2x2x8xf32>) -> tensor<2x2x8xf32> {
    %not = mfuse.logical_not %mask : (tensor<2x2x1xi1>) -> tensor<2x2x1xi1>
    %sel = mfuse.select %not, %a, %b : (tensor<2x2x1xi1>, tensor<2x2x8xf32>, tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    %out = mfuse.neg %sel : (tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    return %out : tensor<2x2x8xf32>
  }
}
