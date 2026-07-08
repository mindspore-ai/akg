// RUN: mfusion-opt %s --pass-pipeline='builtin.module(func.func(mfuse-decompose-matmul-with-bias-for-dvm-cluster,canonicalize,mfuse-fold-bnsd-flatten-sum,canonicalize,mfuse-dvm-cluster),canonicalize)' | FileCheck %s
//
// Verifies the pre-cluster fold exposes a producer -> reduce_sum edge that the
// DVM cluster pass then places inside a single mfuse.fused region (a
// pointwise-to-reduce fusion). With guard=true (default) the mfuse.add producer
// satisfies the ELEMWISE profitability guard, so the rewrite fires and
// clustering fuses add + reduce_sum into one fused region. Since the old
// flatten branch is reduce-only in this test, the fold must erase the dead
// permute/clone branch and avoid yielding the producer as an extra fused output.

module {
  // CHECK-LABEL: func.func @cluster_after_fold
  // CHECK-NOT: torch.aten.clone
  // CHECK-NOT: mfuse.permute
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
  // CHECK: ^bb0(%[[A:.*]]: tensor<2x3x4x5xf32>, %[[B:.*]]: tensor<2x3x4x5xf32>):
  // CHECK: %[[ADD:.*]] = mfuse.add %[[A]], %[[B]] : (tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
  // CHECK: %[[RED:.*]] = mfuse.reduce_sum %[[ADD]] {dimensions = [0, 2], keepdim = false} : (tensor<2x3x4x5xf32>) -> tensor<3x5xf32>
  // CHECK: %[[RESHAPE:.*]] = mfuse.reshape %[[RED]] : (tensor<3x5xf32>) -> tensor<1x15xf32>
  // CHECK: mfuse.yield %[[RESHAPE]] : tensor<1x15xf32>
  // CHECK-NOT: torch.aten.clone
  // CHECK-NOT: mfuse.permute
  // CHECK: return %[[FUSED]]
  func.func @cluster_after_fold(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x4x5xf32>)
      -> tensor<1x15xf32> {
    %int0 = torch.constant.int 0
    %0 = mfuse.add %arg0, %arg1 : (tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %1 = mfuse.permute %0, [0, 2, 1, 3] : (tensor<2x3x4x5xf32>) -> tensor<2x4x3x5xf32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<2x4x3x5xf32> to !torch.vtensor<[2,4,3,5],f32>
    %3 = torch.aten.clone %2, %int0 : !torch.vtensor<[2,4,3,5],f32>, !torch.int -> !torch.vtensor<[2,4,3,5],f32>
    %4 = builtin.unrealized_conversion_cast %3 : !torch.vtensor<[2,4,3,5],f32> to tensor<2x4x3x5xf32>
    %5 = mfuse.reshape %4 : (tensor<2x4x3x5xf32>) -> tensor<8x15xf32>
    %6 = mfuse.reduce_sum %5 {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
    return %6 : tensor<1x15xf32>
  }

  // CHECK-LABEL: func.func @cluster_after_matmul_with_bias_fold
  // CHECK-SAME: %[[MATMUL_LHS:.*]]: tensor<2x3x4x7xf32>
  // CHECK-SAME: %[[MATMUL_RHS:.*]]: tensor<7x5xf32>
  // CHECK-SAME: %[[BIAS:.*]]: tensor<5xf32>
  // CHECK-NOT: mfuse.matmul_with_bias
  // CHECK-NOT: torch.aten.clone
  // CHECK-NOT: mfuse.permute
  // CHECK: %[[MM:.*]] = mfuse.matmul %[[MATMUL_LHS]], %[[MATMUL_RHS]]
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[MM]], %[[BIAS]] {fusion_type = "dvm"}
  // CHECK: ^bb0(%[[INNER_MM:.*]]: tensor<2x3x4x5xf32>, %[[INNER_BIAS:.*]]: tensor<5xf32>):
  // CHECK: %[[ADD:.*]] = mfuse.add %[[INNER_MM]], %[[INNER_BIAS]] : (tensor<2x3x4x5xf32>, tensor<5xf32>) -> tensor<2x3x4x5xf32>
  // CHECK: %[[RED:.*]] = mfuse.reduce_sum %[[ADD]] {dimensions = [0, 2], keepdim = false} : (tensor<2x3x4x5xf32>) -> tensor<3x5xf32>
  // CHECK: %[[RESHAPE:.*]] = mfuse.reshape %[[RED]] : (tensor<3x5xf32>) -> tensor<1x15xf32>
  // CHECK: mfuse.yield %[[RESHAPE]] : tensor<1x15xf32>
  // CHECK-NOT: torch.aten.clone
  // CHECK-NOT: mfuse.permute
  // CHECK: return %[[FUSED]]
  func.func @cluster_after_matmul_with_bias_fold(
      %arg0: tensor<2x3x4x7xf32>,
      %arg1: tensor<7x5xf32>,
      %bias: tensor<5xf32>) -> tensor<1x15xf32> {
    %int0 = torch.constant.int 0
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %bias
        : (tensor<2x3x4x7xf32>, tensor<7x5xf32>, tensor<5xf32>) -> tensor<2x3x4x5xf32>
    %1 = mfuse.permute %0, [0, 2, 1, 3] : (tensor<2x3x4x5xf32>) -> tensor<2x4x3x5xf32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<2x4x3x5xf32> to !torch.vtensor<[2,4,3,5],f32>
    %3 = torch.aten.clone %2, %int0 : !torch.vtensor<[2,4,3,5],f32>, !torch.int -> !torch.vtensor<[2,4,3,5],f32>
    %4 = builtin.unrealized_conversion_cast %3 : !torch.vtensor<[2,4,3,5],f32> to tensor<2x4x3x5xf32>
    %5 = mfuse.reshape %4 : (tensor<2x4x3x5xf32>) -> tensor<8x15xf32>
    %6 = mfuse.reduce_sum %5 {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
    return %6 : tensor<1x15xf32>
  }
}
