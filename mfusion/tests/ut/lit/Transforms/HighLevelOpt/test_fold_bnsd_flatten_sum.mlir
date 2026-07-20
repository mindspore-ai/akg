// RUN: mfusion-opt %s --pass-pipeline='builtin.module(func.func(mfuse-fold-bnsd-flatten-sum))' | FileCheck %s --check-prefix=GUARDED
// RUN: mfusion-opt %s --pass-pipeline='builtin.module(func.func(mfuse-fold-bnsd-flatten-sum{guard=false}))' | FileCheck %s --check-prefix=UNGUARDED
//
// GUARDED (default) requires a DVM-clusterable ELEMWISE producer on the [B,N,S,D]
// source. UNGUARDED reproduces the legacy unconditional behavior for every
// semantic match (block-arg and opaque-producer sources get rewritten too).

module {
  // ----- Positive (guarded): mfuse.add producer rewrites to direct reduce [0,2].
  // GUARDED-LABEL: func.func @fold_flatten_sum_elemwise_producer
  // GUARDED: %[[SRC:.*]] = mfuse.add %arg0, %arg1
  // GUARDED: %[[DIRECT:.*]] = mfuse.reduce_sum %[[SRC]] {dimensions = [0, 2], keepdim = false} : (tensor<2x3x4x5xf32>) -> tensor<3x5xf32>
  // GUARDED: %[[RESHAPE:.*]] = mfuse.reshape %[[DIRECT]] : (tensor<3x5xf32>) -> tensor<1x15xf32>
  // GUARDED-NOT: torch.aten.clone
  // GUARDED-NOT: mfuse.permute
  // GUARDED: return %[[RESHAPE]]
  func.func @fold_flatten_sum_elemwise_producer(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x4x5xf32>)
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

  // ----- Positive (guarded): mfuse.mul producer, keepdim=false -> rank-1 [N*D] result.
  // GUARDED-LABEL: func.func @fold_flatten_sum_elemwise_producer_keepdim_false
  // GUARDED: %[[SRC:.*]] = mfuse.mul %arg0, %arg1
  // GUARDED: %[[DIRECT:.*]] = mfuse.reduce_sum %[[SRC]] {dimensions = [0, 2], keepdim = false} : (tensor<2x3x4x5xf32>) -> tensor<3x5xf32>
  // GUARDED: %[[RESHAPE:.*]] = mfuse.reshape %[[DIRECT]] : (tensor<3x5xf32>) -> tensor<15xf32>
  // GUARDED-NOT: torch.aten.clone
  // GUARDED-NOT: mfuse.permute
  // GUARDED: return %[[RESHAPE]]
  func.func @fold_flatten_sum_elemwise_producer_keepdim_false(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x4x5xf32>)
      -> tensor<15xf32> {
    %int0 = torch.constant.int 0
    %0 = mfuse.mul %arg0, %arg1 : (tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %1 = mfuse.permute %0, [0, 2, 1, 3] : (tensor<2x3x4x5xf32>) -> tensor<2x4x3x5xf32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<2x4x3x5xf32> to !torch.vtensor<[2,4,3,5],f32>
    %3 = torch.aten.clone %2, %int0 : !torch.vtensor<[2,4,3,5],f32>, !torch.int -> !torch.vtensor<[2,4,3,5],f32>
    %4 = builtin.unrealized_conversion_cast %3 : !torch.vtensor<[2,4,3,5],f32> to tensor<2x4x3x5xf32>
    %5 = mfuse.reshape %4 : (tensor<2x4x3x5xf32>) -> tensor<8x15xf32>
    %6 = mfuse.reduce_sum %5 {dimensions = [0], keepdim = false} : (tensor<8x15xf32>) -> tensor<15xf32>
    return %6 : tensor<15xf32>
  }

  // ----- Positive (guarded): producer result shared with a matmul consumer;
  // only the reduce branch is redirected, the flatten chain for mm is preserved.
  // GUARDED-LABEL: func.func @fold_flatten_sum_elemwise_producer_shared_with_mm
  // GUARDED: %[[SRC:.*]] = mfuse.add %arg0, %arg1
  // GUARDED: %[[PERM:.*]] = mfuse.permute %[[SRC]], [0, 2, 1, 3]
  // GUARDED: %[[CLONE:.*]] = torch.aten.clone
  // GUARDED: %[[FLAT:.*]] = mfuse.reshape {{.*}} : (tensor<2x4x3x5xf32>) -> tensor<8x15xf32>
  // GUARDED: mfuse.aclnn.mm %[[FLAT]], {{.*}}
  // GUARDED: %[[DIRECT:.*]] = mfuse.reduce_sum %[[SRC]] {dimensions = [0, 2], keepdim = false} : (tensor<2x3x4x5xf32>) -> tensor<3x5xf32>
  // GUARDED: %[[RESHAPE:.*]] = mfuse.reshape %[[DIRECT]] : (tensor<3x5xf32>) -> tensor<1x15xf32>
  // GUARDED: return %[[RESHAPE]]
  func.func @fold_flatten_sum_elemwise_producer_shared_with_mm(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x4x5xf32>, %w: tensor<15x7xf32>)
      -> (tensor<1x15xf32>, tensor<8x7xf32>) {
    %int0 = torch.constant.int 0
    %0 = mfuse.add %arg0, %arg1 : (tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %1 = mfuse.permute %0, [0, 2, 1, 3] : (tensor<2x3x4x5xf32>) -> tensor<2x4x3x5xf32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<2x4x3x5xf32> to !torch.vtensor<[2,4,3,5],f32>
    %3 = torch.aten.clone %2, %int0 : !torch.vtensor<[2,4,3,5],f32>, !torch.int -> !torch.vtensor<[2,4,3,5],f32>
    %4 = builtin.unrealized_conversion_cast %3 : !torch.vtensor<[2,4,3,5],f32> to tensor<2x4x3x5xf32>
    %5 = mfuse.reshape %4 : (tensor<2x4x3x5xf32>) -> tensor<8x15xf32>
    %7 = mfuse.aclnn.mm %5, %w : (tensor<8x15xf32>, tensor<15x7xf32>) -> tensor<8x7xf32>
    %8 = mfuse.reduce_sum %5 {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
    return %8, %7 : tensor<1x15xf32>, tensor<8x7xf32>
  }

  // ----- Negative (guarded): source is a func argument (no fusible producer).
  // GUARDED-LABEL: func.func @preserve_no_producer_block_arg
  // GUARDED: mfuse.permute %arg0, [0, 2, 1, 3]
  // GUARDED: mfuse.reduce_sum {{.*}} {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
  // GUARDED-NOT: dimensions = [0, 2]
  //
  // UNGUARDED ignores the missing producer and rewrites the block-argument source.
  // UNGUARDED-LABEL: func.func @preserve_no_producer_block_arg
  // UNGUARDED: %[[DIRECT:.*]] = mfuse.reduce_sum %arg0 {dimensions = [0, 2], keepdim = false} : (tensor<2x3x4x5xf32>) -> tensor<3x5xf32>
  // UNGUARDED: mfuse.reshape %[[DIRECT]] : (tensor<3x5xf32>) -> tensor<1x15xf32>
  func.func @preserve_no_producer_block_arg(%arg0: tensor<2x3x4x5xf32>)
      -> tensor<1x15xf32> {
    %int0 = torch.constant.int 0
    %0 = mfuse.permute %arg0, [0, 2, 1, 3] : (tensor<2x3x4x5xf32>) -> tensor<2x4x3x5xf32>
    %1 = builtin.unrealized_conversion_cast %0 : tensor<2x4x3x5xf32> to !torch.vtensor<[2,4,3,5],f32>
    %2 = torch.aten.clone %1, %int0 : !torch.vtensor<[2,4,3,5],f32>, !torch.int -> !torch.vtensor<[2,4,3,5],f32>
    %3 = builtin.unrealized_conversion_cast %2 : !torch.vtensor<[2,4,3,5],f32> to tensor<2x4x3x5xf32>
    %4 = mfuse.reshape %3 : (tensor<2x4x3x5xf32>) -> tensor<8x15xf32>
    %5 = mfuse.reduce_sum %4 {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
    return %5 : tensor<1x15xf32>
  }

  // ----- Negative (guarded): producer is mfuse.permute (OPAQUE), not ELEMWISE.
  // mfuse.matmul / batch_matmul / grouped_matmul are also OPAQUE per OpRegister.h.
  // GUARDED-LABEL: func.func @preserve_opaque_permute_producer
  // GUARDED: mfuse.permute {{.*}} [0, 1, 2, 3]
  // GUARDED: mfuse.permute {{.*}} [0, 2, 1, 3]
  // GUARDED: mfuse.reduce_sum {{.*}} {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
  // GUARDED-NOT: dimensions = [0, 2]
  //
  // UNGUARDED ignores the producer kind and rewrites anyway.
  // UNGUARDED-LABEL: func.func @preserve_opaque_permute_producer
  // UNGUARDED: mfuse.reduce_sum {{.*}} {dimensions = [0, 2], keepdim = false}
  func.func @preserve_opaque_permute_producer(%arg0: tensor<2x3x4x5xf32>)
      -> tensor<1x15xf32> {
    %int0 = torch.constant.int 0
    %0 = mfuse.permute %arg0, [0, 1, 2, 3] : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %1 = mfuse.permute %0, [0, 2, 1, 3] : (tensor<2x3x4x5xf32>) -> tensor<2x4x3x5xf32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<2x4x3x5xf32> to !torch.vtensor<[2,4,3,5],f32>
    %3 = torch.aten.clone %2, %int0 : !torch.vtensor<[2,4,3,5],f32>, !torch.int -> !torch.vtensor<[2,4,3,5],f32>
    %4 = builtin.unrealized_conversion_cast %3 : !torch.vtensor<[2,4,3,5],f32> to tensor<2x4x3x5xf32>
    %5 = mfuse.reshape %4 : (tensor<2x4x3x5xf32>) -> tensor<8x15xf32>
    %6 = mfuse.reduce_sum %5 {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
    return %6 : tensor<1x15xf32>
  }

  // ----- Negative (both modes): wrong permute keeps the original graph.
  // GUARDED-LABEL: func.func @preserve_wrong_permute
  // UNGUARDED-LABEL: func.func @preserve_wrong_permute
  // GUARDED: mfuse.permute %arg0, [0, 1, 3, 2]
  // GUARDED: mfuse.reduce_sum {{.*}} {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
  // UNGUARDED: mfuse.permute %arg0, [0, 1, 3, 2]
  // UNGUARDED: mfuse.reduce_sum {{.*}} {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
  func.func @preserve_wrong_permute(%arg0: tensor<2x3x4x5xf32>)
      -> tensor<1x15xf32> {
    %int0 = torch.constant.int 0
    %0 = mfuse.permute %arg0, [0, 1, 3, 2] : (tensor<2x3x4x5xf32>) -> tensor<2x3x5x4xf32>
    %1 = builtin.unrealized_conversion_cast %0 : tensor<2x3x5x4xf32> to !torch.vtensor<[2,3,5,4],f32>
    %2 = torch.aten.clone %1, %int0 : !torch.vtensor<[2,3,5,4],f32>, !torch.int -> !torch.vtensor<[2,3,5,4],f32>
    %3 = builtin.unrealized_conversion_cast %2 : !torch.vtensor<[2,3,5,4],f32> to tensor<2x3x5x4xf32>
    %4 = mfuse.reshape %3 : (tensor<2x3x5x4xf32>) -> tensor<8x15xf32>
    %5 = mfuse.reduce_sum %4 {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
    return %5 : tensor<1x15xf32>
  }

  // ----- Negative (both modes): non-contiguous clone (memory_format != 0).
  // GUARDED-LABEL: func.func @preserve_non_contiguous_clone
  // UNGUARDED-LABEL: func.func @preserve_non_contiguous_clone
  // GUARDED: torch.aten.clone
  // GUARDED: mfuse.reduce_sum {{.*}} {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
  // UNGUARDED: torch.aten.clone
  // UNGUARDED: mfuse.reduce_sum {{.*}} {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
  func.func @preserve_non_contiguous_clone(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x4x5xf32>)
      -> tensor<1x15xf32> {
    %int3 = torch.constant.int 3
    %0 = mfuse.add %arg0, %arg1 : (tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %1 = mfuse.permute %0, [0, 2, 1, 3] : (tensor<2x3x4x5xf32>) -> tensor<2x4x3x5xf32>
    %2 = builtin.unrealized_conversion_cast %1 : tensor<2x4x3x5xf32> to !torch.vtensor<[2,4,3,5],f32>
    %3 = torch.aten.clone %2, %int3 : !torch.vtensor<[2,4,3,5],f32>, !torch.int -> !torch.vtensor<[2,4,3,5],f32>
    %4 = builtin.unrealized_conversion_cast %3 : !torch.vtensor<[2,4,3,5],f32> to tensor<2x4x3x5xf32>
    %5 = mfuse.reshape %4 : (tensor<2x4x3x5xf32>) -> tensor<8x15xf32>
    %6 = mfuse.reduce_sum %5 {dimensions = [0], keepdim = true} : (tensor<8x15xf32>) -> tensor<1x15xf32>
    return %6 : tensor<1x15xf32>
  }

  // ----- Negative (both modes): dynamic source shape is not rewritten.
  // GUARDED-LABEL: func.func @preserve_dynamic_shape
  // UNGUARDED-LABEL: func.func @preserve_dynamic_shape
  // GUARDED: mfuse.permute %arg0, [0, 2, 1, 3]
  // GUARDED: mfuse.reduce_sum {{.*}} {dimensions = [0], keepdim = true} : (tensor<?x15xf32>) -> tensor<1x15xf32>
  // UNGUARDED: mfuse.permute %arg0, [0, 2, 1, 3]
  // UNGUARDED: mfuse.reduce_sum {{.*}} {dimensions = [0], keepdim = true} : (tensor<?x15xf32>) -> tensor<1x15xf32>
  func.func @preserve_dynamic_shape(%arg0: tensor<?x3x4x5xf32>)
      -> tensor<1x15xf32> {
    %int0 = torch.constant.int 0
    %0 = mfuse.permute %arg0, [0, 2, 1, 3] : (tensor<?x3x4x5xf32>) -> tensor<?x4x3x5xf32>
    %1 = builtin.unrealized_conversion_cast %0 : tensor<?x4x3x5xf32> to !torch.vtensor<[?,4,3,5],f32>
    %2 = torch.aten.clone %1, %int0 : !torch.vtensor<[?,4,3,5],f32>, !torch.int -> !torch.vtensor<[?,4,3,5],f32>
    %3 = builtin.unrealized_conversion_cast %2 : !torch.vtensor<[?,4,3,5],f32> to tensor<?x4x3x5xf32>
    %4 = mfuse.reshape %3 : (tensor<?x4x3x5xf32>) -> tensor<?x15xf32>
    %5 = mfuse.reduce_sum %4 {dimensions = [0], keepdim = true} : (tensor<?x15xf32>) -> tensor<1x15xf32>
    return %5 : tensor<1x15xf32>
  }
}
