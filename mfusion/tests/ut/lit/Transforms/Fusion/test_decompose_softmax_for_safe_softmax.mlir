// RUN: mfusion-opt %s --fuse-safe-softmax-dvm | FileCheck %s

module {
  // Safe-softmax semantic graph: fused _softmax + broadcast select with zero branch.
  // CHECK-LABEL: func @test_decompose_safe_softmax
  // CHECK: %[[FUSED:.*]] = mfuse.fused{{.*}}fusion_type = "dvm"
  // CHECK-SAME: mfusion.dvm_fuse_kind = "safe_softmax"
  // CHECK-NOT: torch.aten._softmax
  // CHECK-NOT: mfuse.softmax
  // CHECK: ^bb0(
  // CHECK: mfuse.reduce_max
  // CHECK: mfuse.sub
  // CHECK: mfuse.exp
  // CHECK: mfuse.reduce_sum
  // CHECK: mfuse.div
  // CHECK: mfuse.logical_not
  // CHECK: mfuse.select
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  // CHECK-NOT: mfuse.fused
  func.func @test_decompose_safe_softmax(%scores: tensor<2x2x8xf32>, %mask: tensor<2x2x1xi1>)
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

  // mfuse.softmax path (post convert-torch-to-mfuse).
  // CHECK-LABEL: func @test_fuse_mfuse_softmax_safe_softmax
  // CHECK: mfuse.fused
  // CHECK-NOT: mfuse.softmax
  // CHECK: mfuse.reduce_max
  func.func @test_fuse_mfuse_softmax_safe_softmax(%scores: tensor<2x2x8xf32>, %mask: tensor<2x2x1xi1>)
      -> tensor<2x2x8xf32> {
    %softmax = mfuse.softmax %scores {dim = 2 : i64, half_to_float = false}
        : (tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    %cst = mfuse.constant dense<0.000000e+00> : tensor<f32, {is_scalar = ""}>
    %zero = mfuse.full %cst : (tensor<f32, {is_scalar = ""}>) -> tensor<2x2x8xf32>
    %not = mfuse.logical_not %mask : (tensor<2x2x1xi1>) -> tensor<2x2x1xi1>
    %sel = mfuse.select %not, %zero, %softmax
        : (tensor<2x2x1xi1>, tensor<2x2x8xf32>, tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    return %sel : tensor<2x2x8xf32>
  }

  // Albert-style 4D attention scores: mfuse.softmax may carry dim = -1 from torch.
  // CHECK-LABEL: func @test_fuse_mfuse_softmax_negative_dim
  // CHECK: mfuse.fused
  // CHECK-NOT: mfuse.softmax
  // CHECK: mfuse.reduce_max
  // CHECK-SAME: dimensions = [3]
  func.func @test_fuse_mfuse_softmax_negative_dim(%scores: tensor<4x12x512x512xf32>, %mask: tensor<4x12x512x1xi1>)
      -> tensor<4x12x512x512xf32> {
    %softmax = mfuse.softmax %scores {dim = -1 : i64, half_to_float = false}
        : (tensor<4x12x512x512xf32>) -> tensor<4x12x512x512xf32>
    %cst = mfuse.constant dense<0.000000e+00> : tensor<f32, {is_scalar = ""}>
    %zero = mfuse.full %cst : (tensor<f32, {is_scalar = ""}>) -> tensor<4x12x512x512xf32>
    %not = mfuse.logical_not %mask : (tensor<4x12x512x1xi1>) -> tensor<4x12x512x1xi1>
    %sel = mfuse.select %not, %zero, %softmax
        : (tensor<4x12x512x1xi1>, tensor<4x12x512x512xf32>, tensor<4x12x512x512xf32>) -> tensor<4x12x512x512xf32>
    return %sel : tensor<4x12x512x512xf32>
  }

  // Standalone _softmax must stay fused (BERT-style path without post-softmax zero select).
  // CHECK-LABEL: func @test_keep_standalone_softmax
  // CHECK: torch.aten._softmax
  // CHECK-NOT: mfuse.reduce_max
  // CHECK-NOT: mfuse.fused
  func.func @test_keep_standalone_softmax(%scores: !torch.vtensor<[2,2,8],f32>) -> !torch.vtensor<[2,2,8],f32> {
    %dim = torch.constant.int 2
    %half = torch.constant.bool false
    %softmax = torch.aten._softmax %scores, %dim, %half
        : !torch.vtensor<[2,2,8],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,2,8],f32>
    return %softmax : !torch.vtensor<[2,2,8],f32>
  }

  // Standalone mfuse.softmax must be preserved for BERT after torch conversion.
  // CHECK-LABEL: func @test_keep_standalone_mfuse_softmax
  // CHECK: mfuse.softmax
  // CHECK-NOT: mfuse.reduce_max
  // CHECK-NOT: mfuse.fused
  func.func @test_keep_standalone_mfuse_softmax(%scores: tensor<2x2x8xf32>) -> tensor<2x2x8xf32> {
    %softmax = mfuse.softmax %scores {dim = 2 : i64, half_to_float = false}
        : (tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    return %softmax : tensor<2x2x8xf32>
  }

  // Pre-softmax mask with large negative constant (not zero) must not trigger decompose.
  // CHECK-LABEL: func @test_skip_pre_softmax_mask
  // CHECK: torch.aten._softmax
  // CHECK-NOT: mfuse.reduce_max
  // CHECK-NOT: mfuse.fused
  func.func @test_skip_pre_softmax_mask(%scores: tensor<2x2x8xf32>, %mask: tensor<2x2x1xi1>)
      -> tensor<2x2x8xf32> {
    %neg = mfuse.constant dense<-1.000000e+09> : tensor<f32, {is_scalar = ""}>
    %neg_full = mfuse.full %neg : (tensor<f32, {is_scalar = ""}>) -> tensor<2x2x8xf32>
    %masked = mfuse.select %mask, %neg_full, %scores
        : (tensor<2x2x1xi1>, tensor<2x2x8xf32>, tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    %masked_torch = builtin.unrealized_conversion_cast %masked
        : tensor<2x2x8xf32> to !torch.vtensor<[2,2,8],f32>
    %dim = torch.constant.int 2
    %half = torch.constant.bool false
    %softmax_torch = torch.aten._softmax %masked_torch, %dim, %half
        : !torch.vtensor<[2,2,8],f32>, !torch.int, !torch.bool -> !torch.vtensor<[2,2,8],f32>
    %softmax = builtin.unrealized_conversion_cast %softmax_torch
        : !torch.vtensor<[2,2,8],f32> to tensor<2x2x8xf32>
    return %softmax : tensor<2x2x8xf32>
  }

  // Decomposed softmax chain: verify tag lives on the mfuse.fused wrapper only.
  // CHECK-LABEL: func @test_decomposed_chain_tags
  // CHECK: %[[FUSED:.*]] = mfuse.fused{{.*}}fusion_type = "dvm"
  // CHECK-SAME: mfusion.dvm_fuse_kind = "safe_softmax"
  // CHECK: ^bb0(
  // CHECK: mfuse.sub
  // CHECK: mfuse.exp
  // CHECK: mfuse.reduce_sum
  // CHECK: mfuse.div
  // CHECK: mfuse.full
  // CHECK: mfuse.logical_not
  // CHECK: mfuse.select
  // CHECK: mfuse.reshape
  // CHECK: return %[[FUSED]]
  // CHECK-NOT: mfusion.dvm_fuse_kind = "broadcast_cond_select"
  func.func @test_decomposed_chain_tags(%scores: tensor<2x2x8xf32>, %amax: tensor<2x2x1xf32>,
      %mask: tensor<2x2x1xi1>) -> tensor<4x8xf32> {
    %sub = mfuse.sub %scores, %amax : (tensor<2x2x8xf32>, tensor<2x2x1xf32>) -> tensor<2x2x8xf32>
    %exp = mfuse.exp %sub : (tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    %sum = mfuse.reduce_sum %exp {dimensions = [2], keepdim = true} : (tensor<2x2x8xf32>) -> tensor<2x2x1xf32>
    %softmax = mfuse.div %exp, %sum : (tensor<2x2x8xf32>, tensor<2x2x1xf32>) -> tensor<2x2x8xf32>
    %cst = mfuse.constant dense<0.000000e+00> : tensor<f32, {is_scalar = ""}>
    %zero = mfuse.full %cst : (tensor<f32, {is_scalar = ""}>) -> tensor<2x2x8xf32>
    %not = mfuse.logical_not %mask : (tensor<2x2x1xi1>) -> tensor<2x2x1xi1>
    %sel = mfuse.select %not, %zero, %softmax : (tensor<2x2x1xi1>, tensor<2x2x8xf32>, tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    %out = mfuse.reshape %sel : (tensor<2x2x8xf32>) -> tensor<4x8xf32>
    return %out : tensor<4x8xf32>
  }

  // Generic broadcast select without safe-softmax semantics must not fuse.
  // CHECK-LABEL: func @test_broadcast_cond_select_no_tag
  // CHECK-NOT: mfusion.dvm_fuse_kind = "safe_softmax"
  // CHECK-NOT: mfuse.fused
  func.func @test_broadcast_cond_select_no_tag(%mask: tensor<2x2x1xi1>, %a: tensor<2x2x8xf32>,
      %b: tensor<2x2x8xf32>) -> tensor<2x2x8xf32> {
    %not = mfuse.logical_not %mask : (tensor<2x2x1xi1>) -> tensor<2x2x1xi1>
    %sel = mfuse.select %not, %a, %b : (tensor<2x2x1xi1>, tensor<2x2x8xf32>, tensor<2x2x8xf32>) -> tensor<2x2x8xf32>
    return %sel : tensor<2x2x8xf32>
  }
}
