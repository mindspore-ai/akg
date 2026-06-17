// RUN: mfusion-opt %s --split | FileCheck %s

module {
  // Suffix keepdim reductions followed by scalar div should stay in one dvm
  // fused region so DVM can see the mean-style reduce neighborhood.
  // CHECK-LABEL: func.func @test_reduce_bwd_suffix_dims_div_scalar
  // CHECK-SAME: %[[ARG0:.*]]: tensor<1x16x7x7xf32>
  // CHECK: %[[CST:.*]] = mfuse.constant dense<4.900000e+01> : tensor<f64, {is_scalar = ""}>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[ARG0]], %[[CST]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<1x16x7x7xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<1x16x1x1xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<1x16x7x7xf32>, %[[IN1:.*]]: tensor<f64, {is_scalar = ""}>):
  // CHECK: %[[ABS:.*]] = mfuse.abs %[[IN0]] : (tensor<1x16x7x7xf32>) -> tensor<1x16x7x7xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[ABS]] {dimensions = [3, 2], keepdim = true} : (tensor<1x16x7x7xf32>) -> tensor<1x16x1x1xf32>
  // CHECK: %[[DIV:.*]] = mfuse.div %[[REDUCE]], %[[IN1]] : (tensor<1x16x1x1xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<1x16x1x1xf32>
  // CHECK: mfuse.yield %[[DIV]] : tensor<1x16x1x1xf32>
  // CHECK: return %[[FUSED]] : tensor<1x16x1x1xf32>
  func.func @test_reduce_bwd_suffix_dims_div_scalar(%arg0: tensor<1x16x7x7xf32>) -> tensor<1x16x1x1xf32> {
    %c49 = mfuse.constant dense<4.900000e+01> : tensor<f64, {is_scalar = ""}>
    %0 = mfuse.fused %arg0, %c49 {fusion_type = "dvm"} : (tensor<1x16x7x7xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<1x16x1x1xf32> {
    ^bb0(%arg1: tensor<1x16x7x7xf32>, %arg2: tensor<f64, {is_scalar = ""}>):
      %1 = mfuse.abs %arg1 : (tensor<1x16x7x7xf32>) -> tensor<1x16x7x7xf32>
      %2 = mfuse.reduce_sum %1 {dimensions = [3, 2], keepdim = true} : (tensor<1x16x7x7xf32>) -> tensor<1x16x1x1xf32>
      %3 = mfuse.div %2, %arg2 : (tensor<1x16x1x1xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<1x16x1x1xf32>
      mfuse.yield %3 : tensor<1x16x1x1xf32>
    }
    return %0 : tensor<1x16x1x1xf32>
  }

  // Broadcast-style post-reduce chains should stay fused with the reduce.
  // CHECK-LABEL: func.func @test_reduce_bwd_broadcast_div_add_chain
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>, %[[ARG1:.*]]: tensor<2x4xf32>
  // CHECK: %[[CST:.*]] = mfuse.constant dense<2.000000e+00> : tensor<f64, {is_scalar = ""}>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[ARG0]], %[[ARG1]], %[[CST]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<2x4xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<2x4xf32>, %[[IN1:.*]]: tensor<2x4xf32>, %[[IN2:.*]]: tensor<f64, {is_scalar = ""}>):
  // CHECK: %[[ABS:.*]] = mfuse.abs %[[IN0]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[ABS]] {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  // CHECK: %[[BCAST:.*]] = mfuse.broadcast_to %[[REDUCE]] : (tensor<2x1xf32>) -> tensor<2x4xf32>
  // CHECK: %[[DIV:.*]] = mfuse.div %[[BCAST]], %[[IN2]] : (tensor<2x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<2x4xf32>
  // CHECK: %[[ADD:.*]] = mfuse.add %[[DIV]], %[[IN1]] : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: mfuse.yield %[[ADD]] : tensor<2x4xf32>
  // CHECK: return %[[FUSED]] : tensor<2x4xf32>
  func.func @test_reduce_bwd_broadcast_div_add_chain(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c2 = mfuse.constant dense<2.000000e+00> : tensor<f64, {is_scalar = ""}>
    %0 = mfuse.fused %arg0, %arg1, %c2 {fusion_type = "dvm"} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<2x4xf32> {
    ^bb0(%arg2: tensor<2x4xf32>, %arg3: tensor<2x4xf32>, %arg4: tensor<f64, {is_scalar = ""}>):
      %1 = mfuse.abs %arg2 : (tensor<2x4xf32>) -> tensor<2x4xf32>
      %2 = mfuse.reduce_sum %1 {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
      %3 = mfuse.broadcast_to %2 : (tensor<2x1xf32>) -> tensor<2x4xf32>
      %4 = mfuse.div %3, %arg4 : (tensor<2x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<2x4xf32>
      %5 = mfuse.add %4, %arg3 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      mfuse.yield %5 : tensor<2x4xf32>
    }
    return %0 : tensor<2x4xf32>
  }

  // LayerNorm-style neighborhoods with post-reduce broadcast and multiple
  // outputs should stay fused so mfusion can hand them to DVM spec codegen.
  // CHECK-LABEL: func.func @test_reduce_bwd_layernorm_style_multi_output
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>, %[[ARG1:.*]]: tensor<2x1xf32>
  // CHECK: %[[CST:.*]] = mfuse.constant dense<4.000000e+00> : tensor<f64, {is_scalar = ""}>
  // CHECK: %[[FUSED:.*]]:2 = mfuse.fused %[[ARG0]], %[[ARG1]], %[[CST]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x4xf32>, tensor<2x1xf32>, tensor<f64, {is_scalar = ""}>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<2x4xf32>, %[[IN1:.*]]: tensor<2x1xf32>, %[[IN2:.*]]: tensor<f64, {is_scalar = ""}>):
  // CHECK: %[[ABS:.*]] = mfuse.abs %[[IN0]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[ABS]] {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  // CHECK: %[[DIV1:.*]] = mfuse.div %[[REDUCE]], %[[IN2]] : (tensor<2x1xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<2x1xf32>
  // CHECK: %[[BCAST1:.*]] = mfuse.broadcast_to %[[DIV1]] : (tensor<2x1xf32>) -> tensor<2x4xf32>
  // CHECK: %[[SUB:.*]] = mfuse.sub %[[IN0]], %[[BCAST1]] : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[BCAST2:.*]] = mfuse.broadcast_to %[[IN1]] : (tensor<2x1xf32>) -> tensor<2x4xf32>
  // CHECK: %[[DIV2:.*]] = mfuse.div %[[SUB]], %[[BCAST2]] : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: mfuse.yield %[[SUB]], %[[DIV2]] : tensor<2x4xf32>, tensor<2x4xf32>
  // CHECK: return %[[FUSED]]#0, %[[FUSED]]#1 : tensor<2x4xf32>, tensor<2x4xf32>
  func.func @test_reduce_bwd_layernorm_style_multi_output(%arg0: tensor<2x4xf32>, %arg1: tensor<2x1xf32>)
      -> (tensor<2x4xf32>, tensor<2x4xf32>) {
    %c4 = mfuse.constant dense<4.000000e+00> : tensor<f64, {is_scalar = ""}>
    %0:2 = mfuse.fused %arg0, %arg1, %c4 {fusion_type = "dvm"} : (tensor<2x4xf32>, tensor<2x1xf32>, tensor<f64, {is_scalar = ""}>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
    ^bb0(%arg2: tensor<2x4xf32>, %arg3: tensor<2x1xf32>, %arg4: tensor<f64, {is_scalar = ""}>):
      %1 = mfuse.abs %arg2 : (tensor<2x4xf32>) -> tensor<2x4xf32>
      %2 = mfuse.reduce_sum %1 {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
      %3 = mfuse.div %2, %arg4 : (tensor<2x1xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<2x1xf32>
      %4 = mfuse.broadcast_to %3 : (tensor<2x1xf32>) -> tensor<2x4xf32>
      %5 = mfuse.sub %arg2, %4 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %6 = mfuse.broadcast_to %arg3 : (tensor<2x1xf32>) -> tensor<2x4xf32>
      %7 = mfuse.div %5, %6 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      mfuse.yield %5, %7 : tensor<2x4xf32>, tensor<2x4xf32>
    }
    return %0#0, %0#1 : tensor<2x4xf32>, tensor<2x4xf32>
  }

  // Non-suffix keepdim reductions can also sink into post-reduce consumers.
  // CHECK-LABEL: func.func @test_reduce_bwd_non_contiguous_dims
  // CHECK-SAME: %[[ARG0:.*]]: tensor<4x32x7x7xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[ARG0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<4x32x7x7xf32>) -> tensor<1x32x1x1xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<4x32x7x7xf32>):
  // CHECK: %[[ABS:.*]] = mfuse.abs %[[IN0]] : (tensor<4x32x7x7xf32>) -> tensor<4x32x7x7xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[ABS]] {dimensions = [0, 2, 3], keepdim = true} : (tensor<4x32x7x7xf32>) -> tensor<1x32x1x1xf32>
  // CHECK: %[[SQRT:.*]] = mfuse.sqrt %[[REDUCE]] : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
  // CHECK: mfuse.yield %[[SQRT]] : tensor<1x32x1x1xf32>
  // CHECK: return %[[FUSED]] : tensor<1x32x1x1xf32>
  func.func @test_reduce_bwd_non_contiguous_dims(%arg0: tensor<4x32x7x7xf32>) -> tensor<1x32x1x1xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4x32x7x7xf32>) -> tensor<1x32x1x1xf32> {
    ^bb0(%arg1: tensor<4x32x7x7xf32>):
      %1 = mfuse.abs %arg1 : (tensor<4x32x7x7xf32>) -> tensor<4x32x7x7xf32>
      %2 = mfuse.reduce_sum %1 {dimensions = [0, 2, 3], keepdim = true} : (tensor<4x32x7x7xf32>) -> tensor<1x32x1x1xf32>
      %3 = mfuse.sqrt %2 : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
      mfuse.yield %3 : tensor<1x32x1x1xf32>
    }
    return %0 : tensor<1x32x1x1xf32>
  }

  // A terminal reshape user alone should not be fused by reduce_bwd.
  // CHECK-LABEL: func.func @test_reduce_bwd_terminal_reshape_user
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>
  // CHECK: %[[FUSED0:.*]] = mfuse.fused %[[ARG0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x4xf32>) -> tensor<2x1xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<2x4xf32>):
  // CHECK: %[[ABS:.*]] = mfuse.abs %[[IN0]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[ABS]] {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  // CHECK: mfuse.yield %[[REDUCE]] : tensor<2x1xf32>
  // CHECK: %[[RESHAPE:.*]] = mfuse.reshape %[[FUSED0]] : (tensor<2x1xf32>) -> tensor<2xf32>
  // CHECK: return %[[RESHAPE]] : tensor<2xf32>
  func.func @test_reduce_bwd_terminal_reshape_user(%arg0: tensor<2x4xf32>) -> tensor<2xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<2x4xf32>) -> tensor<2xf32> {
    ^bb0(%arg1: tensor<2x4xf32>):
      %1 = mfuse.abs %arg1 : (tensor<2x4xf32>) -> tensor<2x4xf32>
      %2 = mfuse.reduce_sum %1 {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
      %3 = mfuse.reshape %2 : (tensor<2x1xf32>) -> tensor<2xf32>
      mfuse.yield %3 : tensor<2xf32>
    }
    return %0 : tensor<2xf32>
  }

  // Reshape followed by more post-reduce computation should still fuse.
  // CHECK-LABEL: func.func @test_reduce_bwd_reshape_with_post_op
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[ARG0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x4xf32>) -> tensor<2xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<2x4xf32>):
  // CHECK: %[[ABS:.*]] = mfuse.abs %[[IN0]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[ABS]] {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  // CHECK: %[[RESHAPE:.*]] = mfuse.reshape %[[REDUCE]] : (tensor<2x1xf32>) -> tensor<2xf32>
  // CHECK: %[[SQRT:.*]] = mfuse.sqrt %[[RESHAPE]] : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: mfuse.yield %[[SQRT]] : tensor<2xf32>
  // CHECK: return %[[FUSED]] : tensor<2xf32>
  func.func @test_reduce_bwd_reshape_with_post_op(%arg0: tensor<2x4xf32>) -> tensor<2xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<2x4xf32>) -> tensor<2xf32> {
    ^bb0(%arg1: tensor<2x4xf32>):
      %1 = mfuse.abs %arg1 : (tensor<2x4xf32>) -> tensor<2x4xf32>
      %2 = mfuse.reduce_sum %1 {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
      %3 = mfuse.reshape %2 : (tensor<2x1xf32>) -> tensor<2xf32>
      %4 = mfuse.sqrt %3 : (tensor<2xf32>) -> tensor<2xf32>
      mfuse.yield %4 : tensor<2xf32>
    }
    return %0 : tensor<2xf32>
  }

  // keepdim=false reductions are not spec-friendly and must stay split.
  // CHECK-LABEL: func.func @test_reduce_bwd_keepdim_false
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>
  // CHECK: %[[FUSED0:.*]] = mfuse.fused %[[ARG0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x4xf32>) -> tensor<2xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<2x4xf32>):
  // CHECK: %[[ABS:.*]] = mfuse.abs %[[IN0]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[ABS]] {dimensions = [1], keepdim = false} : (tensor<2x4xf32>) -> tensor<2xf32>
  // CHECK: mfuse.yield %[[REDUCE]] : tensor<2xf32>
  // CHECK: %[[SQRT:.*]] = mfuse.sqrt %[[FUSED0]] : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[SQRT]] : tensor<2xf32>
  func.func @test_reduce_bwd_keepdim_false(%arg0: tensor<2x4xf32>) -> tensor<2xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<2x4xf32>) -> tensor<2xf32> {
    ^bb0(%arg1: tensor<2x4xf32>):
      %1 = mfuse.abs %arg1 : (tensor<2x4xf32>) -> tensor<2x4xf32>
      %2 = mfuse.reduce_sum %1 {dimensions = [1], keepdim = false} : (tensor<2x4xf32>) -> tensor<2xf32>
      %3 = mfuse.sqrt %2 : (tensor<2xf32>) -> tensor<2xf32>
      mfuse.yield %3 : tensor<2xf32>
    }
    return %0 : tensor<2xf32>
  }

  // A reduce area with multiple post-reduce user areas must not sink into any
  // one of them.
  // CHECK-LABEL: func.func @test_reduce_bwd_multi_user_reduce_output
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>, %[[ARG1:.*]]: tensor<2x1xf32>
  // CHECK: %[[FUSED0:.*]] = mfuse.fused %[[ARG0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x4xf32>) -> tensor<2x1xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<2x4xf32>):
  // CHECK: %[[ABS:.*]] = mfuse.abs %[[IN0]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[ABS]] {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  // CHECK: mfuse.yield %[[REDUCE]] : tensor<2x1xf32>
  // CHECK: %[[SQRT:.*]] = mfuse.sqrt %[[FUSED0]] : (tensor<2x1xf32>) -> tensor<2x1xf32>
  // CHECK: %[[MUL:.*]] = mfuse.mul %[[FUSED0]], %[[ARG1]] : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
  // CHECK: return %[[SQRT]], %[[MUL]] : tensor<2x1xf32>, tensor<2x1xf32>
  func.func @test_reduce_bwd_multi_user_reduce_output(%arg0: tensor<2x4xf32>, %arg1: tensor<2x1xf32>)
      -> (tensor<2x1xf32>, tensor<2x1xf32>) {
    %0:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<2x4xf32>, tensor<2x1xf32>) -> (tensor<2x1xf32>, tensor<2x1xf32>) {
    ^bb0(%arg2: tensor<2x4xf32>, %arg3: tensor<2x1xf32>):
      %1 = mfuse.abs %arg2 : (tensor<2x4xf32>) -> tensor<2x4xf32>
      %2 = mfuse.reduce_sum %1 {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
      %3 = mfuse.sqrt %2 : (tensor<2x1xf32>) -> tensor<2x1xf32>
      %4 = mfuse.mul %2, %arg3 : (tensor<2x1xf32>, tensor<2x1xf32>) -> tensor<2x1xf32>
      mfuse.yield %3, %4 : tensor<2x1xf32>, tensor<2x1xf32>
    }
    return %0#0, %0#1 : tensor<2x1xf32>, tensor<2x1xf32>
  }

  // Post-reduce areas with more than three outputs should stay separate.
  // CHECK-LABEL: func.func @test_reduce_bwd_post_reduce_too_many_outputs
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>
  // CHECK: %[[FUSED0:.*]] = mfuse.fused %[[ARG0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x4xf32>) -> tensor<2x1xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<2x4xf32>):
  // CHECK: %[[ABS:.*]] = mfuse.abs %[[IN0]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[ABS]] {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
  // CHECK: mfuse.yield %[[REDUCE]] : tensor<2x1xf32>
  // CHECK: %[[FUSED1:.*]]:4 = mfuse.fused %[[FUSED0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x1xf32>) -> (tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>)
  // CHECK: ^bb0(%[[IN1:.*]]: tensor<2x1xf32>):
  // CHECK: %[[SQRT:.*]] = mfuse.sqrt %[[IN1]] : (tensor<2x1xf32>) -> tensor<2x1xf32>
  // CHECK: %[[RECIP:.*]] = mfuse.reciprocal %[[SQRT]] : (tensor<2x1xf32>) -> tensor<2x1xf32>
  // CHECK: %[[NEG:.*]] = mfuse.neg %[[RECIP]] : (tensor<2x1xf32>) -> tensor<2x1xf32>
  // CHECK: %[[EXP:.*]] = mfuse.exp %[[NEG]] : (tensor<2x1xf32>) -> tensor<2x1xf32>
  // CHECK: mfuse.yield %[[SQRT]], %[[RECIP]], %[[NEG]], %[[EXP]] : tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>
  // CHECK: return %[[FUSED1]]#0, %[[FUSED1]]#1, %[[FUSED1]]#2, %[[FUSED1]]#3 : tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>
  func.func @test_reduce_bwd_post_reduce_too_many_outputs(%arg0: tensor<2x4xf32>)
      -> (tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) {
    %0:4 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<2x4xf32>) -> (tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>) {
    ^bb0(%arg1: tensor<2x4xf32>):
      %1 = mfuse.abs %arg1 : (tensor<2x4xf32>) -> tensor<2x4xf32>
      %2 = mfuse.reduce_sum %1 {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
      %3 = mfuse.sqrt %2 : (tensor<2x1xf32>) -> tensor<2x1xf32>
      %4 = mfuse.reciprocal %3 : (tensor<2x1xf32>) -> tensor<2x1xf32>
      %5 = mfuse.neg %4 : (tensor<2x1xf32>) -> tensor<2x1xf32>
      %6 = mfuse.exp %5 : (tensor<2x1xf32>) -> tensor<2x1xf32>
      mfuse.yield %3, %4, %5, %6 : tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>
    }
    return %0#0, %0#1, %0#2, %0#3 : tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>, tensor<2x1xf32>
  }
}
