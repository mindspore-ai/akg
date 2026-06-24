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

  // CHECK-LABEL: func.func @test_reduce_bwd_keepdim_false
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x4xf32>
  // CHECK: %[[FUSED0:.*]] = mfuse.fused %[[ARG0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x4xf32>) -> tensor<2xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<2x4xf32>):
  // CHECK: %[[ABS:.*]] = mfuse.abs %[[IN0]] : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[ABS]] {dimensions = [1], keepdim = false} : (tensor<2x4xf32>) -> tensor<2xf32>
  // CHECK: %[[SQRT:.*]] = mfuse.sqrt %[[REDUCE]] : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: mfuse.yield %[[SQRT]] : tensor<2xf32>
  // CHECK: return %[[FUSED0]] : tensor<2xf32>
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

  // CHECK-LABEL: func.func @test_masked_bn_like_fused_split
  // First fusedop: vector prep branch.
  // CHECK: %[[FUSED0:.*]]:2 = mfuse.fused %arg3, %arg6 {fusion_type = "dvm"}
  // CHECK: %[[EPS:.*]] = mfuse.constant dense<9.9999997400000002E-6>
  // CHECK: %[[INV_STD:.*]] = mfuse.rsqrt
  // CHECK: %[[GAMMA_RESHAPE:.*]] = mfuse.reshape
  // CHECK: mfuse.yield %[[INV_STD]], %[[GAMMA_RESHAPE]]
  // Second fusedop: masked input path.
  // CHECK: %[[FUSED1:.*]]:3 = mfuse.fused %arg0, %arg1, %arg4, %arg5, %[[FUSED0]]#1 {fusion_type = "dvm"}
  // CHECK: %[[INPUT_RESHAPE:.*]] = mfuse.reshape
  // CHECK: %[[FULL:.*]] = mfuse.full
  // CHECK: %[[SELECT:.*]] = mfuse.select
  // CHECK: %[[STAT_RESHAPE:.*]] = mfuse.reshape
  // CHECK: %[[SUM1:.*]] = mfuse.reduce_sum %[[SELECT]] {dimensions = [0, 2, 3], keepdim = false}
  // CHECK: %[[SCALED:.*]] = mfuse.mul
  // CHECK: %[[OUT:.*]] = mfuse.mul %[[SELECT]], %{{.*}} : (tensor<4x128x7x7xf32>, tensor<1x128x1x1xf32>) -> tensor<4x128x7x7xf32>
  // CHECK: mfuse.yield %[[SUM1]], %[[SCALED]], %[[OUT]]
  // Third fusedop: trailing stat_out path.
  // CHECK: %[[FUSED2:.*]] = mfuse.fused %[[FUSED1]]#1, %[[FUSED0]]#0 {fusion_type = "dvm"}
  // CHECK: %[[SUM2:.*]] = mfuse.reduce_sum %{{.*}} {dimensions = [0, 2, 3], keepdim = false}
  // CHECK: %[[STAT:.*]] = mfuse.mul %[[SUM2]], %{{.*}} : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
  // CHECK: mfuse.yield %[[STAT]]
  // CHECK: return %[[FUSED1]]#0, %[[FUSED1]]#2, %[[FUSED2]]
  func.func @test_masked_bn_like_fused_split(%arg0: tensor<4x128xf32>, %arg1: tensor<4x128x7x7xi1>,
      %arg2: tensor<4x128x7x7xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128xf32>,
      %arg5: tensor<4x128x7x7xf32>, %arg6: tensor<128xf32>)
      -> (tensor<128xf32>, tensor<4x128x7x7xf32>, tensor<128xf32>) {
    %0:3 = mfuse.fused %arg3, %arg6, %arg0, %arg1, %arg4, %arg5 {fusion_type = "dvm"} : (tensor<128xf32>, tensor<128xf32>, tensor<4x128xf32>, tensor<4x128x7x7xi1>, tensor<128xf32>, tensor<4x128x7x7xf32>) -> (tensor<128xf32>, tensor<4x128x7x7xf32>, tensor<128xf32>) {
    ^bb0(%arg7: tensor<128xf32>, %arg8: tensor<128xf32>, %arg9: tensor<4x128xf32>, %arg10: tensor<4x128x7x7xi1>, %arg11: tensor<128xf32>, %arg12: tensor<4x128x7x7xf32>):
      %1 = mfuse.constant dense<49> : tensor<i64, {is_scalar = ""}>
      %2 = mfuse.constant dense<0.000000e+00> : tensor<f64, {is_scalar = ""}>
      %3 = mfuse.constant dense<9.9999997400000002E-6> : tensor<f64, {is_scalar = ""}>
      %4 = mfuse.add %arg7, %3 : (tensor<128xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<128xf32>
      %5 = mfuse.rsqrt %4 : (tensor<128xf32>) -> tensor<128xf32>
      %6 = mfuse.mul %5, %arg8 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      %7 = mfuse.reshape %6 : (tensor<128xf32>) -> tensor<1x128x1x1xf32>
      %8 = mfuse.reshape %arg9 : (tensor<4x128xf32>) -> tensor<4x128x1x1xf32>
      %9 = mfuse.broadcast_to %8 : (tensor<4x128x1x1xf32>) -> tensor<4x128x7x7xf32>
      %10 = mfuse.div %9, %1 : (tensor<4x128x7x7xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<4x128x7x7xf32>
      %11 = mfuse.full %2 {device = "npu:0", dtype = 6 : i64, layout = 0 : i64, pin_memory = false} : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
      %12 = mfuse.select %arg10, %11, %10 : (tensor<4x128x7x7xi1>, tensor<f32>, tensor<4x128x7x7xf32>) -> tensor<4x128x7x7xf32>
      %13 = mfuse.reshape %arg11 : (tensor<128xf32>) -> tensor<1x128x1x1xf32>
      %14 = mfuse.reduce_sum %12 {dimensions = [0, 2, 3], keepdim = false} : (tensor<4x128x7x7xf32>) -> tensor<128xf32>
      %15 = mfuse.sub %arg12, %13 : (tensor<4x128x7x7xf32>, tensor<1x128x1x1xf32>) -> tensor<4x128x7x7xf32>
      %16 = mfuse.mul %12, %15 : (tensor<4x128x7x7xf32>, tensor<4x128x7x7xf32>) -> tensor<4x128x7x7xf32>
      %17 = mfuse.mul %12, %7 : (tensor<4x128x7x7xf32>, tensor<1x128x1x1xf32>) -> tensor<4x128x7x7xf32>
      %18 = mfuse.reduce_sum %16 {dimensions = [0, 2, 3], keepdim = false} : (tensor<4x128x7x7xf32>) -> tensor<128xf32>
      %19 = mfuse.mul %18, %5 : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
      mfuse.yield %14, %17, %19 : tensor<128xf32>, tensor<4x128x7x7xf32>, tensor<128xf32>
    }
    return %0#0, %0#1, %0#2 : tensor<128xf32>, tensor<4x128x7x7xf32>, tensor<128xf32>
  }

  // Typical reduce_bwd split shape:
  // 1. vector-prep branch is split out first,
  // 2. masked producer region stays as one fusedop,
  // 3. input reshapes stay inside the rebuilt producer fusedop,
  // 4. one reduce_sum stays as an output of the producer fusedop,
  // 5. the sibling reduce_sum sinks into its post-reduce mul user.
  // CHECK-LABEL: func.func @test_reduce_bwd_standalone_reduce_sum_with_reshape
  // CHECK: %[[FUSED0:.*]]:3 = mfuse.fused %arg1, %arg4 {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<184xf32>, tensor<184xf32>) -> (tensor<184x1x1xf32>, tensor<184xf32>, tensor<1x184x1x1xf32>)
  // CHECK: %[[FUSED1:.*]]:4 = mfuse.fused %arg0, %arg2, %arg3, %[[FUSED0]]#0, %arg4, %arg5, %[[FUSED0]]#2 {fusion_type = "dvm"}
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<4x184xf32>, %[[MEAN_IN:.*]]: tensor<184xf32>, %[[X_IN:.*]]: tensor<4x184x4x4xf32>, %[[INV_STD0:.*]]: tensor<184x1x1xf32>, %[[SCALE_IN:.*]]: tensor<184xf32>, %[[BIAS_IN:.*]]: tensor<184xf32>, %[[INV_STD1:.*]]: tensor<1x184x1x1xf32>):
  // CHECK: %[[INPUT_RESHAPE:.*]] = mfuse.reshape %[[IN0]] : (tensor<4x184xf32>) -> tensor<4x184x1x1xf32>
  // CHECK: %[[BCAST:.*]] = mfuse.broadcast_to %[[INPUT_RESHAPE]] : (tensor<4x184x1x1xf32>) -> tensor<4x184x4x4xf32>
  // CHECK: %[[MEAN0:.*]] = mfuse.reshape %[[MEAN_IN]] : (tensor<184xf32>) -> tensor<184x1x1xf32>
  // CHECK: %[[SCALE:.*]] = mfuse.reshape %[[SCALE_IN]] : (tensor<184xf32>) -> tensor<184x1x1xf32>
  // CHECK: %[[SELECT:.*]] = mfuse.select
  // CHECK: %[[MEAN1:.*]] = mfuse.reshape %[[MEAN_IN]] : (tensor<184xf32>) -> tensor<1x184x1x1xf32>
  // CHECK: %[[SUM0:.*]] = mfuse.reduce_sum %[[SELECT]] {dimensions = [0, 2, 3], keepdim = false} : (tensor<4x184x4x4xf32>) -> tensor<184xf32>
  // CHECK: %[[MUL0:.*]] = mfuse.mul %[[SELECT]], %{{.*}} : (tensor<4x184x4x4xf32>, tensor<4x184x4x4xf32>) -> tensor<4x184x4x4xf32>
  // CHECK: %[[MUL1:.*]] = mfuse.mul %[[SELECT]], %{{.*}} : (tensor<4x184x4x4xf32>, tensor<1x184x1x1xf32>) -> tensor<4x184x4x4xf32>
  // CHECK: mfuse.yield %{{.*}}, %[[SUM0]], %[[MUL0]], %[[MUL1]]
  // CHECK: %[[FUSED2:.*]] = mfuse.fused %[[FUSED1]]#2, %[[FUSED0]]#1 {fusion_type = "dvm"}
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<4x184x4x4xf32>, %[[IN1:.*]]: tensor<184xf32>):
  // CHECK: %[[SUM1:.*]] = mfuse.reduce_sum %[[IN0]] {dimensions = [0, 2, 3], keepdim = false} : (tensor<4x184x4x4xf32>) -> tensor<184xf32>
  // CHECK: %[[STAT:.*]] = mfuse.mul %[[SUM1]], %[[IN1]] : (tensor<184xf32>, tensor<184xf32>) -> tensor<184xf32>
  // CHECK: mfuse.yield %[[STAT]] : tensor<184xf32>
  // CHECK: return %[[FUSED1]]#0, %[[FUSED1]]#1, %[[FUSED1]]#3, %[[FUSED2]] : tensor<f32>, tensor<184xf32>, tensor<4x184x4x4xf32>, tensor<184xf32>
  func.func @test_reduce_bwd_standalone_reduce_sum_with_reshape(
      %arg0: tensor<4x184xf32>,
      %arg1: tensor<184xf32>,
      %arg2: tensor<184xf32>,
      %arg3: tensor<4x184x4x4xf32>,
      %arg4: tensor<184xf32>,
      %arg5: tensor<184xf32>) -> (tensor<f32>, tensor<184xf32>, tensor<4x184x4x4xf32>, tensor<184xf32>) {
    %0:4 = mfuse.fused %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {fusion_type = "dvm"} : (tensor<4x184xf32>, tensor<184xf32>, tensor<184xf32>, tensor<4x184x4x4xf32>, tensor<184xf32>, tensor<184xf32>) -> (tensor<f32>, tensor<184xf32>, tensor<4x184x4x4xf32>, tensor<184xf32>) {
    ^bb0(%arg6: tensor<4x184xf32>, %arg7: tensor<184xf32>, %arg8: tensor<184xf32>, %arg9: tensor<4x184x4x4xf32>, %arg10: tensor<184xf32>, %arg11: tensor<184xf32>):
      %1 = mfuse.constant dense<0> : tensor<i64, {is_scalar = ""}>
      %2 = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
      %3 = mfuse.constant dense<16> : tensor<i64, {is_scalar = ""}>
      %4 = mfuse.constant dense<1.000000e-05> : tensor<f64, {is_scalar = ""}>
      %5 = mfuse.constant dense<0.000000e+00> : tensor<f64, {is_scalar = ""}>
      %6 = mfuse.reshape %arg6 : (tensor<4x184xf32>) -> tensor<4x184x1x1xf32>
      %7 = mfuse.broadcast_to %6 : (tensor<4x184x1x1xf32>) -> tensor<4x184x4x4xf32>
      %8 = mfuse.div %7, %3 : (tensor<4x184x4x4xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<4x184x4x4xf32>
      %9 = mfuse.add %arg7, %4 : (tensor<184xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<184xf32>
      %10 = mfuse.rsqrt %9 : (tensor<184xf32>) -> tensor<184xf32>
      %11 = mfuse.mul %10, %2 : (tensor<184xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<184xf32>
      %12 = mfuse.reshape %arg8 : (tensor<184xf32>) -> tensor<184x1x1xf32>
      %13 = mfuse.reshape %11 : (tensor<184xf32>) -> tensor<184x1x1xf32>
      %14 = mfuse.sub %arg9, %12 : (tensor<4x184x4x4xf32>, tensor<184x1x1xf32>) -> tensor<4x184x4x4xf32>
      %15 = mfuse.mul %14, %13 : (tensor<4x184x4x4xf32>, tensor<184x1x1xf32>) -> tensor<4x184x4x4xf32>
      %16 = mfuse.reshape %arg10 : (tensor<184xf32>) -> tensor<184x1x1xf32>
      %17 = mfuse.mul %15, %16 : (tensor<4x184x4x4xf32>, tensor<184x1x1xf32>) -> tensor<4x184x4x4xf32>
      %18 = mfuse.reshape %arg11 : (tensor<184xf32>) -> tensor<184x1x1xf32>
      %19 = mfuse.add %17, %18 : (tensor<4x184x4x4xf32>, tensor<184x1x1xf32>) -> tensor<4x184x4x4xf32>
      %20 = mfuse.relu %19 : (tensor<4x184x4x4xf32>) -> tensor<4x184x4x4xf32>
      %21 = mfuse.le %20, %1 : (tensor<4x184x4x4xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<4x184x4x4xi1>
      %22 = mfuse.full %5 {device = "npu:0", dtype = 6 : i64, layout = 0 : i64, pin_memory = false} : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
      %23 = mfuse.select %21, %22, %8 : (tensor<4x184x4x4xi1>, tensor<f32>, tensor<4x184x4x4xf32>) -> tensor<4x184x4x4xf32>
      %24 = mfuse.rsqrt %9 : (tensor<184xf32>) -> tensor<184xf32>
      %25 = mfuse.reshape %arg8 : (tensor<184xf32>) -> tensor<1x184x1x1xf32>
      %26 = mfuse.reduce_sum %23 {dimensions = [0, 2, 3], keepdim = false} : (tensor<4x184x4x4xf32>) -> tensor<184xf32>
      %27 = mfuse.sub %arg9, %25 : (tensor<4x184x4x4xf32>, tensor<1x184x1x1xf32>) -> tensor<4x184x4x4xf32>
      %28 = mfuse.mul %23, %27 : (tensor<4x184x4x4xf32>, tensor<4x184x4x4xf32>) -> tensor<4x184x4x4xf32>
      %29 = mfuse.reduce_sum %28 {dimensions = [0, 2, 3], keepdim = false} : (tensor<4x184x4x4xf32>) -> tensor<184xf32>
      %30 = mfuse.mul %24, %arg10 : (tensor<184xf32>, tensor<184xf32>) -> tensor<184xf32>
      %31 = mfuse.reshape %30 : (tensor<184xf32>) -> tensor<1x184x1x1xf32>
      %32 = mfuse.mul %23, %31 : (tensor<4x184x4x4xf32>, tensor<1x184x1x1xf32>) -> tensor<4x184x4x4xf32>
      %33 = mfuse.mul %29, %24 : (tensor<184xf32>, tensor<184xf32>) -> tensor<184xf32>
      mfuse.yield %22, %26, %32, %33 : tensor<f32>, tensor<184xf32>, tensor<4x184x4x4xf32>, tensor<184xf32>
    }
    return %0#0, %0#1, %0#2, %0#3 : tensor<f32>, tensor<184xf32>, tensor<4x184x4x4xf32>, tensor<184xf32>
  }

}
