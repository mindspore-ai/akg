// RUN: mfusion-opt %s --fuse-layer-norm-dvm | FileCheck %s

module {
  // BERT pm04-style decomposed LN (reduce_mean centering + sqrt var rstd).
  // CHECK-LABEL: func @test_layer_norm_dvm_tag
  // CHECK: mfuse.sqrt
  // CHECK: mfuse.reduce_mean
  // CHECK: mfusion.dvm_fuse_group = "layer_norm#
  // CHECK: mfusion.dvm_fuse_role = "member"
  // CHECK: mfusion.dvm_fuse_kind = "layer_norm"
  // CHECK: mfusion.layer_norm_dvm
  func.func @test_layer_norm_dvm_tag(%x: tensor<2x4xf32>, %gamma: tensor<4xf32>, %beta: tensor<4xf32>,
      %var: tensor<2x1xf32>) -> tensor<2x4xf32> {
    %c_eps = mfuse.constant dense<9.99999974E-6> : tensor<f32, {is_scalar = ""}>
    %sqrt = mfuse.sqrt %var : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %rstd = mfuse.add %sqrt, %c_eps : (tensor<2x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1xf32>
    %mean = mfuse.reduce_mean %x {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
    %mean_bc = mfuse.broadcast_to %mean : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %sub = mfuse.sub %x, %mean_bc : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %gamma_bc = mfuse.broadcast_to %gamma : (tensor<4xf32>) -> tensor<2x4xf32>
    %mul = mfuse.mul %gamma_bc, %sub : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %rstd_bc = mfuse.broadcast_to %rstd : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %div = mfuse.div %mul, %rstd_bc : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %beta_bc = mfuse.broadcast_to %beta : (tensor<4xf32>) -> tensor<2x4xf32>
    %out = mfuse.add %div, %beta_bc : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }

  // aclnn.var_mean path is not matched by fuse-layer-norm-dvm (decompose via AFTER_MANUAL_FUSION first).
  // CHECK-LABEL: func @test_layer_norm_dvm_no_tag
  // CHECK-NOT: mfusion.layer_norm_dvm
  func.func @test_layer_norm_dvm_no_tag(%x: tensor<4x197x384xf32>, %gamma: tensor<384xf32>,
      %beta: tensor<384xf32>) -> tensor<4x197x384xf32> {
    %variance, %mean = mfuse.aclnn.var_mean %x {dim = [2], correction = 0 : i64, keepdim = true}
        : (tensor<4x197x384xf32>) -> (tensor<4x197x384xf32>, tensor<4x197x384xf32>)
    %c_eps = mfuse.constant dense<9.99999994E-6> : tensor<f32, {is_scalar = ""}>
    %var_eps = mfuse.add %variance, %c_eps : (tensor<4x197x384xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x197x384xf32>
    %rstd = mfuse.rsqrt %var_eps : (tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %sub = mfuse.sub %x, %mean : (tensor<4x197x384xf32>, tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %norm = mfuse.mul %sub, %rstd : (tensor<4x197x384xf32>, tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %scaled = mfuse.mul %norm, %gamma : (tensor<4x197x384xf32>, tensor<384xf32>) -> tensor<4x197x384xf32>
    %out = mfuse.add %scaled, %beta : (tensor<4x197x384xf32>, tensor<384xf32>) -> tensor<4x197x384xf32>
    return %out : tensor<4x197x384xf32>
  }

  // CHECK-LABEL: func @test_layer_norm_dvm_bwd_grad_div
  // CHECK: mfusion.dvm_fuse_group = "layer_norm#
  // CHECK: mfusion.dvm_fuse_role = "member"
  // CHECK: mfusion.layer_norm_dvm
  func.func @test_layer_norm_dvm_bwd_grad_div(%x: tensor<2x4xf32>, %rstd: tensor<2x1xf32>) -> tensor<2x4xf32> {
    %c4 = mfuse.constant dense<4.000000e+00> : tensor<f32, {is_scalar = ""}>
    %sum = mfuse.reduce_sum %x {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
    %mean = mfuse.div %sum, %c4 : (tensor<2x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1xf32>
    %mean_bc = mfuse.broadcast_to %mean : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %sub = mfuse.sub %x, %mean_bc : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %rstd_bc = mfuse.broadcast_to %rstd : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %grad = mfuse.div %sub, %rstd_bc : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %grad : tensor<2x4xf32>
  }

  // cqxn/cuahir sums tagged; shared rstd+eps add gets affinity tag (not split merge).
  // CHECK-LABEL: func @test_layer_norm_dvm_bwd_shared_rstd
  // CHECK: mfuse.add
  // CHECK-SAME: mfusion.dvm_fuse_role = "affinity"
  // CHECK-SAME: mfusion.layer_norm_dvm_affinity
  // CHECK: dimensions = [0, 1], keepdim = true
  // CHECK-SAME: mfusion.dvm_fuse_role = "member"
  // CHECK-SAME: mfusion.layer_norm_dvm
  // CHECK: dimensions = [2], keepdim = true
  // CHECK-SAME: mfusion.dvm_fuse_role = "member"
  // CHECK-SAME: mfusion.layer_norm_dvm
  func.func @test_layer_norm_dvm_bwd_shared_rstd(%grad: tensor<2x4x8xf32>, %gamma: tensor<8xf32>,
      %saved: tensor<2x4x8xf32>, %rstd: tensor<2x1x1xf32>) -> (tensor<1x1x8xf32>, tensor<2x4x1xf32>) {
    %eps = mfuse.constant dense<1.000000e-06> : tensor<f32, {is_scalar = ""}>
    %add_eps = mfuse.add %rstd, %eps : (tensor<2x1x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1x1xf32>
    %rstd_eps = mfuse.broadcast_to %add_eps : (tensor<2x1x1xf32>) -> tensor<2x4x8xf32>
    %div_c = mfuse.div %grad, %rstd_eps : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %mul_c = mfuse.mul %div_c, %saved : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %sum_c = mfuse.reduce_sum %mul_c {dimensions = [0, 1], keepdim = true} : (tensor<2x4x8xf32>) -> tensor<1x1x8xf32>
    %neg = mfuse.neg %grad : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %gamma_bc = mfuse.broadcast_to %gamma : (tensor<8xf32>) -> tensor<2x4x8xf32>
    %mul_g = mfuse.mul %gamma_bc, %saved : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %div1 = mfuse.div %mul_g, %rstd_eps : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %div2 = mfuse.div %div1, %rstd_eps : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %mul_h = mfuse.mul %neg, %div2 : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %sum_h = mfuse.reduce_sum %mul_h {dimensions = [2], keepdim = true} : (tensor<2x4x8xf32>) -> tensor<2x4x1xf32>
    return %sum_c, %sum_h : tensor<1x1x8xf32>, tensor<2x4x1xf32>
  }
}
