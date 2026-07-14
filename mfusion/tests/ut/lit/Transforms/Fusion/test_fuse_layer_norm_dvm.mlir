// RUN: mfusion-opt %s --fuse-layer-norm-dvm | FileCheck %s
// RUN: mfusion-opt %s --fuse-layer-norm-dvm --canonicalize | FileCheck %s --check-prefix=TAG
// RUN: mfusion-opt %s --mfuse-fusion="kernel-generator=akg" | FileCheck %s --check-prefix=NON-DVM

module {
  // BERT pm04-style decomposed LN (reduce_mean centering + sqrt var rstd).
  // NON-DVM-LABEL: func @test_layer_norm_dvm_tag
  // NON-DVM-NOT: mfuse.fused
  // NON-DVM: return
  // CHECK-LABEL: func @test_layer_norm_dvm_tag
  // CHECK: mfuse.fused
  // CHECK-SAME: fusion_type = "dvm"
  // CHECK-SAME: mfusion.dvm_fuse_group = "layer_norm#0"
  // CHECK-SAME: mfusion.dvm_fuse_kind = "layer_norm"
  // CHECK-SAME: mfusion.dvm_fuse_role = "member"
  // CHECK-DAG: mfuse.sqrt
  // CHECK-DAG: mfuse.reduce_sum
  // CHECK-DAG: mfuse.div
  // CHECK-DAG: mfuse.yield
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

  // Candidate-local preparation must not decompose an unrelated mean in the same function.
  // CHECK-LABEL: func @test_layer_norm_with_unrelated_mean
  // CHECK: mfuse.fused
  // CHECK-COUNT-1: mfuse.reduce_mean
  func.func @test_layer_norm_with_unrelated_mean(%x: tensor<2x4xf32>, %other: tensor<2x8xf32>,
      %gamma: tensor<4xf32>, %beta: tensor<4xf32>, %var: tensor<2x1xf32>)
      -> (tensor<2x4xf32>, tensor<2x1xf32>) {
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f32, {is_scalar = ""}>
    %sqrt = mfuse.sqrt %var : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %denom = mfuse.add %sqrt, %eps : (tensor<2x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1xf32>
    %mean = mfuse.reduce_mean %x {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
    %center = mfuse.sub %x, %mean : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
    %scaled = mfuse.mul %gamma, %center : (tensor<4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %norm = mfuse.div %scaled, %denom : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
    %out = mfuse.add %norm, %beta : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    %other_mean = mfuse.reduce_mean %other {dimensions = [1], keepdim = true}
        : (tensor<2x8xf32>) -> tensor<2x1xf32>
    return %out, %other_mean : tensor<2x4xf32>, tensor<2x1xf32>
  }

  // Rsqrt(var + eps) is a different normalization form and must remain unfused.
  // CHECK-LABEL: func @test_layer_norm_dvm_no_tag
  // CHECK-NOT: mfuse.fused
  // CHECK-NOT: mfusion.dvm_fuse_kind = "layer_norm"
  func.func @test_layer_norm_dvm_no_tag(%x: tensor<4x197x384xf32>, %variance: tensor<4x197x1xf32>,
      %gamma: tensor<384xf32>, %beta: tensor<384xf32>) -> tensor<4x197x384xf32> {
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f32, {is_scalar = ""}>
    %var_eps = mfuse.add %variance, %eps : (tensor<4x197x1xf32>, tensor<f32, {is_scalar = ""}>)
        -> tensor<4x197x1xf32>
    %rstd = mfuse.rsqrt %var_eps : (tensor<4x197x1xf32>) -> tensor<4x197x1xf32>
    %mean = mfuse.reduce_mean %x {dimensions = [2], keepdim = true}
        : (tensor<4x197x384xf32>) -> tensor<4x197x1xf32>
    %center = mfuse.sub %x, %mean : (tensor<4x197x384xf32>, tensor<4x197x1xf32>) -> tensor<4x197x384xf32>
    %scaled = mfuse.mul %gamma, %center : (tensor<384xf32>, tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %norm = mfuse.div %scaled, %rstd : (tensor<4x197x384xf32>, tensor<4x197x1xf32>) -> tensor<4x197x384xf32>
    %out = mfuse.add %norm, %beta : (tensor<4x197x384xf32>, tensor<384xf32>) -> tensor<4x197x384xf32>
    return %out : tensor<4x197x384xf32>
  }

  // BERT uses correction=1 aclnn.var_mean followed by sqrt(var) + eps.
  // The DVM pass must decompose var_mean locally before matching the LayerNorm island.
  // CHECK-LABEL: func @test_layer_norm_dvm_var_mean_correction_one
  // CHECK: mfuse.fused
  // CHECK-SAME: fusion_type = "dvm"
  // CHECK-SAME: mfusion.dvm_fuse_group = "layer_norm#0"
  // CHECK-SAME: mfusion.dvm_fuse_kind = "layer_norm"
  // CHECK-NOT: mfuse.aclnn.var_mean
  func.func @test_layer_norm_dvm_var_mean_correction_one(%x: tensor<4x128x768xf32>,
      %gamma: tensor<768xf32>, %beta: tensor<768xf32>)
      -> (tensor<4x128x768xf32>, !torch.vtensor<[4,128,1],f32>, !torch.vtensor<[4,128,768],f32>) {
    %eps = mfuse.constant dense<9.9999999999999995E-7> : tensor<f64, {is_scalar = ""}>
    %variance, %mean = mfuse.aclnn.var_mean %x {correction = 1 : i64, dim = [2], keepdim = true}
        : (tensor<4x128x768xf32>) -> (tensor<4x128x1xf32>, tensor<4x128x1xf32>)
    %sqrt = mfuse.sqrt %variance : (tensor<4x128x1xf32>) -> tensor<4x128x1xf32>
    %saved_sqrt = builtin.unrealized_conversion_cast %sqrt
        : tensor<4x128x1xf32> to !torch.vtensor<[4,128,1],f32>
    %center = mfuse.sub %x, %mean : (tensor<4x128x768xf32>, tensor<4x128x1xf32>) -> tensor<4x128x768xf32>
    %saved_center = builtin.unrealized_conversion_cast %center
        : tensor<4x128x768xf32> to !torch.vtensor<[4,128,768],f32>
    %scaled = mfuse.mul %gamma, %center : (tensor<768xf32>, tensor<4x128x768xf32>) -> tensor<4x128x768xf32>
    %denom = mfuse.add %sqrt, %eps : (tensor<4x128x1xf32>, tensor<f64, {is_scalar = ""}>)
        -> tensor<4x128x1xf32>
    %norm = mfuse.div %scaled, %denom : (tensor<4x128x768xf32>, tensor<4x128x1xf32>)
        -> tensor<4x128x768xf32>
    %out = mfuse.add %norm, %beta : (tensor<4x128x768xf32>, tensor<768xf32>) -> tensor<4x128x768xf32>
    return %out, %saved_sqrt, %saved_center
        : tensor<4x128x768xf32>, !torch.vtensor<[4,128,1],f32>, !torch.vtensor<[4,128,768],f32>
  }

  // An additional mfuse compute user is not a boundary export and must still block fusion.
  // CHECK-LABEL: func @test_layer_norm_dvm_extra_compute_user_no_fuse
  // CHECK-NOT: mfuse.fused
  func.func @test_layer_norm_dvm_extra_compute_user_no_fuse(%x: tensor<2x4xf32>,
      %gamma: tensor<4xf32>, %beta: tensor<4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f32, {is_scalar = ""}>
    %variance, %mean = mfuse.aclnn.var_mean %x {correction = 1 : i64, dim = [1], keepdim = true}
        : (tensor<2x4xf32>) -> (tensor<2x1xf32>, tensor<2x1xf32>)
    %sqrt = mfuse.sqrt %variance : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %center = mfuse.sub %x, %mean : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
    %extra = mfuse.neg %center : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %scaled = mfuse.mul %gamma, %center : (tensor<4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %denom = mfuse.add %sqrt, %eps : (tensor<2x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1xf32>
    %norm = mfuse.div %scaled, %denom : (tensor<2x4xf32>, tensor<2x1xf32>) -> tensor<2x4xf32>
    %out = mfuse.add %norm, %beta : (tensor<2x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
    return %out, %extra : tensor<2x4xf32>, tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_layer_norm_dvm_bwd_grad_div
  // CHECK: mfuse.fused
  // CHECK-SAME: mfusion.dvm_fuse_kind = "layer_norm"
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

  // Shared rstd+eps stays outside fused islands; cqxn/cuahir each materialize one island.
  // CHECK-LABEL: func @test_layer_norm_dvm_bwd_shared_rstd
  // CHECK-DAG: mfuse.add
  // CHECK-COUNT-2: mfuse.fused
  // CHECK-SAME: mfusion.dvm_fuse_kind = "layer_norm"
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

  // TAG-LABEL: func @test_two_layer_norm_groups
  // TAG-DAG: mfusion.dvm_fuse_group = "layer_norm#0"
  // TAG-DAG: mfusion.dvm_fuse_group = "layer_norm#1"
  // TAG-DAG: mfuse.fused
  // TAG-DAG: mfuse.fused
  // TAG-NOT: mfusion.dvm_fuse_group = "layer_norm#2"
  //
  func.func @test_two_layer_norm_groups(%x: tensor<2x4xf32>, %gamma: tensor<4xf32>, %beta: tensor<4xf32>,
      %var0: tensor<2x1xf32>, %var1: tensor<2x1xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
    %c_eps = mfuse.constant dense<9.99999974E-6> : tensor<f32, {is_scalar = ""}>

    %sqrt0 = mfuse.sqrt %var0 : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %rstd0 = mfuse.add %sqrt0, %c_eps : (tensor<2x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1xf32>
    %mean0 = mfuse.reduce_mean %x {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
    %mean_bc0 = mfuse.broadcast_to %mean0 : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %sub0 = mfuse.sub %x, %mean_bc0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %gamma_bc0 = mfuse.broadcast_to %gamma : (tensor<4xf32>) -> tensor<2x4xf32>
    %mul0 = mfuse.mul %gamma_bc0, %sub0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %rstd_bc0 = mfuse.broadcast_to %rstd0 : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %div0 = mfuse.div %mul0, %rstd_bc0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %beta_bc0 = mfuse.broadcast_to %beta : (tensor<4xf32>) -> tensor<2x4xf32>
    %out0 = mfuse.add %div0, %beta_bc0 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    %sqrt1 = mfuse.sqrt %var1 : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %rstd1 = mfuse.add %sqrt1, %c_eps : (tensor<2x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1xf32>
    %mean1 = mfuse.reduce_mean %x {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
    %mean_bc1 = mfuse.broadcast_to %mean1 : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %sub1 = mfuse.sub %x, %mean_bc1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %gamma_bc1 = mfuse.broadcast_to %gamma : (tensor<4xf32>) -> tensor<2x4xf32>
    %mul1 = mfuse.mul %gamma_bc1, %sub1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %rstd_bc1 = mfuse.broadcast_to %rstd1 : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %div1 = mfuse.div %mul1, %rstd_bc1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %beta_bc1 = mfuse.broadcast_to %beta : (tensor<4xf32>) -> tensor<2x4xf32>
    %out1 = mfuse.add %div1, %beta_bc1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    return %out0, %out1 : tensor<2x4xf32>, tensor<2x4xf32>
  }

  // Division direction is semantic: denominator / scaled is not LayerNorm.
  // CHECK-LABEL: func @test_layer_norm_reversed_div_no_fuse
  // CHECK-NOT: mfuse.fused
  func.func @test_layer_norm_reversed_div_no_fuse(%x: tensor<2x4xf32>, %gamma: tensor<4xf32>,
      %beta: tensor<4xf32>, %var: tensor<2x1xf32>) -> tensor<2x4xf32> {
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f32, {is_scalar = ""}>
    %sqrt = mfuse.sqrt %var : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %denom = mfuse.add %sqrt, %eps : (tensor<2x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1xf32>
    %mean = mfuse.reduce_mean %x {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
    %mean_bc = mfuse.broadcast_to %mean : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %center = mfuse.sub %x, %mean_bc : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %gamma_bc = mfuse.broadcast_to %gamma : (tensor<4xf32>) -> tensor<2x4xf32>
    %scaled = mfuse.mul %center, %gamma_bc : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %denom_bc = mfuse.broadcast_to %denom : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %reversed = mfuse.div %denom_bc, %scaled : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %beta_bc = mfuse.broadcast_to %beta : (tensor<4xf32>) -> tensor<2x4xf32>
    %out = mfuse.add %reversed, %beta_bc : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }

  // A function without a forward candidate must not be changed by LN prep.
  // CHECK-LABEL: func @test_unrelated_reduce_mean_unchanged
  // CHECK: mfuse.reduce_mean
  func.func @test_unrelated_reduce_mean_unchanged(%x: tensor<2x4xf32>) -> tensor<2x1xf32> {
    %mean = mfuse.reduce_mean %x {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
    return %mean : tensor<2x1xf32>
  }
}
