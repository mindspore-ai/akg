// RUN: mfusion-opt %s --pass-pipeline="builtin.module(fuse-batch-norm)" | FileCheck %s --check-prefix=STRICT

module {
  // Decomposition pattern aligned with inductor/mfuse lowering (e.g. ResNet-style BN on NCHW).
  func.func @fuse_batch_norm_nchw(%x: tensor<4x16x32x32xf32>, %mean: tensor<16xf32>, %var: tensor<16xf32>,
      %gamma: tensor<16xf32>, %beta: tensor<16xf32>) -> tensor<4x16x32x32xf32> {
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f64, {is_scalar = ""}>
    %one = mfuse.constant dense<1.000000e+00> : tensor<f64, {is_scalar = ""}>
    %vpe = mfuse.add %var, %eps : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %sv = mfuse.sqrt %vpe : (tensor<16xf32>) -> tensor<16xf32>
    %iv = mfuse.reciprocal %sv : (tensor<16xf32>) -> tensor<16xf32>
    %ivs = mfuse.mul %iv, %one : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %mbc = mfuse.reshape %mean : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %ibc = mfuse.reshape %ivs : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %gbc = mfuse.reshape %gamma : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %bbc = mfuse.reshape %beta : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %xc = mfuse.sub %x, %mbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n0 = mfuse.mul %xc, %ibc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n1 = mfuse.mul %n0, %gbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %out = mfuse.add %n1, %bbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    // STRICT-LABEL: func @fuse_batch_norm_nchw
    // STRICT: mfuse.reciprocal
    // STRICT-NOT: mfuse.aclnn.batch_norm
    return %out : tensor<4x16x32x32xf32>
  }

  func.func @fuse_batch_norm_const_stats_positive(%x: tensor<4x16x32x32xf32>) -> tensor<4x16x32x32xf32> {
    %mean = mfuse.constant dense<1.000000e+00> : tensor<16xf32>
    %var = mfuse.constant dense<4.000000e+00> : tensor<16xf32>
    %gamma = mfuse.constant dense<2.000000e+00> : tensor<16xf32>
    %beta = mfuse.constant dense<5.000000e-01> : tensor<16xf32>
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f64, {is_scalar = ""}>
    %one = mfuse.constant dense<1.000000e+00> : tensor<f64, {is_scalar = ""}>
    %vpe = mfuse.add %var, %eps : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %sv = mfuse.sqrt %vpe : (tensor<16xf32>) -> tensor<16xf32>
    %iv = mfuse.reciprocal %sv : (tensor<16xf32>) -> tensor<16xf32>
    %ivs = mfuse.mul %iv, %one : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %mbc = mfuse.reshape %mean : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %ibc = mfuse.reshape %ivs : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %gbc = mfuse.reshape %gamma : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %bbc = mfuse.reshape %beta : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %xc = mfuse.sub %x, %mbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n0 = mfuse.mul %xc, %ibc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n1 = mfuse.mul %n0, %gbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %out = mfuse.add %n1, %bbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    // STRICT-LABEL: func @fuse_batch_norm_const_stats_positive
    // STRICT: mfuse.reciprocal
    // STRICT-NOT: mfuse.aclnn.batch_norm
    return %out : tensor<4x16x32x32xf32>
  }

  // Negative: mean from external const while var from online reduce path.
  func.func @fuse_batch_norm_mixed_stats_source_negative(
      %x: tensor<4x16x32x32xf32>, %gamma: tensor<16xf32>, %beta: tensor<16xf32>) -> tensor<4x16x32x32xf32> {
    %mean = mfuse.constant dense<1.000000e+00> : tensor<16xf32>
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f64, {is_scalar = ""}>
    %invNhw = mfuse.constant dense<2.44140625E-4> : tensor<f64, {is_scalar = ""}>
    %sum = mfuse.reduce_sum %x {dimensions = [0, 2, 3], keepdim = false}
      : (tensor<4x16x32x32xf32>) -> tensor<16xf32>
    %meanForVar = mfuse.mul %sum, %invNhw : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %meanBcVar = mfuse.reshape %meanForVar : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %xcVar = mfuse.sub %x, %meanBcVar : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %sq = mfuse.mul %xcVar, %xcVar : (tensor<4x16x32x32xf32>, tensor<4x16x32x32xf32>) -> tensor<4x16x32x32xf32>
    %sumVar = mfuse.reduce_sum %sq {dimensions = [0, 2, 3], keepdim = false}
      : (tensor<4x16x32x32xf32>) -> tensor<16xf32>
    %var = mfuse.mul %sumVar, %invNhw : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %vpe = mfuse.add %var, %eps : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %sv = mfuse.sqrt %vpe : (tensor<16xf32>) -> tensor<16xf32>
    %iv = mfuse.reciprocal %sv : (tensor<16xf32>) -> tensor<16xf32>
    %ivBc = mfuse.reshape %iv : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %gBc = mfuse.reshape %gamma : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %bBc = mfuse.reshape %beta : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %meanBcBn = mfuse.reshape %mean : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %xcBn = mfuse.sub %x, %meanBcBn : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n0 = mfuse.mul %xcBn, %ivBc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n1 = mfuse.mul %n0, %gBc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %out = mfuse.add %n1, %bBc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    // STRICT-LABEL: func @fuse_batch_norm_mixed_stats_source_negative
    // STRICT: mfuse.reciprocal
    // STRICT-NOT: mfuse.aclnn.batch_norm
    return %out : tensor<4x16x32x32xf32>
  }

  // Negative: eps is abnormally large and should fail conservative BN filter.
  func.func @fuse_batch_norm_large_eps_negative(
      %x: tensor<4x16x32x32xf32>, %mean: tensor<16xf32>, %var: tensor<16xf32>,
      %gamma: tensor<16xf32>, %beta: tensor<16xf32>) -> tensor<4x16x32x32xf32> {
    %eps = mfuse.constant dense<1.00000000E-1> : tensor<f64, {is_scalar = ""}>
    %one = mfuse.constant dense<1.000000e+00> : tensor<f64, {is_scalar = ""}>
    %vpe = mfuse.add %var, %eps : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %sv = mfuse.sqrt %vpe : (tensor<16xf32>) -> tensor<16xf32>
    %iv = mfuse.reciprocal %sv : (tensor<16xf32>) -> tensor<16xf32>
    %ivs = mfuse.mul %iv, %one : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %mbc = mfuse.reshape %mean : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %ibc = mfuse.reshape %ivs : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %gbc = mfuse.reshape %gamma : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %bbc = mfuse.reshape %beta : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %xc = mfuse.sub %x, %mbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n0 = mfuse.mul %xc, %ibc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n1 = mfuse.mul %n0, %gbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %out = mfuse.add %n1, %bbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    // STRICT-LABEL: func @fuse_batch_norm_large_eps_negative
    // STRICT: mfuse.reciprocal
    // STRICT-NOT: mfuse.aclnn.batch_norm
    return %out : tensor<4x16x32x32xf32>
  }

  func.func @fuse_batch_norm_dynamic_rank1_negative(
      %x: tensor<4x16x32x32xf32>, %mean: tensor<?xf32>, %var: tensor<?xf32>,
      %gamma: tensor<?xf32>, %beta: tensor<?xf32>) -> tensor<4x16x32x32xf32> {
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f64, {is_scalar = ""}>
    %vpe = mfuse.add %var, %eps : (tensor<?xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<?xf32>
    %sv = mfuse.sqrt %vpe : (tensor<?xf32>) -> tensor<?xf32>
    %iv = mfuse.reciprocal %sv : (tensor<?xf32>) -> tensor<?xf32>
    %mbc = mfuse.reshape %mean : (tensor<?xf32>) -> tensor<16x1x1xf32>
    %ibc = mfuse.reshape %iv : (tensor<?xf32>) -> tensor<16x1x1xf32>
    %gbc = mfuse.reshape %gamma : (tensor<?xf32>) -> tensor<16x1x1xf32>
    %bbc = mfuse.reshape %beta : (tensor<?xf32>) -> tensor<16x1x1xf32>
    %xc = mfuse.sub %x, %mbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n0 = mfuse.mul %xc, %ibc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n1 = mfuse.mul %n0, %gbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %out = mfuse.add %n1, %bbc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    // STRICT-LABEL: func @fuse_batch_norm_dynamic_rank1_negative
    // STRICT: mfuse.reciprocal
    // STRICT-NOT: mfuse.aclnn.batch_norm
    return %out : tensor<4x16x32x32xf32>
  }

  func.func @fuse_batch_norm_with_ex_var_semantics(
      %x: tensor<4x16x32x32xf32>, %gamma: tensor<16xf32>, %beta: tensor<16xf32>) -> tensor<4x16x32x32xf32> {
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f64, {is_scalar = ""}>
    %invNhw = mfuse.constant dense<2.44140625E-4> : tensor<f64, {is_scalar = ""}>
    %sum = mfuse.reduce_sum %x {dimensions = [0, 2, 3], keepdim = false}
      : (tensor<4x16x32x32xf32>) -> tensor<16xf32>
    %mean = mfuse.mul %sum, %invNhw : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %meanBcVar = mfuse.reshape %mean : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %xcVar = mfuse.sub %x, %meanBcVar : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %sq = mfuse.mul %xcVar, %xcVar : (tensor<4x16x32x32xf32>, tensor<4x16x32x32xf32>) -> tensor<4x16x32x32xf32>
    %sumVar = mfuse.reduce_sum %sq {dimensions = [0, 2, 3], keepdim = false}
      : (tensor<4x16x32x32xf32>) -> tensor<16xf32>
    %var = mfuse.mul %sumVar, %invNhw : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %vpe = mfuse.add %var, %eps : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %sv = mfuse.sqrt %vpe : (tensor<16xf32>) -> tensor<16xf32>
    %iv = mfuse.reciprocal %sv : (tensor<16xf32>) -> tensor<16xf32>
    %ivBc = mfuse.reshape %iv : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %gBc = mfuse.reshape %gamma : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %bBc = mfuse.reshape %beta : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %meanBcBn = mfuse.reshape %mean : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %xcBn = mfuse.sub %x, %meanBcBn : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n0 = mfuse.mul %xcBn, %ivBc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n1 = mfuse.mul %n0, %gBc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %out = mfuse.add %n1, %bBc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    // STRICT-LABEL: func @fuse_batch_norm_with_ex_var_semantics
    // STRICT: mfuse.aclnn.batch_norm
    return %out : tensor<4x16x32x32xf32>
  }

  // Positive: addBeta has multiple users (returned twice), should still fuse.
  func.func @fuse_batch_norm_addbeta_multi_user_positive(
      %x: tensor<4x16x32x32xf32>, %gamma: tensor<16xf32>, %beta: tensor<16xf32>)
      -> (tensor<4x16x32x32xf32>, tensor<4x16x32x32xf32>) {
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f64, {is_scalar = ""}>
    %invNhw = mfuse.constant dense<2.44140625E-4> : tensor<f64, {is_scalar = ""}>
    %sum = mfuse.reduce_sum %x {dimensions = [0, 2, 3], keepdim = false}
      : (tensor<4x16x32x32xf32>) -> tensor<16xf32>
    %mean = mfuse.mul %sum, %invNhw : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %meanBcVar = mfuse.reshape %mean : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %xcVar = mfuse.sub %x, %meanBcVar : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %sq = mfuse.mul %xcVar, %xcVar : (tensor<4x16x32x32xf32>, tensor<4x16x32x32xf32>) -> tensor<4x16x32x32xf32>
    %sumVar = mfuse.reduce_sum %sq {dimensions = [0, 2, 3], keepdim = false}
      : (tensor<4x16x32x32xf32>) -> tensor<16xf32>
    %var = mfuse.mul %sumVar, %invNhw : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %vpe = mfuse.add %var, %eps : (tensor<16xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<16xf32>
    %sv = mfuse.sqrt %vpe : (tensor<16xf32>) -> tensor<16xf32>
    %iv = mfuse.reciprocal %sv : (tensor<16xf32>) -> tensor<16xf32>
    %ivBc = mfuse.reshape %iv : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %gBc = mfuse.reshape %gamma : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %bBc = mfuse.reshape %beta : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %meanBcBn = mfuse.reshape %mean : (tensor<16xf32>) -> tensor<16x1x1xf32>
    %xcBn = mfuse.sub %x, %meanBcBn : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n0 = mfuse.mul %xcBn, %ivBc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %n1 = mfuse.mul %n0, %gBc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %out = mfuse.add %n1, %bBc : (tensor<4x16x32x32xf32>, tensor<16x1x1xf32>) -> tensor<4x16x32x32xf32>
    // STRICT-LABEL: func @fuse_batch_norm_addbeta_multi_user_positive
    // STRICT: %[[BN:.+]] = mfuse.aclnn.batch_norm
    // STRICT: return %[[BN]], %[[BN]]
    return %out, %out : tensor<4x16x32x32xf32>, tensor<4x16x32x32xf32>
  }

  // LayerNorm-like negative case #1:
  func.func @layer_norm_like_last_dim_negative(
      %x: tensor<4x8xf32>, %gamma: tensor<8xf32>, %beta: tensor<8xf32>) -> tensor<4x8xf32> {
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f64, {is_scalar = ""}>
    %invHidden = mfuse.constant dense<1.25000000E-1> : tensor<f64, {is_scalar = ""}>
    %sum = mfuse.reduce_sum %x {dimensions = [1], keepdim = true} : (tensor<4x8xf32>) -> tensor<4x1xf32>
    %mean = mfuse.mul %sum, %invHidden : (tensor<4x1xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x1xf32>
    %xc = mfuse.sub %x, %mean : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
    %sq = mfuse.mul %xc, %xc : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    %sumVar = mfuse.reduce_sum %sq {dimensions = [1], keepdim = true} : (tensor<4x8xf32>) -> tensor<4x1xf32>
    %var = mfuse.mul %sumVar, %invHidden : (tensor<4x1xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x1xf32>
    %vpe = mfuse.add %var, %eps : (tensor<4x1xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x1xf32>
    %sv = mfuse.sqrt %vpe : (tensor<4x1xf32>) -> tensor<4x1xf32>
    %iv = mfuse.reciprocal %sv : (tensor<4x1xf32>) -> tensor<4x1xf32>
    %n0 = mfuse.mul %xc, %iv : (tensor<4x8xf32>, tensor<4x1xf32>) -> tensor<4x8xf32>
    %g = mfuse.reshape %gamma : (tensor<8xf32>) -> tensor<1x8xf32>
    %b = mfuse.reshape %beta : (tensor<8xf32>) -> tensor<1x8xf32>
    %n1 = mfuse.mul %n0, %g : (tensor<4x8xf32>, tensor<1x8xf32>) -> tensor<4x8xf32>
    %out = mfuse.add %n1, %b : (tensor<4x8xf32>, tensor<1x8xf32>) -> tensor<4x8xf32>
    // STRICT-LABEL: func @layer_norm_like_last_dim_negative
    // STRICT: mfuse.reciprocal
    // STRICT-NOT: mfuse.aclnn.batch_norm
    return %out : tensor<4x8xf32>
  }

  // LayerNorm-like negative case #2:
  func.func @layer_norm_like_c_axis_negative(
      %x: tensor<4x16x32x32xf32>, %gamma: tensor<16xf32>, %beta: tensor<16xf32>) -> tensor<4x16x32x32xf32> {
    %eps = mfuse.constant dense<9.99999974E-6> : tensor<f64, {is_scalar = ""}>
    %invC = mfuse.constant dense<6.25000000E-2> : tensor<f64, {is_scalar = ""}>
    %sum = mfuse.reduce_sum %x {dimensions = [1], keepdim = true}
      : (tensor<4x16x32x32xf32>) -> tensor<4x1x32x32xf32>
    %mean = mfuse.mul %sum, %invC : (tensor<4x1x32x32xf32>, tensor<f64, {is_scalar = ""}>)
      -> tensor<4x1x32x32xf32>
    %xc = mfuse.sub %x, %mean : (tensor<4x16x32x32xf32>, tensor<4x1x32x32xf32>) -> tensor<4x16x32x32xf32>
    %sq = mfuse.mul %xc, %xc : (tensor<4x16x32x32xf32>, tensor<4x16x32x32xf32>) -> tensor<4x16x32x32xf32>
    %sumVar = mfuse.reduce_sum %sq {dimensions = [1], keepdim = true}
      : (tensor<4x16x32x32xf32>) -> tensor<4x1x32x32xf32>
    %var = mfuse.mul %sumVar, %invC : (tensor<4x1x32x32xf32>, tensor<f64, {is_scalar = ""}>)
      -> tensor<4x1x32x32xf32>
    %vpe = mfuse.add %var, %eps : (tensor<4x1x32x32xf32>, tensor<f64, {is_scalar = ""}>)
      -> tensor<4x1x32x32xf32>
    %sv = mfuse.sqrt %vpe : (tensor<4x1x32x32xf32>) -> tensor<4x1x32x32xf32>
    %iv = mfuse.reciprocal %sv : (tensor<4x1x32x32xf32>) -> tensor<4x1x32x32xf32>
    %n0 = mfuse.mul %xc, %iv : (tensor<4x16x32x32xf32>, tensor<4x1x32x32xf32>) -> tensor<4x16x32x32xf32>
    %g = mfuse.reshape %gamma : (tensor<16xf32>) -> tensor<1x16x1x1xf32>
    %b = mfuse.reshape %beta : (tensor<16xf32>) -> tensor<1x16x1x1xf32>
    %n1 = mfuse.mul %n0, %g : (tensor<4x16x32x32xf32>, tensor<1x16x1x1xf32>) -> tensor<4x16x32x32xf32>
    %out = mfuse.add %n1, %b : (tensor<4x16x32x32xf32>, tensor<1x16x1x1xf32>) -> tensor<4x16x32x32xf32>
    // STRICT-LABEL: func @layer_norm_like_c_axis_negative
    // STRICT: mfuse.reciprocal
    // STRICT-NOT: mfuse.aclnn.batch_norm
    return %out : tensor<4x16x32x32xf32>
  }
}
