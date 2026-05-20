// RUN: mfusion-opt %s --fuse-layernorm | FileCheck %s

module {
  // CHECK-LABEL: func @test_layer_norm_fusion
  // CHECK-SAME: (%[[X:.*]]: tensor<4x197x384xf32>, %[[GAMMA:.*]]: tensor<384xf32>, %[[BETA:.*]]: tensor<384xf32>)
  // CHECK: %[[OUT:.*]] = mfuse.aclnn.layer_norm %[[X]], %[[GAMMA]], %[[BETA]]
  // CHECK: return %[[OUT]]
  func.func @test_layer_norm_fusion(%x: tensor<4x197x384xf32>, %gamma: tensor<384xf32>, %beta: tensor<384xf32>)
      -> tensor<4x197x384xf32> {
    %variance, %mean = mfuse.aclnn.var_mean %x {dim = [2], correction = 0 : i64, keepdim = true} : (tensor<4x197x384xf32>) -> (tensor<4x197x384xf32>, tensor<4x197x384xf32>)
    %eps = mfuse.constant dense<1.000000e-05> : tensor<f32, {is_scalar = ""}>
    %var_plus_eps = mfuse.add %variance, %eps : (tensor<4x197x384xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x197x384xf32>
    %rstd = mfuse.rsqrt %var_plus_eps : (tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %x_minus_mean = mfuse.sub %x, %mean : (tensor<4x197x384xf32>, tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %normalized = mfuse.mul %x_minus_mean, %rstd : (tensor<4x197x384xf32>, tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %gamma_scaled = mfuse.mul %normalized, %gamma : (tensor<4x197x384xf32>, tensor<384xf32>) -> tensor<4x197x384xf32>
    %output = mfuse.add %gamma_scaled, %beta : (tensor<4x197x384xf32>, tensor<384xf32>) -> tensor<4x197x384xf32>
    return %output : tensor<4x197x384xf32>
  }

  // CHECK-LABEL: func @test_layer_norm_fusion_multiple_trailing_dims
  // CHECK-SAME: (%[[X:.*]]: tensor<4x197x384xf32>, %[[GAMMA:.*]]: tensor<197x384xf32>, %[[BETA:.*]]: tensor<197x384xf32>)
  // CHECK: %[[OUT:.*]] = mfuse.aclnn.layer_norm %[[X]], %[[GAMMA]], %[[BETA]]
  // CHECK: return %[[OUT]]
  func.func @test_layer_norm_fusion_multiple_trailing_dims(%x: tensor<4x197x384xf32>, %gamma: tensor<197x384xf32>, %beta: tensor<197x384xf32>)
      -> tensor<4x197x384xf32> {
    %variance, %mean = mfuse.aclnn.var_mean %x {dim = [1, 2], correction = 0 : i64, keepdim = true} : (tensor<4x197x384xf32>) -> (tensor<4x197x384xf32>, tensor<4x197x384xf32>)
    %eps = mfuse.constant dense<1.000000e-05> : tensor<f32, {is_scalar = ""}>
    %var_plus_eps = mfuse.add %variance, %eps : (tensor<4x197x384xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x197x384xf32>
    %rstd = mfuse.rsqrt %var_plus_eps : (tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %x_minus_mean = mfuse.sub %x, %mean : (tensor<4x197x384xf32>, tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %normalized = mfuse.mul %x_minus_mean, %rstd : (tensor<4x197x384xf32>, tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %gamma_scaled = mfuse.mul %normalized, %gamma : (tensor<4x197x384xf32>, tensor<197x384xf32>) -> tensor<4x197x384xf32>
    %output = mfuse.add %gamma_scaled, %beta : (tensor<4x197x384xf32>, tensor<197x384xf32>) -> tensor<4x197x384xf32>
    return %output : tensor<4x197x384xf32>
  }

  // CHECK-LABEL: func @test_layer_norm_fusion_all_dims
  // CHECK-SAME: (%[[X:.*]]: tensor<384xf32>, %[[GAMMA:.*]]: tensor<384xf32>, %[[BETA:.*]]: tensor<384xf32>)
  // CHECK: %[[OUT:.*]] = mfuse.aclnn.layer_norm %[[X]], %[[GAMMA]], %[[BETA]]
  // CHECK: return %[[OUT]]
  func.func @test_layer_norm_fusion_all_dims(%x: tensor<384xf32>, %gamma: tensor<384xf32>, %beta: tensor<384xf32>)
      -> tensor<384xf32> {
    %variance, %mean = mfuse.aclnn.var_mean %x {dim = [], correction = 0 : i64, keepdim = true} : (tensor<384xf32>) -> (tensor<384xf32>, tensor<384xf32>)
    %eps = mfuse.constant dense<1.000000e-05> : tensor<f32, {is_scalar = ""}>
    %var_plus_eps = mfuse.add %variance, %eps : (tensor<384xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<384xf32>
    %rstd = mfuse.rsqrt %var_plus_eps : (tensor<384xf32>) -> tensor<384xf32>
    %x_minus_mean = mfuse.sub %x, %mean : (tensor<384xf32>, tensor<384xf32>) -> tensor<384xf32>
    %normalized = mfuse.mul %x_minus_mean, %rstd : (tensor<384xf32>, tensor<384xf32>) -> tensor<384xf32>
    %gamma_scaled = mfuse.mul %normalized, %gamma : (tensor<384xf32>, tensor<384xf32>) -> tensor<384xf32>
    %output = mfuse.add %gamma_scaled, %beta : (tensor<384xf32>, tensor<384xf32>) -> tensor<384xf32>
    return %output : tensor<384xf32>
  }

  // CHECK-LABEL: func @test_layer_norm_no_fusion_non_consecutive_dims
  // CHECK-NOT: mfuse.aclnn.layer_norm
  func.func @test_layer_norm_no_fusion_non_consecutive_dims(%x: tensor<4x197x384xf32>, %gamma: tensor<384xf32>, %beta: tensor<384xf32>)
      -> tensor<4x197x384xf32> {
    %variance, %mean = mfuse.aclnn.var_mean %x {dim = [0, 2], correction = 0 : i64, keepdim = true} : (tensor<4x197x384xf32>) -> (tensor<4x197x384xf32>, tensor<4x197x384xf32>)
    %eps = mfuse.constant dense<1.000000e-05> : tensor<f32, {is_scalar = ""}>
    %var_plus_eps = mfuse.add %variance, %eps : (tensor<4x197x384xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x197x384xf32>
    %rstd = mfuse.rsqrt %var_plus_eps : (tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %x_minus_mean = mfuse.sub %x, %mean : (tensor<4x197x384xf32>, tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %normalized = mfuse.mul %x_minus_mean, %rstd : (tensor<4x197x384xf32>, tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %gamma_scaled = mfuse.mul %normalized, %gamma : (tensor<4x197x384xf32>, tensor<384xf32>) -> tensor<4x197x384xf32>
    %output = mfuse.add %gamma_scaled, %beta : (tensor<4x197x384xf32>, tensor<384xf32>) -> tensor<4x197x384xf32>
    return %output : tensor<4x197x384xf32>
  }

  // CHECK-LABEL: func @test_layer_norm_no_fusion_non_trailing_dims
  // CHECK-NOT: mfuse.aclnn.layer_norm
  func.func @test_layer_norm_no_fusion_non_trailing_dims(%x: tensor<4x197x384xf32>, %gamma: tensor<197xf32>, %beta: tensor<197xf32>)
      -> tensor<4x197x384xf32> {
    %variance, %mean = mfuse.aclnn.var_mean %x {dim = [1], correction = 0 : i64, keepdim = true} : (tensor<4x197x384xf32>) -> (tensor<4x197x384xf32>, tensor<4x197x384xf32>)
    %eps = mfuse.constant dense<1.000000e-05> : tensor<f32, {is_scalar = ""}>
    %var_plus_eps = mfuse.add %variance, %eps : (tensor<4x197x384xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x197x384xf32>
    %rstd = mfuse.rsqrt %var_plus_eps : (tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %x_minus_mean = mfuse.sub %x, %mean : (tensor<4x197x384xf32>, tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %normalized = mfuse.mul %x_minus_mean, %rstd : (tensor<4x197x384xf32>, tensor<4x197x384xf32>) -> tensor<4x197x384xf32>
    %gamma_scaled = mfuse.mul %normalized, %gamma : (tensor<4x197x384xf32>, tensor<197xf32>) -> tensor<4x197x384xf32>
    %output = mfuse.add %gamma_scaled, %beta : (tensor<4x197x384xf32>, tensor<197xf32>) -> tensor<4x197x384xf32>
    return %output : tensor<4x197x384xf32>
  }

  // CHECK-LABEL: func @test_layer_norm_no_fusion_keepdim_false
  // CHECK-NOT: mfuse.aclnn.layer_norm
  func.func @test_layer_norm_no_fusion_keepdim_false(%x: tensor<4x197x384xf32>, %gamma: tensor<384xf32>, %beta: tensor<384xf32>)
      -> tensor<4x197x384xf32> {
    %variance, %mean = mfuse.aclnn.var_mean %x {dim = [2], correction = 0 : i64, keepdim = false} : (tensor<4x197x384xf32>) -> (tensor<4x197xf32>, tensor<4x197xf32>)
    %eps = mfuse.constant dense<1.000000e-05> : tensor<f32, {is_scalar = ""}>
    %var_plus_eps = mfuse.add %variance, %eps : (tensor<4x197xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<4x197xf32>
    %rstd = mfuse.rsqrt %var_plus_eps : (tensor<4x197xf32>) -> tensor<4x197xf32>
    %x_minus_mean = mfuse.sub %x, %mean : (tensor<4x197x384xf32>, tensor<4x197xf32>) -> tensor<4x197x384xf32>
    %normalized = mfuse.mul %x_minus_mean, %rstd : (tensor<4x197x384xf32>, tensor<4x197xf32>) -> tensor<4x197x384xf32>
    %gamma_scaled = mfuse.mul %normalized, %gamma : (tensor<4x197x384xf32>, tensor<384xf32>) -> tensor<4x197x384xf32>
    %output = mfuse.add %gamma_scaled, %beta : (tensor<4x197x384xf32>, tensor<384xf32>) -> tensor<4x197x384xf32>
    return %output : tensor<4x197x384xf32>
  }
}

