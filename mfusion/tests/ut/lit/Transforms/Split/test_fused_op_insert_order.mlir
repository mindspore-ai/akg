// RUN: mfusion-opt %s --split | FileCheck %s
module {
  // CHECK-LABEL: func.func @test_mfuse_fused_basic
  func.func @test_mfuse_fused_basic(
    %input: tensor<4x128x7x7xf32>,
    %scalar: tensor<f32>,
    %alt_value: tensor<4x128x7x7xf32>,
    %stat1: tensor<128xf32>,
    %stat2: tensor<128xf32>,
    %input2: tensor<4x128x7x7xf32>,
    %gamma: tensor<128xf32>
  ) -> (tensor<128xf32>, tensor<4x128x7x7xf32>, tensor<128xf32>) {
    // First fusedop: stat1 path (add, rsqrt, mul, reshape)
    // CHECK: %{{.*}}:2 = mfuse.fused
    // CHECK: (tensor<128xf32>, tensor<128xf32>) -> (tensor<128xf32>, tensor<1x128x1x1xf32>)
    // CHECK: mfuse.add
    // CHECK: mfuse.rsqrt
    // CHECK: mfuse.mul
    // CHECK: mfuse.reshape
    // CHECK: mfuse.yield
    // Second fusedop: input path (le, select, reshape, reduce_sum, sub, mul)
    // CHECK: %{{.*}}:3 = mfuse.fused
    // CHECK: (tensor<4x128x7x7xf32>, tensor<f32>, tensor<4x128x7x7xf32>, tensor<128xf32>, tensor<4x128x7x7xf32>, tensor<1x128x1x1xf32>) -> (tensor<128xf32>, tensor<4x128x7x7xf32>, tensor<4x128x7x7xf32>)
    // CHECK: mfuse.le
    // CHECK: mfuse.select
    // CHECK: mfuse.reshape
    // CHECK: mfuse.reduce_sum
    // CHECK: mfuse.sub
    // CHECK: mfuse.mul
    // CHECK: mfuse.mul
    // CHECK: mfuse.yield
    // Remaining ops outside fusedops
    // CHECK: mfuse.reduce_sum
    // CHECK: mfuse.mul
    %0:3 = mfuse.fused %input, %scalar, %alt_value, %stat1, %stat2, %input2, %gamma
      {fusion_type = "dvm"} : (
        tensor<4x128x7x7xf32>, tensor<f32>, tensor<4x128x7x7xf32>, tensor<128xf32>, tensor<128xf32>,
        tensor<4x128x7x7xf32>, tensor<128xf32>
      ) -> (tensor<128xf32>, tensor<4x128x7x7xf32>, tensor<128xf32>) {
      ^bb0(%arg_x: tensor<4x128x7x7xf32>, %arg_scalar: tensor<f32>, %arg_alt: tensor<4x128x7x7xf32>,
           %arg_stat1: tensor<128xf32>, %arg_stat2: tensor<128xf32>, %arg_input2: tensor<4x128x7x7xf32>,
           %arg_gamma: tensor<128xf32>):
        %eps = mfuse.constant dense<9.99999974E-6> : tensor<f64, {is_scalar = ""}>
        %zero = mfuse.constant dense<0.000000e+00> : tensor<f64, {is_scalar = ""}>
        %mask = mfuse.le %arg_x, %zero : (tensor<4x128x7x7xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x128x7x7xi1>
        %selected = mfuse.select %mask, %arg_scalar, %arg_alt : (tensor<4x128x7x7xi1>, tensor<f32>, tensor<4x128x7x7xf32>) -> tensor<4x128x7x7xf32>
        %var_eps = mfuse.add %arg_stat1, %eps : (tensor<128xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<128xf32>
        %inv_std = mfuse.rsqrt %var_eps : (tensor<128xf32>) -> tensor<128xf32>
        %stat2_reshaped = mfuse.reshape %arg_stat2 {is_squeeze_like = true} : (tensor<128xf32>) -> tensor<1x128x1x1xf32>
        %sum1 = mfuse.reduce_sum %selected {dimensions = [0, 2, 3], keepdim = false} : (tensor<4x128x7x7xf32>) -> tensor<128xf32>
        %centered = mfuse.sub %arg_input2, %stat2_reshaped : (tensor<4x128x7x7xf32>, tensor<1x128x1x1xf32>) -> tensor<4x128x7x7xf32>
        %scaled = mfuse.mul %selected, %centered : (tensor<4x128x7x7xf32>, tensor<4x128x7x7xf32>) -> tensor<4x128x7x7xf32>
        %sum2 = mfuse.reduce_sum %scaled {dimensions = [0, 2, 3], keepdim = false} : (tensor<4x128x7x7xf32>) -> tensor<128xf32>
        %gamma_scaled = mfuse.mul %inv_std, %arg_gamma : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
        %gamma_reshaped = mfuse.reshape %gamma_scaled {is_squeeze_like = true} : (tensor<128xf32>) -> tensor<1x128x1x1xf32>
        %out = mfuse.mul %selected, %gamma_reshaped : (tensor<4x128x7x7xf32>, tensor<1x128x1x1xf32>) -> tensor<4x128x7x7xf32>
        %stat_out = mfuse.mul %sum2, %inv_std : (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
        mfuse.yield %sum1, %out, %stat_out : tensor<128xf32>, tensor<4x128x7x7xf32>, tensor<128xf32>
    }
    return %0#0, %0#1, %0#2 : tensor<128xf32>, tensor<4x128x7x7xf32>, tensor<128xf32>
  }
}
