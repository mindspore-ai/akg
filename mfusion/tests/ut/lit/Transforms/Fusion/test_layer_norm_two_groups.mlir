// RUN: mfusion-opt %s --fuse-layer-norm-dvm --canonicalize | FileCheck %s --check-prefix=TAG
// RUN: mfusion-opt %s --fuse-layer-norm-dvm --mfuse-dvm-cluster --split --canonicalize | FileCheck %s --check-prefix=SPLIT

// Two independent LayerNorm chains must receive distinct dvm_fuse_group ids.
module {
  // TAG-LABEL: func @test_two_layer_norm_groups
  // TAG-DAG: mfusion.dvm_fuse_group = "layer_norm#0"
  // TAG-DAG: mfusion.dvm_fuse_group = "layer_norm#1"
  // TAG-NOT: mfusion.dvm_fuse_group = "layer_norm#2"
  //
  // SPLIT-LABEL: func @test_two_layer_norm_groups
  // SPLIT: mfuse.fused
  // SPLIT-NOT: mfuse.aclnn
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
}
