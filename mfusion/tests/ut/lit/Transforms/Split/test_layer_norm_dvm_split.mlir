// RUN: mfusion-opt %s --fuse-layer-norm-dvm --mfuse-dvm-cluster --split | FileCheck %s --check-prefix=FWD
// RUN: mfusion-opt %s --split | FileCheck %s --check-prefix=BWD

module {
  // Exclusive mean stays in the matcher island; one fused kernel after cluster+split.
  // FWD-LABEL: func @test_layer_norm_dvm_split_fwd
  // FWD-COUNT-1: mfuse.fused
  // FWD: mfuse.sqrt
  // FWD: mfuse.reduce_sum
  // FWD: mfuse.yield
  // FWD-DAG: fusion_type = "dvm"
  // FWD-DAG: mfusion.dvm_fuse_group = "layer_norm#0"
  // FWD-DAG: mfusion.dvm_fuse_kind = "layer_norm"
  // FWD-DAG: mfusion.dvm_fuse_role = "member"
  // FWD-NOT: mfuse.aclnn
  func.func @test_layer_norm_dvm_split_fwd(%x: tensor<2x4xf32>, %gamma: tensor<4xf32>, %beta: tensor<4xf32>,
      %var: tensor<2x1xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
    %c4 = mfuse.constant dense<4.000000e+00> : tensor<f32, {is_scalar = ""}>
    %c_eps = mfuse.constant dense<9.99999974E-6> : tensor<f32, {is_scalar = ""}>
    %sqrt = mfuse.sqrt %var : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %rstd = mfuse.add %sqrt, %c_eps : (tensor<2x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1xf32>
    %sum = mfuse.reduce_sum %x {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
    %mean = mfuse.div %sum, %c4 : (tensor<2x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1xf32>
    %mean_bc = mfuse.broadcast_to %mean : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %sub = mfuse.sub %x, %mean_bc : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %gamma_bc = mfuse.broadcast_to %gamma : (tensor<4xf32>) -> tensor<2x4xf32>
    %mul = mfuse.mul %gamma_bc, %sub : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %rstd_bc = mfuse.broadcast_to %rstd : (tensor<2x1xf32>) -> tensor<2x4xf32>
    %div = mfuse.div %mul, %rstd_bc : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %beta_bc = mfuse.broadcast_to %beta : (tensor<4xf32>) -> tensor<2x4xf32>
    %out = mfuse.add %div, %beta_bc : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %sub, %out : tensor<2x4xf32>, tensor<2x4xf32>
  }

  // BWD-LABEL: func @test_layer_norm_dvm_split_bwd
  // BWD-COUNT-2: mfuse.fused
  // BWD: mfuse.reduce_sum
  func.func @test_layer_norm_dvm_split_bwd(%x: tensor<2x4x8xf32>, %saved: tensor<2x4x8xf32>,
      %rstd: tensor<2x1x1xf32>) -> tensor<1x1x8xf32> {
    %eps = mfuse.constant dense<1.000000e-06> : tensor<f32, {is_scalar = ""}>
    %c8 = mfuse.constant dense<8.000000e+00> : tensor<f32, {is_scalar = ""}>
    %body = mfuse.fused %x, %rstd, %c8 {
      fusion_type = "dvm",
      mfusion.dvm_fuse_kind = "layer_norm",
      mfusion.dvm_fuse_group = "layer_norm#0",
      mfusion.dvm_fuse_role = "member"
    } : (tensor<2x4x8xf32>, tensor<2x1x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x4x8xf32> {
    ^bb0(%arg0: tensor<2x4x8xf32>, %arg1: tensor<2x1x1xf32>, %arg2: tensor<f32, {is_scalar = ""}>):
      %sum = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = true}
          : (tensor<2x4x8xf32>) -> tensor<2x1x1xf32>
      %mean = mfuse.div %sum, %arg2
          : (tensor<2x1x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1x1xf32>
      %mean_bc = mfuse.broadcast_to %mean : (tensor<2x1x1xf32>) -> tensor<2x4x8xf32>
      %sub = mfuse.sub %arg0, %mean_bc : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
      %rstd_bc = mfuse.broadcast_to %arg1 : (tensor<2x1x1xf32>) -> tensor<2x4x8xf32>
      %div = mfuse.div %sub, %rstd_bc : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
      mfuse.yield %div : tensor<2x4x8xf32>
    }
    %vec = mfuse.fused %body, %saved, %rstd, %eps {
      fusion_type = "dvm",
      mfusion.dvm_fuse_kind = "layer_norm",
      mfusion.dvm_fuse_group = "layer_norm#1",
      mfusion.dvm_fuse_role = "member"
    } : (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<2x1x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<1x1x8xf32> {
    ^bb0(%arg3: tensor<2x4x8xf32>, %arg4: tensor<2x4x8xf32>, %arg5: tensor<2x1x1xf32>,
         %arg6: tensor<f32, {is_scalar = ""}>):
      %add_eps = mfuse.add %arg5, %arg6
          : (tensor<2x1x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1x1xf32>
      %add_bc = mfuse.broadcast_to %add_eps : (tensor<2x1x1xf32>) -> tensor<2x4x8xf32>
      %div2 = mfuse.div %arg3, %add_bc : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
      %mul_s = mfuse.mul %div2, %arg4 : (tensor<2x4x8xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
      %sum = mfuse.reduce_sum %mul_s {dimensions = [0, 1], keepdim = true}
          : (tensor<2x4x8xf32>) -> tensor<1x1x8xf32>
      mfuse.yield %sum : tensor<1x1x8xf32>
    }
    return %vec : tensor<1x1x8xf32>
  }

}
