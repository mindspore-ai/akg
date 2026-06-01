// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --canonicalize | FileCheck %s --check-prefix=CONVERT
// RUN: mfusion-opt %s --decompose="pattern-type=AFTER_MANUAL_FUSION op-list=aclnnvar" | FileCheck %s --check-prefix=DECOMPOSE
// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --decompose="pattern-type=AFTER_MANUAL_FUSION op-list=aclnnvar" --decompose="pattern-type=AFTER_MANUAL_FUSION" --fuse-layer-norm-dvm --canonicalize | FileCheck %s --check-prefix=PIPE

module {
  // CONVERT-LABEL: func @test_convert_var_correction
  // CONVERT-NOT: torch.aten.var.correction
  // CONVERT: mfuse.aclnn.var
  func.func @test_convert_var_correction(%x: tensor<2x4xf32>) -> tensor<2x1xf32> {
    %torch_x = builtin.unrealized_conversion_cast %x : tensor<2x4xf32> to !torch.vtensor<[2,4],f32>
    %int1 = torch.constant.int 1
    %dim = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %float1 = torch.constant.float 1.000000e+00
    %true = torch.constant.bool true
    %torch_var = torch.aten.var.correction %torch_x, %dim, %float1, %true
        : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.float, !torch.bool -> !torch.vtensor<[2,1],f32>
    %var = builtin.unrealized_conversion_cast %torch_var : !torch.vtensor<[2,1],f32> to tensor<2x1xf32>
    return %var : tensor<2x1xf32>
  }

  // DECOMPOSE-LABEL: func @test_decompose_var
  // DECOMPOSE-NOT: mfuse.aclnn.var
  // DECOMPOSE: mfuse.reduce_mean
  // DECOMPOSE: mfuse.reduce_sum
  func.func @test_decompose_var(%x: tensor<2x4xf32>) -> tensor<2x1xf32> {
    %mean = mfuse.reduce_mean %x {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
    %var = mfuse.aclnn.var %x {dim = [1], correction = 1 : i64, keepdim = true}
        : (tensor<2x4xf32>) -> tensor<2x1xf32>
    return %var : tensor<2x1xf32>
  }

  // PIPE-LABEL: func @test_var_pipeline_and_tag
  // PIPE-NOT: mfuse.aclnn.var
  // PIPE-NOT: torch.aten.var.correction
  // PIPE-NOT: mfuse.reduce_mean
  // PIPE-COUNT-2: mfuse.reduce_sum
  // PIPE: mfusion.dvm_fuse_group = "layer_norm#0"
  // PIPE: mfusion.layer_norm_dvm
  func.func @test_var_pipeline_and_tag(%x: tensor<2x4xf32>, %gamma: tensor<4xf32>, %beta: tensor<4xf32>) -> tensor<2x4xf32> {
    %c_eps = mfuse.constant dense<9.99999974E-6> : tensor<f32, {is_scalar = ""}>
    %mean = mfuse.reduce_mean %x {dimensions = [1], keepdim = true} : (tensor<2x4xf32>) -> tensor<2x1xf32>
    %torch_x = builtin.unrealized_conversion_cast %x : tensor<2x4xf32> to !torch.vtensor<[2,4],f32>
    %int1 = torch.constant.int 1
    %dim = torch.prim.ListConstruct %int1 : (!torch.int) -> !torch.list<int>
    %float1 = torch.constant.float 1.000000e+00
    %true = torch.constant.bool true
    %torch_var = torch.aten.var.correction %torch_x, %dim, %float1, %true
        : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.float, !torch.bool -> !torch.vtensor<[2,1],f32>
    %var = builtin.unrealized_conversion_cast %torch_var : !torch.vtensor<[2,1],f32> to tensor<2x1xf32>
    %sqrt = mfuse.sqrt %var : (tensor<2x1xf32>) -> tensor<2x1xf32>
    %rstd = mfuse.add %sqrt, %c_eps : (tensor<2x1xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x1xf32>
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
}
