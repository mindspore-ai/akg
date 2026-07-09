// RUN: mfusion-opt %s --fuse-safe-softmax-dvm | FileCheck %s

module {
  // Albert-style reused zero buffer: layer-1 fused op result #0 feeds later selects.
  // CHECK-LABEL: func @test_reuse_safe_softmax_zero_buffer
  // CHECK-COUNT-2: mfuse.fused
  // CHECK-NOT: torch.aten.any.dim
  // CHECK-NOT: mfuse.softmax
  // CHECK: mfuse.eq
  // CHECK: mfuse.reduce_max
  // CHECK-SAME: tensor<4x12x512x512xi1>
  func.func @test_reuse_safe_softmax_zero_buffer(%scores1: tensor<4x12x512x512xf32>,
      %scores2: tensor<4x12x512x512xf32>, %mask1: tensor<4x12x512x1xi1>,
      %mask2: tensor<4x12x512x1xi1>) -> (tensor<4x12x512x512xf32>, tensor<4x12x512x512xf32>) {
    %neg = mfuse.constant dense<0xFF800000> : tensor<f64, {is_scalar = ""}>
    %softmax1 = mfuse.softmax %scores1 {dim = -1 : i64, half_to_float = false}
        : (tensor<4x12x512x512xf32>) -> tensor<4x12x512x512xf32>
    %cst = mfuse.constant dense<0.000000e+00> : tensor<f32, {is_scalar = ""}>
    %zero = mfuse.full %cst : (tensor<f32, {is_scalar = ""}>) -> tensor<4x12x512x512xf32>
    %not1 = mfuse.logical_not %mask1 : (tensor<4x12x512x1xi1>) -> tensor<4x12x512x1xi1>
    %sel1 = mfuse.select %not1, %zero, %softmax1
        : (tensor<4x12x512x1xi1>, tensor<4x12x512x512xf32>, tensor<4x12x512x512xf32>) -> tensor<4x12x512x512xf32>

    %eq = mfuse.eq %scores2, %neg : (tensor<4x12x512x512xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x12x512x512xi1>
    %eq_not = mfuse.logical_not %eq : (tensor<4x12x512x512xi1>) -> tensor<4x12x512x512xi1>
    %eq_torch = builtin.unrealized_conversion_cast %eq_not
        : tensor<4x12x512x512xi1> to !torch.vtensor<[4,12,512,512],i1>
    %dim = torch.constant.int -1
    %true = torch.constant.bool true
    %any_torch = torch.aten.any.dim %eq_torch, %dim, %true
        : !torch.vtensor<[4,12,512,512],i1>, !torch.int, !torch.bool -> !torch.vtensor<[4,12,512,1],i1>
    %any = builtin.unrealized_conversion_cast %any_torch : !torch.vtensor<[4,12,512,1],i1> to tensor<4x12x512x1xi1>
    %not2 = mfuse.logical_not %any : (tensor<4x12x512x1xi1>) -> tensor<4x12x512x1xi1>
    %softmax2 = mfuse.softmax %scores2 {dim = -1 : i64, half_to_float = false}
        : (tensor<4x12x512x512xf32>) -> tensor<4x12x512x512xf32>
    %sel2 = mfuse.select %not2, %zero, %softmax2
        : (tensor<4x12x512x1xi1>, tensor<4x12x512x512xf32>, tensor<4x12x512x512xf32>) -> tensor<4x12x512x512xf32>
    return %sel1, %sel2 : tensor<4x12x512x512xf32>, tensor<4x12x512x512xf32>
  }
}
