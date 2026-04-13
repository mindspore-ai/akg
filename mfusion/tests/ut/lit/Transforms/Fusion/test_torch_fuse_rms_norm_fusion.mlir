// RUN: mfusion-opt %s --torch-fuse-rms-norm | FileCheck %s

module {
  // CHECK-LABEL: func @test_rms_norm_fusion_complete
  // Complete RMSNorm formula: pow(x, 2) -> mean -> add(eps) -> rsqrt -> mul(x, rsqrt) -> mul(gamma)
  func.func @test_rms_norm_fusion_complete(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05

    // Step 1: pow(x, 2)
    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>

    // Step 2: mean(pow, dim=-1, keepdim=True)
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>

    // Step 3: add(mean, eps)
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>

    // Step 4: rsqrt(add)
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>

    // Step 5: mul(x, rsqrt) - normalization
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>

    // Step 6: mul(norm, gamma) - scaling
    %out = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    // Torch prints multi-result torch.operator as "%N:2 = ..." with plain "(...)" operands.
    // CHECK: {{.*}}:2 = torch.operator "torch.npu.npu_rms_norm"({{[^)]*}}) : (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float) -> (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32>)
    // CHECK-NOT: torch.aten.pow.Tensor_Scalar
    // CHECK-NOT: torch.aten.mean.dim
    // CHECK-NOT: torch.aten.add.Scalar
    // CHECK-NOT: torch.aten.rsqrt
    // CHECK-NOT: torch.aten.mul.Tensor

    return %out : !torch.vtensor<[2,4],f32>
  }

  // CHECK-LABEL: func @test_rms_norm_inference_second_fused_result_unused
  //
  // Minimal inference-style RmsNorm: only the final mul(norm, gamma) is visible to consumers
  // (e.g. return). aten.rsqrt has exactly one use — feeding mul(x, rsqrt) inside the chain.
  //
  // Why fused#1 is not "rewired" in the IR:
  // - torch-fuse-rms-norm materializes torch.npu.npu_rms_norm with two results (y, rstd) for
  //   parity with aclnn/npu and for training graphs where rsqrt may have multiple uses.
  // - It replaces the final output mul with fused#0. It runs replaceAllUsesWith(rsqrt, fused#1)
  //   only when rsqrt has more than one use. Here rsqrt has one use, so nothing is linked to
  //   fused#1; after the pass erases the decomposed ops, fused#1 has no users. That is expected,
  //   not a missing replacement.
  //
  func.func @test_rms_norm_inference_second_fused_result_unused(%x: !torch.vtensor<[1,2],f32>,
      %gamma: !torch.vtensor<[1,2],f32>) -> !torch.vtensor<[1,2],f32> {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05
    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[1,2],f32>, !torch.int -> !torch.vtensor<[1,2],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[1,2],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,2,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[1,2,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,2,1],f32>
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[1,2,1],f32> -> !torch.vtensor<[1,2,1],f32>
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[1,2],f32>, !torch.vtensor<[1,2,1],f32> -> !torch.vtensor<[1,2],f32>
    %out = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[1,2],f32>, !torch.vtensor<[1,2],f32> -> !torch.vtensor<[1,2],f32>

    // Two-result op; downstream of this function only needs y (result #0).
    // CHECK: {{.*}}:2 = torch.operator "torch.npu.npu_rms_norm"({{[^)]*}}) : (!torch.vtensor<[1,2],f32>, !torch.vtensor<[1,2],f32>, !torch.float) -> (!torch.vtensor<[1,2],f32>, !torch.vtensor<[1,2,1],f32>)
    // CHECK: return %{{[0-9]+}}#0 : !torch.vtensor<[1,2],f32>

    return %out : !torch.vtensor<[1,2],f32>
  }

  // CHECK-LABEL: func @test_rms_norm_fusion_mixed_precision
  // Mixed-precision RmsNorm with dtype casts
  func.func @test_rms_norm_fusion_mixed_precision(%x_bf16: !torch.vtensor<[1,4,8],bf16>, %gamma: !torch.vtensor<[8],bf16>) -> !torch.vtensor<[1,4,8],bf16> {
    %int6 = torch.constant.int 6
    %int15 = torch.constant.int 15
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 9.999990e-07

    // Cast bf16 to f32 for computation
    %x_f32 = torch.operator "torch.npu._npu_dtype_cast"(%x_bf16, %int6) : (!torch.vtensor<[1,4,8],bf16>, !torch.int) -> !torch.vtensor<[1,4,8],f32>

    // Step 1: pow(x, 2)
    %pow = torch.aten.pow.Tensor_Scalar %x_f32, %int2 : !torch.vtensor<[1,4,8],f32>, !torch.int -> !torch.vtensor<[1,4,8],f32>

    // Step 2: mean(pow, dim=-1, keepdim=True)
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[1,4,8],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,4,1],f32>

    // Step 3: add(mean, eps)
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[1,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1,4,1],f32>

    // Step 4: rsqrt(add)
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[1,4,1],f32> -> !torch.vtensor<[1,4,1],f32>

    // Step 5: mul(x, rsqrt) - normalization
    %norm = torch.aten.mul.Tensor %x_f32, %rsqrt : !torch.vtensor<[1,4,8],f32>, !torch.vtensor<[1,4,1],f32> -> !torch.vtensor<[1,4,8],f32>

    // Cast f32 back to bf16
    %norm_bf16 = torch.operator "torch.npu._npu_dtype_cast"(%norm, %int15) : (!torch.vtensor<[1,4,8],f32>, !torch.int) -> !torch.vtensor<[1,4,8],bf16>

    // Step 6: mul(norm, gamma) - scaling
    %out = torch.aten.mul.Tensor %norm_bf16, %gamma : !torch.vtensor<[1,4,8],bf16>, !torch.vtensor<[8],bf16> -> !torch.vtensor<[1,4,8],bf16>

    // CHECK: {{.*}}:2 = torch.operator "torch.npu.npu_rms_norm"({{[^)]*}}) : (!torch.vtensor<[1,4,8],bf16>, !torch.vtensor<[8],bf16>, !torch.float) -> (!torch.vtensor<[1,4,8],bf16>, !torch.vtensor<[1,4,1],f32>)
    // CHECK-NOT: torch.aten.pow.Tensor_Scalar
    // CHECK-NOT: torch.aten.mean.dim
    // CHECK-NOT: torch.aten.add.Scalar
    // CHECK-NOT: torch.aten.rsqrt
    // CHECK-NOT: torch.aten.mul.Tensor

    return %out : !torch.vtensor<[1,4,8],bf16>
  }

  // CHECK-LABEL: func @test_rms_norm_no_fusion_wrong_power
  // Should not fuse when power is not 2
  func.func @test_rms_norm_no_fusion_wrong_power(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
    %int3 = torch.constant.int 3  // Wrong power
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05

    %pow = torch.aten.pow.Tensor_Scalar %x, %int3 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>
    %out = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    // Should not fuse because power is 3, not 2
    // CHECK: torch.aten.pow.Tensor_Scalar
    // CHECK: torch.aten.mean.dim
    // CHECK: torch.aten.add.Scalar
    // CHECK: torch.aten.rsqrt
    // CHECK: torch.aten.mul.Tensor
    // CHECK-NOT: torch.operator "torch.npu.npu_rms_norm"

    return %out : !torch.vtensor<[2,4],f32>
  }

  // CHECK-LABEL: func @test_rms_norm_no_fusion_wrong_add_alpha
  // Should not fuse when add alpha is not 1
  func.func @test_rms_norm_no_fusion_wrong_add_alpha(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %two = torch.constant.int 2  // Wrong alpha
    %eps = torch.constant.float 1.000000e-05

    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %two : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>
    %out = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    // Should not fuse because add alpha is 2, not 1
    // CHECK: torch.aten.pow.Tensor_Scalar
    // CHECK: torch.aten.mean.dim
    // CHECK: torch.aten.add.Scalar
    // CHECK: torch.aten.rsqrt
    // CHECK: torch.aten.mul.Tensor
    // CHECK-NOT: torch.operator "torch.npu.npu_rms_norm"

    return %out : !torch.vtensor<[2,4],f32>
  }

  // CHECK-LABEL: func @test_rms_norm_no_fusion_multiple_norm_uses
  // Should not fuse when norm has multiple uses
  func.func @test_rms_norm_no_fusion_multiple_norm_uses(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>) {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05

    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>
    
    // Multiple uses of norm - should prevent fusion
    %out1 = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    %out2 = torch.aten.mul.Tensor %gamma, %norm : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    // Should not fuse because norm has multiple uses
    // CHECK: torch.aten.pow.Tensor_Scalar
    // CHECK: torch.aten.mean.dim
    // CHECK: torch.aten.add.Scalar
    // CHECK: torch.aten.rsqrt
    // CHECK: torch.aten.mul.Tensor
    // CHECK-NOT: torch.operator "torch.npu.npu_rms_norm"

    return %out1, %out2 : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>
  }

  // CHECK-LABEL: func @test_rms_norm_training_graph
  // Training graph: rsqrt result has multiple uses (for backward pass)
  func.func @test_rms_norm_training_graph(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32>) {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05

    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>
    %out = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    // Training graph: return both output and rsqrt for backward pass
    // CHECK: {{.*}}:2 = torch.operator "torch.npu.npu_rms_norm"({{[^)]*}}) : (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float) -> (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32>)
    // Torch prints return types as a comma-separated list without outer parens.
    // CHECK: return %[[ID:[0-9]+]]#0, %[[ID]]#1 : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32>

    return %out, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32>
  }

  // CHECK-LABEL: func @test_rms_norm_fusion_reciprocal_sqrt
  // reciprocal(sqrt(add)) is mathematically rsqrt(add) for positive add.
  func.func @test_rms_norm_fusion_reciprocal_sqrt(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05
    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>
    %sqrt = torch.aten.sqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>
    %scale = torch.aten.reciprocal %sqrt : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>
    %norm = torch.aten.mul.Tensor %x, %scale : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>
    %out = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    // CHECK: {{.*}}:2 = torch.operator "torch.npu.npu_rms_norm"({{[^)]*}}) : (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float) -> (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32>)
    // CHECK-NOT: torch.aten.sqrt
    // CHECK-NOT: torch.aten.reciprocal
    // CHECK-NOT: torch.aten.rsqrt

    return %out : !torch.vtensor<[2,4],f32>
  }

  // CHECK-LABEL: func @test_rms_norm_fusion_pow_neg_half
  // pow(add, -0.5) matches rsqrt(add) for positive add.
  func.func @test_rms_norm_fusion_pow_neg_half(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05
    %nh = torch.constant.float -5.000000e-01
    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>
    %scale = torch.aten.pow.Tensor_Scalar %add, %nh : !torch.vtensor<[2,4,1],f32>, !torch.float -> !torch.vtensor<[2,4,1],f32>
    %norm = torch.aten.mul.Tensor %x, %scale : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>
    %out = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    // CHECK: {{.*}}:2 = torch.operator "torch.npu.npu_rms_norm"({{[^)]*}}) : (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float) -> (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32>)
    // CHECK-NOT: torch.aten.pow.Tensor_Scalar
    // CHECK-NOT: torch.aten.rsqrt

    return %out : !torch.vtensor<[2,4],f32>
  }

  // CHECK-LABEL: func @test_rms_norm_fusion_mul_xx_variance
  // mean.dim(mul(x,x)) instead of mean.dim(pow(x,2))
  func.func @test_rms_norm_fusion_mul_xx_variance(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05
    %xx = torch.aten.mul.Tensor %x, %x : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %xx, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>
    %rsqrt = torch.aten.rsqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>
    %norm = torch.aten.mul.Tensor %x, %rsqrt : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>
    %out = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    // CHECK: {{.*}}:2 = torch.operator "torch.npu.npu_rms_norm"({{[^)]*}}) : (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float) -> (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32>)
    // CHECK-NOT: torch.aten.pow.Tensor_Scalar
    // CHECK-NOT: torch.aten.mean.dim
    // CHECK-NOT: torch.aten.mul.Tensor

    return %out : !torch.vtensor<[2,4],f32>
  }

  // CHECK-LABEL: func @test_rms_norm_fusion_div_ones_over_sqrt
  // div(ones_like(sqrt(add)), sqrt(add)) == 1/sqrt(add)
  func.func @test_rms_norm_fusion_div_ones_over_sqrt(%x: !torch.vtensor<[2,4],f32>, %gamma: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
    %int2 = torch.constant.int 2
    %int_neg1 = torch.constant.int -1
    %true = torch.constant.bool true
    %none = torch.constant.none
    %one = torch.constant.int 1
    %eps = torch.constant.float 1.000000e-05
    %pow = torch.aten.pow.Tensor_Scalar %x, %int2 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
    %dims = torch.prim.ListConstruct %int_neg1 : (!torch.int) -> !torch.list<int>
    %mean = torch.aten.mean.dim %pow, %dims, %true, %none : !torch.vtensor<[2,4],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[2,4,1],f32>
    %add = torch.aten.add.Scalar %mean, %eps, %one : !torch.vtensor<[2,4,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[2,4,1],f32>
    %sqrt = torch.aten.sqrt %add : !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>
    %ones = torch.aten.ones_like %sqrt, %none, %none, %none, %none, %none : !torch.vtensor<[2,4,1],f32>, !torch.none, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[2,4,1],f32>
    %scale = torch.aten.div.Tensor %ones, %sqrt : !torch.vtensor<[2,4,1],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4,1],f32>
    %norm = torch.aten.mul.Tensor %x, %scale : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32> -> !torch.vtensor<[2,4],f32>
    %out = torch.aten.mul.Tensor %norm, %gamma : !torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>

    // CHECK: {{.*}}:2 = torch.operator "torch.npu.npu_rms_norm"({{[^)]*}}) : (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4],f32>, !torch.float) -> (!torch.vtensor<[2,4],f32>, !torch.vtensor<[2,4,1],f32>)
    // CHECK-NOT: torch.aten.ones_like
    // CHECK-NOT: torch.aten.div.Tensor
    // CHECK-NOT: torch.aten.sqrt

    return %out : !torch.vtensor<[2,4],f32>
  }
}