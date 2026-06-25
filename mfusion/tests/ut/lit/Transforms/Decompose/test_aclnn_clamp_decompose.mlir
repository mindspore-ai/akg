// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=clamp" | FileCheck %s

func.func @test_aclnn_clamp(%arg0: tensor<4x4xf32>, %min: tensor<f32, {is_scalar = ""}>, %max: tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32> {
  %0 = mfuse.aclnn.clamp %arg0, %min, %max : (tensor<4x4xf32>, tensor<f32, {is_scalar = ""}>, tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.clamp
  // CHECK: mfuse.maximum
  // CHECK: mfuse.minimum
}

// Roundtrip: decomposed clamp must lower to clamp_min/clamp_max, not isnan/where.
// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=clamp" --convert-mfuse-to-torch="kernel-generator=dvm" | FileCheck %s --check-prefix=ROUNDTRIP

func.func @test_aclnn_clamp_roundtrip(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[4,4],f32> to tensor<4x4xf32>
  %min_t = mfuse.constant dense<-8.800000e+00> : tensor<f32, {is_scalar = ""}>
  %max_t = mfuse.constant dense<8.800000e+00> : tensor<f32, {is_scalar = ""}>
  %1 = mfuse.aclnn.clamp %0, %min_t, %max_t : (tensor<4x4xf32>, tensor<f32, {is_scalar = ""}>, tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<4x4xf32> to !torch.vtensor<[4,4],f32>
  return %2 : !torch.vtensor<[4,4],f32>
}

// ROUNDTRIP-LABEL: func.func @test_aclnn_clamp_roundtrip
// ROUNDTRIP-NOT: torch.aten.isnan
// ROUNDTRIP-NOT: torch.aten.where
// ROUNDTRIP-NOT: mfuse.aclnn.clamp
// ROUNDTRIP: torch.aten.clamp_min
// ROUNDTRIP: torch.aten.clamp_max
