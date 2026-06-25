// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=tanh" --convert-mfuse-to-torch="kernel-generator=dvm" | FileCheck %s

// Tanh decompose uses mfuse.maximum/minimum; roundtrip must emit clamp_min/clamp_max, not isnan/where.
func.func @test_tanh_decompose_roundtrip(%arg0: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[4,4],f32> to tensor<4x4xf32>
  %1 = mfuse.aclnn.tanh %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<4x4xf32> to !torch.vtensor<[4,4],f32>
  return %2 : !torch.vtensor<[4,4],f32>
}

// CHECK-LABEL: func.func @test_tanh_decompose_roundtrip
// CHECK-NOT: torch.aten.tanh
// CHECK-NOT: torch.aten.isnan
// CHECK-NOT: torch.aten.where
// CHECK-NOT: mfuse.aclnn.clamp
// CHECK: torch.aten.clamp_min
// CHECK: torch.aten.clamp_max
