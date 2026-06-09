// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse | FileCheck %s

// CHECK-LABEL: func.func @main_clamp_min_scalar
// CHECK: mfuse.maximum {{.*}} : (tensor<2x4xf32>, tensor<f64, {{.*}}>) -> tensor<2x4xf32>
// CHECK-NOT: torch.aten.clamp_min
func.func @main_clamp_min_scalar(%arg0: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %min = torch.constant.float 0.000000e+00
  %0 = torch.aten.clamp_min %arg0, %min : !torch.vtensor<[2,4],f32>, !torch.float -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @main_clamp_min_scalar_f16
// CHECK: mfuse.maximum {{.*}} : (tensor<2x4xf16>, tensor<f64, {{.*}}>) -> tensor<2x4xf16>
// CHECK-NOT: torch.aten.clamp_min
func.func @main_clamp_min_scalar_f16(%arg0: !torch.vtensor<[2,4],f16>) -> !torch.vtensor<[2,4],f16> attributes {torch.assume_strict_symbolic_shapes} {
  %min = torch.constant.float 0.000000e+00
  %0 = torch.aten.clamp_min %arg0, %min : !torch.vtensor<[2,4],f16>, !torch.float -> !torch.vtensor<[2,4],f16>
  return %0 : !torch.vtensor<[2,4],f16>
}

// CHECK-LABEL: func.func @main_clamp_min_tensor
// CHECK: mfuse.maximum {{.*}} : (tensor<2x4xf16>, tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NOT: torch.aten.clamp_min.Tensor
func.func @main_clamp_min_tensor(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.clamp_min.Tensor %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}
