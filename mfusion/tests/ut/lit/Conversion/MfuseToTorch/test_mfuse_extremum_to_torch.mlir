// RUN: mfusion-opt %s --convert-mfuse-to-torch | FileCheck %s

// CHECK-LABEL: func.func @test_maximum_tensor
// CHECK: torch.aten.maximum
// CHECK-NOT: torch.aten.clamp_min
func.func @test_maximum_tensor(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,4],f32> to tensor<2x4xf32>
  %1 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,4],f32> to tensor<2x4xf32>
  %2 = mfuse.maximum %0, %1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<2x4xf32> to !torch.vtensor<[2,4],f32>
  return %3 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @test_maximum_scalar_rhs
// CHECK: %[[MIN:.*]] = torch.constant.float
// CHECK: torch.aten.clamp_min %arg0, %[[MIN]]
// CHECK-NOT: torch.aten.maximum
func.func @test_maximum_scalar_rhs(%arg0: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %min = mfuse.constant dense<0.0> : tensor<f32, {is_scalar = ""}>
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,4],f32> to tensor<2x4xf32>
  %1 = mfuse.maximum %0, %min : (tensor<2x4xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x4xf32>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<2x4xf32> to !torch.vtensor<[2,4],f32>
  return %2 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @test_minimum_tensor
// CHECK: torch.aten.minimum
// CHECK-NOT: torch.aten.clamp_max
func.func @test_minimum_tensor(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,4],f32> to tensor<2x4xf32>
  %1 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,4],f32> to tensor<2x4xf32>
  %2 = mfuse.minimum %0, %1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<2x4xf32> to !torch.vtensor<[2,4],f32>
  return %3 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @test_minimum_scalar_rhs
// CHECK: %[[MAX:.*]] = torch.constant.float
// CHECK: torch.aten.clamp_max %arg0, %[[MAX]]
// CHECK-NOT: torch.aten.minimum
func.func @test_minimum_scalar_rhs(%arg0: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %max = mfuse.constant dense<6.0> : tensor<f32, {is_scalar = ""}>
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,4],f32> to tensor<2x4xf32>
  %1 = mfuse.minimum %0, %max : (tensor<2x4xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2x4xf32>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<2x4xf32> to !torch.vtensor<[2,4],f32>
  return %2 : !torch.vtensor<[2,4],f32>
}
