// RUN: mfusion-opt %s --convert-mfuse-to-torch | FileCheck %s

// CHECK-LABEL: func.func @test_add_scalar
// CHECK: torch.aten.add.Scalar
func.func @test_add_scalar(%arg0: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],f32> {
  %cst = arith.constant dense<3.0> : tensor<f32>
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %1 = mfuse.add %0, %cst : (tensor<2x3x32x32xf32>, tensor<f32>) -> tensor<2x3x32x32xf32>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<2x3x32x32xf32> to !torch.vtensor<[2,3,32,32],f32>
  return %2 : !torch.vtensor<[2,3,32,32],f32>
}

// CHECK-LABEL: func.func @test_mul_scalar
// CHECK: torch.aten.mul.Scalar
func.func @test_mul_scalar(%arg0: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],f32> {
  %cst = arith.constant dense<2.5> : tensor<f32>
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %1 = mfuse.mul %0, %cst : (tensor<2x3x32x32xf32>, tensor<f32>) -> tensor<2x3x32x32xf32>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<2x3x32x32xf32> to !torch.vtensor<[2,3,32,32],f32>
  return %2 : !torch.vtensor<[2,3,32,32],f32>
}

// CHECK-LABEL: func.func @test_sub_scalar
// CHECK: torch.aten.sub.Scalar
func.func @test_sub_scalar(%arg0: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],f32> {
  %cst = arith.constant dense<10.5> : tensor<f32>
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %1 = mfuse.sub %0, %cst : (tensor<2x3x32x32xf32>, tensor<f32>) -> tensor<2x3x32x32xf32>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<2x3x32x32xf32> to !torch.vtensor<[2,3,32,32],f32>
  return %2 : !torch.vtensor<[2,3,32,32],f32>
}

// CHECK-LABEL: func.func @test_div_scalar
// CHECK: torch.aten.div.Scalar
func.func @test_div_scalar(%arg0: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],f32> {
  %cst = arith.constant dense<2.0> : tensor<f32>
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %1 = mfuse.div %0, %cst : (tensor<2x3x32x32xf32>, tensor<f32>) -> tensor<2x3x32x32xf32>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<2x3x32x32xf32> to !torch.vtensor<[2,3,32,32],f32>
  return %2 : !torch.vtensor<[2,3,32,32],f32>
}

// CHECK-LABEL: func.func @test_gt_scalar
// CHECK: torch.aten.gt.Scalar
func.func @test_gt_scalar(%arg0: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],i1> {
  %cst = arith.constant dense<0.5> : tensor<f32>
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %1 = mfuse.gt %0, %cst : (tensor<2x3x32x32xf32>, tensor<f32>) -> tensor<2x3x32x32xi1>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<2x3x32x32xi1> to !torch.vtensor<[2,3,32,32],i1>
  return %2 : !torch.vtensor<[2,3,32,32],i1>
}

// CHECK-LABEL: func.func @test_lt_scalar
// CHECK: torch.aten.lt.Scalar
func.func @test_lt_scalar(%arg0: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],i1> {
  %cst = arith.constant dense<10.0> : tensor<f32>
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %1 = mfuse.lt %0, %cst : (tensor<2x3x32x32xf32>, tensor<f32>) -> tensor<2x3x32x32xi1>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<2x3x32x32xi1> to !torch.vtensor<[2,3,32,32],i1>
  return %2 : !torch.vtensor<[2,3,32,32],i1>
}

// CHECK-LABEL: func.func @test_eq_scalar
// CHECK: torch.aten.eq.Scalar
func.func @test_eq_scalar(%arg0: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],i1> {
  %cst = arith.constant dense<5.5> : tensor<f32>
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %1 = mfuse.eq %0, %cst : (tensor<2x3x32x32xf32>, tensor<f32>) -> tensor<2x3x32x32xi1>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<2x3x32x32xi1> to !torch.vtensor<[2,3,32,32],i1>
  return %2 : !torch.vtensor<[2,3,32,32],i1>
}

// CHECK-LABEL: func.func @test_add_tensor
// CHECK: torch.aten.add.Tensor
func.func @test_add_tensor(%arg0: !torch.vtensor<[2,3,32,32],f32>, %arg1: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],f32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %1 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %2 = mfuse.add %0, %1 : (tensor<2x3x32x32xf32>, tensor<2x3x32x32xf32>) -> tensor<2x3x32x32xf32>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<2x3x32x32xf32> to !torch.vtensor<[2,3,32,32],f32>
  return %3 : !torch.vtensor<[2,3,32,32],f32>
}

// CHECK-LABEL: func.func @test_mul_tensor
// CHECK: torch.aten.mul.Tensor
func.func @test_mul_tensor(%arg0: !torch.vtensor<[2,3,32,32],f32>, %arg1: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],f32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %1 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %2 = mfuse.mul %0, %1 : (tensor<2x3x32x32xf32>, tensor<2x3x32x32xf32>) -> tensor<2x3x32x32xf32>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<2x3x32x32xf32> to !torch.vtensor<[2,3,32,32],f32>
  return %3 : !torch.vtensor<[2,3,32,32],f32>
}

// CHECK-LABEL: func.func @test_sub_tensor
// CHECK: torch.aten.sub.Tensor
func.func @test_sub_tensor(%arg0: !torch.vtensor<[2,3,32,32],f32>, %arg1: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],f32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %1 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %2 = mfuse.sub %0, %1 : (tensor<2x3x32x32xf32>, tensor<2x3x32x32xf32>) -> tensor<2x3x32x32xf32>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<2x3x32x32xf32> to !torch.vtensor<[2,3,32,32],f32>
  return %3 : !torch.vtensor<[2,3,32,32],f32>
}

// CHECK-LABEL: func.func @test_div_tensor
// CHECK: torch.aten.div.Tensor
func.func @test_div_tensor(%arg0: !torch.vtensor<[2,3,32,32],f32>, %arg1: !torch.vtensor<[2,3,32,32],f32>) -> !torch.vtensor<[2,3,32,32],f32> {
  %0 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %1 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,3,32,32],f32> to tensor<2x3x32x32xf32>
  %2 = mfuse.div %0, %1 : (tensor<2x3x32x32xf32>, tensor<2x3x32x32xf32>) -> tensor<2x3x32x32xf32>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<2x3x32x32xf32> to !torch.vtensor<[2,3,32,32],f32>
  return %3 : !torch.vtensor<[2,3,32,32],f32>
}
