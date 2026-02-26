// RUN: mfusion-opt %s --convert-mfuse-to-torch | FileCheck %s

// CHECK-LABEL: func.func @test_div
func.func @test_div(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: torch.aten.div.Tensor
  %0 = mfuse.div %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @test_mul
func.func @test_mul(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: torch.aten.mul.Tensor
  %0 = mfuse.mul %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @test_eq
func.func @test_eq(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.eq.Tensor
  %0 = mfuse.eq %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_ge
func.func @test_ge(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.ge.Tensor
  %0 = mfuse.ge %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_gt
func.func @test_gt(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.gt.Tensor
  %0 = mfuse.gt %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_le
func.func @test_le(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.le.Tensor
  %0 = mfuse.le %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_lt
func.func @test_lt(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.lt.Tensor
  %0 = mfuse.lt %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_ne
func.func @test_ne(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.ne.Tensor
  %0 = mfuse.ne %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_reduce_sum
func.func @test_reduce_sum(%arg0: tensor<2x2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = false, dtype = f32} : (tensor<2x2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @test_reduce_sum_f32_to_f64
func.func @test_reduce_sum_f32_to_f64(%arg0: tensor<2x2xf32>) -> tensor<2xf64> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 7
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = mfuse.reduce_sum %arg0 {dimensions = [0], keepdim = false, dtype = f64} : (tensor<2x2xf32>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// CHECK-LABEL: func.func @test_reduce_sum_i32
func.func @test_reduce_sum_i32(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 3
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = false, dtype = i32} : (tensor<2x3xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: func.func @test_reduce_sum_keepdim
func.func @test_reduce_sum_keepdim(%arg0: tensor<2x2xf32>) -> tensor<2x1xf32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool true
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = true, dtype = f32} : (tensor<2x2xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// CHECK-LABEL: func.func @test_reduce_sum_all_dims
func.func @test_reduce_sum_all_dims(%arg0: tensor<2x3xf32>) -> tensor<f32> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = mfuse.reduce_sum %arg0 {dimensions = [0, 1], keepdim = false, dtype = f32} : (tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_dtype_none
func.func @test_reduce_sum_dtype_none(%arg0: tensor<2x2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.none
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = false, dtype = none} : (tensor<2x2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @test_cast
func.func @test_cast(%arg0: tensor<2x2xf32>) -> tensor<2x2xf16> {
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 5
  // CHECK: torch.aten.to.dtype %{{.*}}, %[[DTYPE]]
  %0 = mfuse.cast %arg0 {dtype = f16} : (tensor<2x2xf32>) -> tensor<2x2xf16>
  return %0 : tensor<2x2xf16>
}

// CHECK-LABEL: func.func @test_cast_i32_to_f64
func.func @test_cast_i32_to_f64(%arg0: tensor<2x2xi32>) -> tensor<2x2xf64> {
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 7
  // CHECK: torch.aten.to.dtype %{{.*}}, %[[DTYPE]]
  %0 = mfuse.cast %arg0 {dtype = f64} : (tensor<2x2xi32>) -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}

// CHECK-LABEL: func.func @test_permute_general
func.func @test_permute_general(%arg0: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  // CHECK-DAG: %[[D0:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[D1:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[D2:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[D0]], %[[D1]], %[[D2]]
  // CHECK: torch.aten.permute %{{.*}}, %[[DIMS]]
  %0 = mfuse.permute %arg0, [2, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>
}

// CHECK-LABEL: func.func @test_permute_identity
func.func @test_permute_identity(%arg0: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // CHECK-DAG: %[[D0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[D1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[D2:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[D0]], %[[D1]], %[[D2]]
  // CHECK: torch.aten.permute %{{.*}}, %[[DIMS]]
  %0 = mfuse.permute %arg0, [0, 1, 2] : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @test_permute_negative_dims
func.func @test_permute_negative_dims(%arg0: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  // CHECK-DAG: %[[D0:.*]] = torch.constant.int -1
  // CHECK-DAG: %[[D1:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[D2:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[D0]], %[[D1]], %[[D2]]
  // CHECK: torch.aten.permute %{{.*}}, %[[DIMS]]
  %0 = mfuse.permute %arg0, [-1, 0, 1] : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>
}

// ============================================================================
// Test cases for mfuse.reshape -> torch.aten.reshape
// ============================================================================

// CHECK-LABEL: func.func @test_reshape_2d_to_2d
func.func @test_reshape_2d_to_2d(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]]
  // CHECK: torch.aten.reshape %{{.*}}, %[[DIMS]]
  %0 = mfuse.reshape %arg0 : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func.func @test_reshape_flatten
func.func @test_reshape_flatten(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 6
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]]
  // CHECK: torch.aten.reshape %{{.*}}, %[[DIMS]]
  %0 = mfuse.reshape %arg0 : (tensor<2x3xf32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL: func.func @test_reshape_expand
func.func @test_reshape_expand(%arg0: tensor<6xf32>) -> tensor<2x3xf32> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]]
  // CHECK: torch.aten.reshape %{{.*}}, %[[DIMS]]
  %0 = mfuse.reshape %arg0 : (tensor<6xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @test_reshape_3d
func.func @test_reshape_3d(%arg0: tensor<2x3x4xf32>) -> tensor<4x2x3xf32> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 4
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[DIM2:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]], %[[DIM2]]
  // CHECK: torch.aten.reshape %{{.*}}, %[[DIMS]]
  %0 = mfuse.reshape %arg0 : (tensor<2x3x4xf32>) -> tensor<4x2x3xf32>
  return %0 : tensor<4x2x3xf32>
}

// CHECK-LABEL: func.func @test_reshape_i32
func.func @test_reshape_i32(%arg0: tensor<2x3xi32>) -> tensor<6xi32> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 6
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]]
  // CHECK: torch.aten.reshape %{{.*}}, %[[DIMS]]
  %0 = mfuse.reshape %arg0 : (tensor<2x3xi32>) -> tensor<6xi32>
  return %0 : tensor<6xi32>
}

// CHECK-LABEL: func.func @test_reshape_f16
func.func @test_reshape_f16(%arg0: tensor<2x3xf16>) -> tensor<6xf16> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 6
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]]
  // CHECK: torch.aten.reshape %{{.*}}, %[[DIMS]]
  %0 = mfuse.reshape %arg0 : (tensor<2x3xf16>) -> tensor<6xf16>
  return %0 : tensor<6xf16>
}

// CHECK-LABEL: func.func @test_reshape_f64
func.func @test_reshape_f64(%arg0: tensor<2x3xf64>) -> tensor<6xf64> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 6
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]]
  // CHECK: torch.aten.reshape %{{.*}}, %[[DIMS]]
  %0 = mfuse.reshape %arg0 : (tensor<2x3xf64>) -> tensor<6xf64>
  return %0 : tensor<6xf64>
}

// CHECK-LABEL: func.func @test_reshape_single_element
func.func @test_reshape_single_element(%arg0: tensor<1xf32>) -> tensor<1x1xf32> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]]
  // CHECK: torch.aten.reshape %{{.*}}, %[[DIMS]]
  %0 = mfuse.reshape %arg0 : (tensor<1xf32>) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}

// CHECK-LABEL: func.func @test_reshape_4d
func.func @test_reshape_4d(%arg0: tensor<2x3x4x5xf32>) -> tensor<6x20xf32> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 6
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 20
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]], %[[DIM1]]
  // CHECK: torch.aten.reshape %{{.*}}, %[[DIMS]]
  %0 = mfuse.reshape %arg0 : (tensor<2x3x4x5xf32>) -> tensor<6x20xf32>
  return %0 : tensor<6x20xf32>
}

// CHECK-LABEL: func.func @test_reshape_scalar_to_1d
func.func @test_reshape_scalar_to_1d(%arg0: tensor<f32>) -> tensor<1xf32> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]]
  // CHECK: torch.aten.reshape %{{.*}}, %[[DIMS]]
  %0 = mfuse.reshape %arg0 : (tensor<f32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}
