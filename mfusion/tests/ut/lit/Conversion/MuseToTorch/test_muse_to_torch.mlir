// RUN: mfusion-opt %s --convert-muse-to-torch | FileCheck %s

// CHECK-LABEL: func.func @test_div
func.func @test_div(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: torch.aten.div.Tensor
  %0 = muse.div %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @test_mul
func.func @test_mul(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: torch.aten.mul.Tensor
  %0 = muse.mul %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @test_eq
func.func @test_eq(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.eq.Tensor
  %0 = muse.eq %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_ge
func.func @test_ge(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.ge.Tensor
  %0 = muse.ge %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_gt
func.func @test_gt(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.gt.Tensor
  %0 = muse.gt %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_le
func.func @test_le(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.le.Tensor
  %0 = muse.le %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_lt
func.func @test_lt(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.lt.Tensor
  %0 = muse.lt %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_ne
func.func @test_ne(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.ne.Tensor
  %0 = muse.ne %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_reduce_sum
func.func @test_reduce_sum(%arg0: tensor<2x2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = muse.reduce_sum %arg0 {dimensions = [1], keepdim = false, dtype = f32} : (tensor<2x2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @test_reduce_sum_f32_to_f64
func.func @test_reduce_sum_f32_to_f64(%arg0: tensor<2x2xf32>) -> tensor<2xf64> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 7
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = muse.reduce_sum %arg0 {dimensions = [0], keepdim = false, dtype = f64} : (tensor<2x2xf32>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// CHECK-LABEL: func.func @test_reduce_sum_i32
func.func @test_reduce_sum_i32(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 3
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = muse.reduce_sum %arg0 {dimensions = [1], keepdim = false, dtype = i32} : (tensor<2x3xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: func.func @test_reduce_sum_keepdim
func.func @test_reduce_sum_keepdim(%arg0: tensor<2x2xf32>) -> tensor<2x1xf32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool true
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = muse.reduce_sum %arg0 {dimensions = [1], keepdim = true, dtype = f32} : (tensor<2x2xf32>) -> tensor<2x1xf32>
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
  %0 = muse.reduce_sum %arg0 {dimensions = [0, 1], keepdim = false, dtype = f32} : (tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_dtype_none
func.func @test_reduce_sum_dtype_none(%arg0: tensor<2x2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.none
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = muse.reduce_sum %arg0 {dimensions = [1], keepdim = false, dtype = none} : (tensor<2x2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @test_cast
func.func @test_cast(%arg0: tensor<2x2xf32>) -> tensor<2x2xf16> {
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 5
  // CHECK: torch.aten.to.dtype %{{.*}}, %[[DTYPE]]
  %0 = muse.cast %arg0 {dtype = f16} : (tensor<2x2xf32>) -> tensor<2x2xf16>
  return %0 : tensor<2x2xf16>
}

// CHECK-LABEL: func.func @test_cast_i32_to_f64
func.func @test_cast_i32_to_f64(%arg0: tensor<2x2xi32>) -> tensor<2x2xf64> {
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 7
  // CHECK: torch.aten.to.dtype %{{.*}}, %[[DTYPE]]
  %0 = muse.cast %arg0 {dtype = f64} : (tensor<2x2xi32>) -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}
