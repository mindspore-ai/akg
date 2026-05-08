// RUN: mfusion-opt %s --convert-mfuse-to-torch | FileCheck %s

// CHECK-LABEL: func.func @test_aclnn_batch_matmul_trans_x1
func.func @test_aclnn_batch_matmul_trans_x1(%arg0: tensor<80x64x204xf32>, %arg1: tensor<80x64x128xf32>) -> tensor<80x204x128xf32> {
  // CHECK-DAG: %[[BM1_C0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[BM1_C1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[BM1_C2:.*]] = torch.constant.int 2
  // CHECK: %[[BM1_PERM:.*]] = torch.prim.ListConstruct %[[BM1_C0]], %[[BM1_C2]], %[[BM1_C1]]
  // CHECK: %[[BM1_LHS:.*]] = torch.aten.permute %arg0, %[[BM1_PERM]]
  // CHECK: torch.aten.bmm %[[BM1_LHS]], %arg1
  %0 = mfuse.aclnn.batch_matmul %arg0, %arg1 {trans_x1 = true} : (tensor<80x64x204xf32>, tensor<80x64x128xf32>) -> tensor<80x204x128xf32>
  return %0 : tensor<80x204x128xf32>
}

// CHECK-LABEL: func.func @test_aclnn_batch_matmul_trans_x2
func.func @test_aclnn_batch_matmul_trans_x2(%arg0: tensor<80x204x64xf32>, %arg1: tensor<80x204x64xf32>) -> tensor<80x204x204xf32> {
  // CHECK-DAG: %[[BM_C0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[BM_C1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[BM_C2:.*]] = torch.constant.int 2
  // CHECK: %[[BM_PERM:.*]] = torch.prim.ListConstruct %[[BM_C0]], %[[BM_C2]], %[[BM_C1]]
  // CHECK: %[[BM_RHS:.*]] = torch.aten.permute %arg1, %[[BM_PERM]]
  // CHECK: torch.aten.bmm %arg0, %[[BM_RHS]]
  %0 = mfuse.aclnn.batch_matmul %arg0, %arg1 {trans_x2 = true} : (tensor<80x204x64xf32>, tensor<80x204x64xf32>) -> tensor<80x204x204xf32>
  return %0 : tensor<80x204x204xf32>
}

// CHECK-LABEL: func.func @test_aclnn_batch_matmul_trans_both
func.func @test_aclnn_batch_matmul_trans_both(%arg0: tensor<80x64x204xf32>, %arg1: tensor<80x128x64xf32>) -> tensor<80x204x128xf32> {
  // CHECK-DAG: %[[BMB_C0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[BMB_C1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[BMB_C2:.*]] = torch.constant.int 2
  // CHECK: %[[BMB_LHS_PERM:.*]] = torch.prim.ListConstruct %[[BMB_C0]], %[[BMB_C2]], %[[BMB_C1]]
  // CHECK: %[[BMB_LHS:.*]] = torch.aten.permute %arg0, %[[BMB_LHS_PERM]]
  // CHECK-DAG: %[[BMB_RHS_C0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[BMB_RHS_C1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[BMB_RHS_C2:.*]] = torch.constant.int 2
  // CHECK: %[[BMB_RHS_PERM:.*]] = torch.prim.ListConstruct %[[BMB_RHS_C0]], %[[BMB_RHS_C2]], %[[BMB_RHS_C1]]
  // CHECK: %[[BMB_RHS:.*]] = torch.aten.permute %arg1, %[[BMB_RHS_PERM]]
  // CHECK: torch.aten.bmm %[[BMB_LHS]], %[[BMB_RHS]]
  %0 = mfuse.aclnn.batch_matmul %arg0, %arg1 {trans_x1 = true, trans_x2 = true} : (tensor<80x64x204xf32>, tensor<80x128x64xf32>) -> tensor<80x204x128xf32>
  return %0 : tensor<80x204x128xf32>
}

// CHECK-LABEL: func.func @test_aclnn_batch_matmul_rank4_trans_x2
func.func @test_aclnn_batch_matmul_rank4_trans_x2(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<2x3x6x5xf32>) -> tensor<2x3x4x6xf32> {
  // CHECK-DAG: %[[BM4_C0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[BM4_C1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[BM4_C2:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[BM4_C3:.*]] = torch.constant.int 3
  // CHECK: %[[BM4_PERM:.*]] = torch.prim.ListConstruct %[[BM4_C0]], %[[BM4_C1]], %[[BM4_C3]], %[[BM4_C2]]
  // CHECK: %[[BM4_RHS:.*]] = torch.aten.permute %arg1, %[[BM4_PERM]]
  // CHECK-NOT: torch.aten.bmm
  // CHECK: torch.aten.matmul %arg0, %[[BM4_RHS]]
  %0 = mfuse.aclnn.batch_matmul %arg0, %arg1 {trans_x2 = true} : (tensor<2x3x4x5xf32>, tensor<2x3x6x5xf32>) -> tensor<2x3x4x6xf32>
  return %0 : tensor<2x3x4x6xf32>
}

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
  %0 = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = false} : (tensor<2x2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @test_reduce_sum_f32_to_f64
func.func @test_reduce_sum_f32_to_f64(%arg0: tensor<2x2xf32>) -> tensor<2xf64> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 7
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = mfuse.reduce_sum %arg0 {dimensions = [0], keepdim = false} : (tensor<2x2xf32>) -> tensor<2xf64>
  return %0 : tensor<2xf64>
}

// CHECK-LABEL: func.func @test_reduce_sum_i32
func.func @test_reduce_sum_i32(%arg0: tensor<2x3xsi32>) -> tensor<2xsi32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 3
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = false} : (tensor<2x3xsi32>) -> tensor<2xsi32>
  return %0 : tensor<2xsi32>
}

// CHECK-LABEL: func.func @test_reduce_sum_keepdim
func.func @test_reduce_sum_keepdim(%arg0: tensor<2x2xf32>) -> tensor<2x1xf32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool true
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = true} : (tensor<2x2xf32>) -> tensor<2x1xf32>
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
  %0 = mfuse.reduce_sum %arg0 {dimensions = [0, 1], keepdim = false} : (tensor<2x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func.func @test_reduce_sum_dtype_none
func.func @test_reduce_sum_dtype_none(%arg0: tensor<2x2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool false
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK: torch.aten.sum.dim_IntList %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = false} : (tensor<2x2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func.func @test_reduce_mean
func.func @test_reduce_mean(%arg0: tensor<2x2xf32>) -> tensor<2x1xf32> {
  // CHECK-DAG: %[[DIM1:.*]] = torch.constant.int 1
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM1]]
  // CHECK-DAG: %[[KEEP:.*]] = torch.constant.bool true
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK: torch.aten.mean.dim %{{.*}}, %[[DIMS]], %[[KEEP]], %[[DTYPE]]
  %0 = mfuse.reduce_mean %arg0 {dimensions = [1], keepdim = true} : (tensor<2x2xf32>) -> tensor<2x1xf32>
  return %0 : tensor<2x1xf32>
}

// CHECK-LABEL: func.func @test_cast
func.func @test_cast(%arg0: tensor<2x2xf32>) -> tensor<2x2xf16> {
  // CHECK:   %[[INT5:.*]] = torch.constant.int 5
  // CHECK:   %[[RESULT:.*]] = torch.prims.convert_element_type %{{.*}}, %[[INT5]] : !torch.vtensor<[2,2],f32>, !torch.int -> !torch.vtensor<[2,2],f16>
  // CHECK:   return %[[RESULT]] : !torch.vtensor<[2,2],f16>
  %0 = mfuse.cast %arg0 : (tensor<2x2xf32>) -> tensor<2x2xf16>
  return %0 : tensor<2x2xf16>
}

// CHECK-LABEL: func.func @test_cast_i32_to_f64
func.func @test_cast_i32_to_f64(%arg0: tensor<2x2xsi32>) -> tensor<2x2xf64> {
  // CHECK:   %[[INT7:.*]] = torch.constant.int 7
  // CHECK:   %[[RESULT:.*]] = torch.prims.convert_element_type %{{.*}}, %[[INT7]] : !torch.vtensor<[2,2],si32>, !torch.int -> !torch.vtensor<[2,2],f64>
  // CHECK:   return %[[RESULT]] : !torch.vtensor<[2,2],f64>
  %0 = mfuse.cast %arg0 : (tensor<2x2xsi32>) -> tensor<2x2xf64>
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
func.func @test_reshape_i32(%arg0: tensor<2x3xsi32>) -> tensor<6xsi32> {
  // CHECK-DAG: %[[DIM0:.*]] = torch.constant.int 6
  // CHECK-DAG: %[[DIMS:.*]] = torch.prim.ListConstruct %[[DIM0]]
  // CHECK: torch.aten.reshape %{{.*}}, %[[DIMS]]
  %0 = mfuse.reshape %arg0 : (tensor<2x3xsi32>) -> tensor<6xsi32>
  return %0 : tensor<6xsi32>
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

// CHECK-LABEL: func.func @test_add_i32
func.func @test_add_i32(%arg0: tensor<2x3xsi32>, %arg1: tensor<2x3xsi32>) -> tensor<2x3xsi32> {
  // CHECK: torch.aten.add.Tensor
  %0 = mfuse.add %arg0, %arg1 : (tensor<2x3xsi32>, tensor<2x3xsi32>) -> tensor<2x3xsi32>
  return %0 : tensor<2x3xsi32>
}

// CHECK-LABEL: func.func @test_add_f16
func.func @test_add_f16(%arg0: tensor<2x3xf16>, %arg1: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // CHECK: torch.aten.add.Tensor
  %0 = mfuse.add %arg0, %arg1 : (tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  return %0 : tensor<2x3xf16>
}

// CHECK-LABEL: func.func @test_add_f64
func.func @test_add_f64(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) -> tensor<2x3xf64> {
  // CHECK: torch.aten.add.Tensor
  %0 = mfuse.add %arg0, %arg1 : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
  return %0 : tensor<2x3xf64>
}

// CHECK-LABEL: func.func @test_sub
func.func @test_sub(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: torch.aten.sub.Tensor
  %0 = mfuse.sub %arg0, %arg1 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @test_mul_tensor
func.func @test_mul_tensor(%arg0: tensor<2x3xf32>, %arg1: tensor<f32>) -> tensor<2x3xf32> {
  // CHECK: torch.aten.mul.Tensor
  %0 = mfuse.mul %arg0, %arg1 : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @test_div_vector
func.func @test_div_vector(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: torch.aten.div.Tensor
  %0 = mfuse.div %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_div_3d
func.func @test_div_3d(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  // CHECK: torch.aten.div.Tensor
  %0 = mfuse.div %arg0, %arg1 : (tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// CHECK-LABEL: func.func @test_eq_i32
func.func @test_eq_i32(%arg0: tensor<2x2xsi32>, %arg1: tensor<2x2xsi32>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.eq.Tensor
  %0 = mfuse.eq %arg0, %arg1 : (tensor<2x2xsi32>, tensor<2x2xsi32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// CHECK-LABEL: func.func @test_gt
func.func @test_gt_scalar(%arg0: tensor<3x3xf32>, %arg1: tensor<f32>) -> tensor<3x3xi1> {
  // CHECK: torch.aten.gt.Tensor
  %0 = mfuse.gt %arg0, %arg1 : (tensor<3x3xf32>, tensor<f32>) -> tensor<3x3xi1>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: func.func @test_le_f64
func.func @test_le_f64(%arg0: tensor<2x2xf64>, %arg1: tensor<2x2xf64>) -> tensor<2x2xi1> {
  // CHECK: torch.aten.le.Tensor
  %0 = mfuse.le %arg0, %arg1 : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}

// mfuse.aclnn.mm: trans_x1/trans_x2 -> discardable attrs on torch.aten.mm (for FX export / DVM trans_a/trans_b).
// CHECK-LABEL: func.func @test_aclnn_mm_trans_b
func.func @test_aclnn_mm_trans_b(%arg0: tensor<2x4xf32>, %arg1: tensor<8x4xf32>) -> tensor<2x8xf32> {
  // CHECK: torch.aten.mm{{.*}}dvm_trans_b = true
  %0 = mfuse.aclnn.mm %arg0, %arg1 {trans_x2 = true} : (tensor<2x4xf32>, tensor<8x4xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// CHECK-LABEL: func.func @test_full_with_explicit_attrs
func.func @test_full_with_explicit_attrs() -> tensor<2x3xf32> {
  // CHECK-DAG: %[[D0:.*]] = torch.constant.int 2
  // CHECK-DAG: %[[D1:.*]] = torch.constant.int 3
  // CHECK-DAG: %[[SIZE:.*]] = torch.prim.ListConstruct %[[D0]], %[[D1]]
  // CHECK-DAG: %[[FILL:.*]] = torch.constant.float 3.500000e+00
  // CHECK-DAG: %[[DTYPE:.*]] = torch.constant.int 6
  // CHECK-DAG: %[[LAYOUT:.*]] = torch.constant.int 0
  // CHECK-DAG: %[[DEVICE:.*]] = torch.constant.device "npu"
  // CHECK-DAG: %[[PIN:.*]] = torch.constant.bool true
  // CHECK: %[[FULL0:.*]] = torch.aten.full %[[SIZE]], %[[FILL]], %[[DTYPE]], %[[LAYOUT]], %[[DEVICE]], %[[PIN]]
  %cst = mfuse.constant dense<3.500000e+00> : tensor<f32, {is_scalar = ""}>
  %0 = mfuse.full %cst {device = "npu", dtype = 6 : i64, layout = 0 : i64, pin_memory = true} : (tensor<f32, {is_scalar = ""}>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @test_full_with_inferred_dtype
func.func @test_full_with_inferred_dtype() -> tensor<4x5xf16> {
  // CHECK-DAG: %[[D2:.*]] = torch.constant.int 4
  // CHECK-DAG: %[[D3:.*]] = torch.constant.int 5
  // CHECK-DAG: %[[SIZE1:.*]] = torch.prim.ListConstruct %[[D2]], %[[D3]]
  // CHECK-DAG: %[[FILL1:.*]] = torch.constant.float 1.250000e+00
  // CHECK-DAG: %[[DTYPE1:.*]] = torch.constant.int 5
  // CHECK-DAG: %[[LAYOUT_NONE:.*]] = torch.constant.none
  // CHECK-DAG: %[[DEVICE_NONE:.*]] = torch.constant.none
  // CHECK-DAG: %[[PIN_NONE:.*]] = torch.constant.none
  // CHECK: %[[FULL1:.*]] = torch.aten.full %[[SIZE1]], %[[FILL1]], %[[DTYPE1]], %[[LAYOUT_NONE]], %[[DEVICE_NONE]], %[[PIN_NONE]]
  %cst = mfuse.constant dense<1.250000e+00> : tensor<f16, {is_scalar = ""}>
  %0 = mfuse.full %cst : (tensor<f16, {is_scalar = ""}>) -> tensor<4x5xf16>
  return %0 : tensor<4x5xf16>
}
