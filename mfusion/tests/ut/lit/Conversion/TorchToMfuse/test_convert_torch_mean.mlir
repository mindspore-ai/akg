// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @convert_mean_suffix_dims_keepdim_true
// CHECK: %[[MEAN:.*]] = mfuse.reduce_mean %{{.*}} {dimensions = [3, 2], keepdim = true}
// CHECK-NOT: torch.aten.mean.dim
func.func @convert_mean_suffix_dims_keepdim_true(%input: !torch.vtensor<[1,16,7,7],f32>) -> !torch.vtensor<[1,16,1,1],f32> {
  %c3 = torch.constant.int 3
  %c2 = torch.constant.int 2
  %true = torch.constant.bool true
  %none = torch.constant.none
  %dims = torch.prim.ListConstruct %c3, %c2 : (!torch.int, !torch.int) -> !torch.list<int>
  %mean = torch.aten.mean.dim %input, %dims, %true, %none : !torch.vtensor<[1,16,7,7],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1,16,1,1],f32>
  return %mean : !torch.vtensor<[1,16,1,1],f32>
}

// CHECK-LABEL: func.func @convert_mean_negative_last_dim_keepdim_true
// CHECK: %[[MEAN:.*]] = mfuse.reduce_mean %{{.*}} {dimensions = [2], keepdim = true}
// CHECK-NOT: torch.aten.mean.dim
func.func @convert_mean_negative_last_dim_keepdim_true(%input: !torch.vtensor<[4,8,16],f32>) -> !torch.vtensor<[4,8,1],f32> {
  %cminus1 = torch.constant.int -1
  %true = torch.constant.bool true
  %none = torch.constant.none
  %dims = torch.prim.ListConstruct %cminus1 : (!torch.int) -> !torch.list<int>
  %mean = torch.aten.mean.dim %input, %dims, %true, %none : !torch.vtensor<[4,8,16],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[4,8,1],f32>
  return %mean : !torch.vtensor<[4,8,1],f32>
}

// CHECK-LABEL: func.func @convert_mean_int_input_float_dtype
// CHECK: %[[MEAN:.*]] = mfuse.reduce_mean %{{.*}} {dimensions = [1], keepdim = true} : (tensor<2x4xsi32>) -> tensor<2x1xf32>
// CHECK-NOT: torch.aten.mean.dim
func.func @convert_mean_int_input_float_dtype(%input: !torch.vtensor<[2,4],si32>) -> !torch.vtensor<[2,1],f32> {
  %c1 = torch.constant.int 1
  %true = torch.constant.bool true
  %dtype = torch.constant.int 6
  %dims = torch.prim.ListConstruct %c1 : (!torch.int) -> !torch.list<int>
  %mean = torch.aten.mean.dim %input, %dims, %true, %dtype : !torch.vtensor<[2,4],si32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[2,1],f32>
  return %mean : !torch.vtensor<[2,1],f32>
}

// CHECK-LABEL: func.func @convert_mean_scalar_empty_dim_keepdim_false
// CHECK: %[[MEAN:.*]] = mfuse.reduce_mean %{{.*}} {dimensions = [], keepdim = false} : (tensor<f32>) -> tensor<f32>
// CHECK-NOT: torch.aten.mean.dim
func.func @convert_mean_scalar_empty_dim_keepdim_false(%input: !torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> {
  %false = torch.constant.bool false
  %none = torch.constant.none
  %dims = torch.prim.ListConstruct  : () -> !torch.list<int>
  %mean = torch.aten.mean.dim %input, %dims, %false, %none : !torch.vtensor<[],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[],f32>
  return %mean : !torch.vtensor<[],f32>
}
