// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @convert_reducesum_dim1_keepdim_false
// CHECK: %[[REDUCE0:.*]] = mfuse.reduce_sum %{{.*}} {dimensions = [1], keepdim = false}
// CHECK-NOT: torch.aten.sum.dim_IntList
func.func @convert_reducesum_dim1_keepdim_false(%input: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %c1 = torch.constant.int 1
  %false = torch.constant.bool false
  %dtype = torch.constant.int 6
  %dims = torch.prim.ListConstruct %c1 : (!torch.int) -> !torch.list<int>
  %sum = torch.aten.sum.dim_IntList %input, %dims, %false, %dtype : !torch.vtensor<[2,3,4],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[2,4],f32>
  return %sum : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_reducesum_empty_dim_keepdim_true
// CHECK: %[[REDUCE1:.*]] = mfuse.reduce_sum %{{.*}} {dimensions = [0, 1, 2], keepdim = true}
// CHECK-NOT: torch.aten.sum.dim_IntList
func.func @convert_reducesum_empty_dim_keepdim_true(%input: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[1,1,1],f32> {
  %true = torch.constant.bool true
  %dtype = torch.constant.int 6
  %dims = torch.prim.ListConstruct  : () -> !torch.list<int>
  %sum = torch.aten.sum.dim_IntList %input, %dims, %true, %dtype : !torch.vtensor<[2,3,4],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,1,1],f32>
  return %sum : !torch.vtensor<[1,1,1],f32>
}

// CHECK-LABEL: func.func @convert_reducesum_negative_dim
// CHECK: %[[REDUCE2:.*]] = mfuse.reduce_sum %{{.*}} {dimensions = [2], keepdim = false}
// CHECK-NOT: torch.aten.sum.dim_IntList
func.func @convert_reducesum_negative_dim(%input: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[2,3],f32> {
  %cneg1 = torch.constant.int -1
  %false = torch.constant.bool false
  %dtype = torch.constant.int 6
  %dims = torch.prim.ListConstruct %cneg1 : (!torch.int) -> !torch.list<int>
  %sum = torch.aten.sum.dim_IntList %input, %dims, %false, %dtype : !torch.vtensor<[2,3,4],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[2,3],f32>
  return %sum : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: func.func @convert_reducesum_multi_negative_dims
// CHECK: %[[REDUCE3:.*]] = mfuse.reduce_sum %{{.*}} {dimensions = [0, 2], keepdim = true}
// CHECK-NOT: torch.aten.sum.dim_IntList
func.func @convert_reducesum_multi_negative_dims(%input: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[1,3,1],f32> {
  %cneg3 = torch.constant.int -3
  %cneg1 = torch.constant.int -1
  %true = torch.constant.bool true
  %dtype = torch.constant.int 6
  %dims = torch.prim.ListConstruct %cneg3, %cneg1 : (!torch.int, !torch.int) -> !torch.list<int>
  %sum = torch.aten.sum.dim_IntList %input, %dims, %true, %dtype : !torch.vtensor<[2,3,4],f32>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[1,3,1],f32>
  return %sum : !torch.vtensor<[1,3,1],f32>
}

// CHECK-LABEL: func.func @convert_reducesum_with_dtype_cast
// CHECK: %[[REDUCE4:.*]] = mfuse.reduce_sum %{{.*}} {dimensions = [0], keepdim = false}
// CHECK-NOT: torch.aten.sum.dim_IntList
func.func @convert_reducesum_with_dtype_cast(%input: !torch.vtensor<[2,3,4],f16>) -> !torch.vtensor<[3,4],f32> {
  %c0 = torch.constant.int 0
  %false = torch.constant.bool false
  %dtype = torch.constant.int 6
  %dims = torch.prim.ListConstruct %c0 : (!torch.int) -> !torch.list<int>
  %sum = torch.aten.sum.dim_IntList %input, %dims, %false, %dtype : !torch.vtensor<[2,3,4],f16>, !torch.list<int>, !torch.bool, !torch.int -> !torch.vtensor<[3,4],f32>
  return %sum : !torch.vtensor<[3,4],f32>
}
