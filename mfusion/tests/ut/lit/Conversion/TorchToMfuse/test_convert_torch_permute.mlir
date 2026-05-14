// ConvertAtenPermute is registered in TorchAtenToMfuse.cc (populateAtenToMfuseCustomPatterns).

// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @convert_permute_basic
// CHECK: %[[PERMUTE0:.*]] = mfuse.permute
// CHECK-NOT: torch.aten.permute
func.func @convert_permute_basic(%input: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[4,2,3],f32> {
  %c2 = torch.constant.int 2
  %c0 = torch.constant.int 0
  %c1 = torch.constant.int 1
  %dims = torch.prim.ListConstruct %c2, %c0, %c1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %permute = torch.aten.permute %input, %dims : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[4,2,3],f32>
  return %permute : !torch.vtensor<[4,2,3],f32>
}

// CHECK-LABEL: func.func @convert_permute_2d
// CHECK: %[[PERMUTE1:.*]] = mfuse.permute
// CHECK-NOT: torch.aten.permute
func.func @convert_permute_2d(%input: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[4,3],f32> {
  %c1 = torch.constant.int 1
  %c0 = torch.constant.int 0
  %dims = torch.prim.ListConstruct %c1, %c0 : (!torch.int, !torch.int) -> !torch.list<int>
  %permute = torch.aten.permute %input, %dims : !torch.vtensor<[3,4],f32>, !torch.list<int> -> !torch.vtensor<[4,3],f32>
  return %permute : !torch.vtensor<[4,3],f32>
}

// CHECK-LABEL: func.func @convert_permute_with_dynamic_dim
// CHECK: mfuse.syminfo =
// CHECK-NOT: torch.bind_symbolic_shape
// CHECK: %[[PERMUTE3:.*]] = mfuse.permute
// CHECK-NOT: torch.aten.permute
func.func @convert_permute_with_dynamic_dim(%input: !torch.vtensor<[2,?,4],f32>) -> !torch.vtensor<[4,2,?],f32> {
  %s0 = torch.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  torch.bind_symbolic_shape %input, [%s0], affine_map<()[s0] -> (2, s0, 4)> : !torch.vtensor<[2,?,4],f32>
  %c2 = torch.constant.int 2
  %c0 = torch.constant.int 0
  %c1 = torch.constant.int 1
  %dims = torch.prim.ListConstruct %c2, %c0, %c1 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %permute = torch.aten.permute %input, %dims : !torch.vtensor<[2,?,4],f32>, !torch.list<int> -> !torch.vtensor<[4,2,?],f32>
  torch.bind_symbolic_shape %permute, [%s0], affine_map<()[s0] -> (4, 2, s0)> : !torch.vtensor<[4,2,?],f32>
  return %permute : !torch.vtensor<[4,2,?],f32>
}

// CHECK-LABEL: func.func @convert_permute_4d
// CHECK: %[[PERMUTE4:.*]] = mfuse.permute
// CHECK-NOT: torch.aten.permute
func.func @convert_permute_4d(%input: !torch.vtensor<[2,3,4,5],f32>) -> !torch.vtensor<[5,4,3,2],f32> {
  %c3 = torch.constant.int 3
  %c2 = torch.constant.int 2
  %c1 = torch.constant.int 1
  %c0 = torch.constant.int 0
  %dims = torch.prim.ListConstruct %c3, %c2, %c1, %c0 : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %permute = torch.aten.permute %input, %dims : !torch.vtensor<[2,3,4,5],f32>, !torch.list<int> -> !torch.vtensor<[5,4,3,2],f32>
  return %permute : !torch.vtensor<[5,4,3,2],f32>
}

// Negative test: dims is not a constant int list, should NOT be converted
// CHECK-LABEL: func.func @convert_permute_non_const_dims
// CHECK-NOT: mfuse.permute
// CHECK: torch.aten.permute
func.func @convert_permute_non_const_dims(%input: !torch.vtensor<[2,3,4],f32>, %dim0: !torch.int, %dim1: !torch.int, %dim2: !torch.int) -> !torch.vtensor<[?,?,?],f32> {
  %dims = torch.prim.ListConstruct %dim0, %dim1, %dim2 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %permute = torch.aten.permute %input, %dims : !torch.vtensor<[2,3,4],f32>, !torch.list<int> -> !torch.vtensor<[?,?,?],f32>
  return %permute : !torch.vtensor<[?,?,?],f32>
}
