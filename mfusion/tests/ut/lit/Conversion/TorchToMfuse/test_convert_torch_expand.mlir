// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @convert_expand_basic
// CHECK: %[[BROADCAST0:.*]] = mfuse.broadcast_to
// CHECK-NOT: torch.aten.expand
func.func @convert_expand_basic(%input: !torch.vtensor<[3,1],f32>) -> !torch.vtensor<[3,4],f32> {
  %c3 = torch.constant.int 3
  %c4 = torch.constant.int 4
  %false = torch.constant.bool false
  %size = torch.prim.ListConstruct %c3, %c4 : (!torch.int, !torch.int) -> !torch.list<int>
  %expand = torch.aten.expand %input, %size, %false : !torch.vtensor<[3,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[3,4],f32>
  return %expand : !torch.vtensor<[3,4],f32>
}

// CHECK-LABEL: func.func @convert_expand_3d
// CHECK: %[[BROADCAST1:.*]] = mfuse.broadcast_to
// CHECK-NOT: torch.aten.expand
func.func @convert_expand_3d(%input: !torch.vtensor<[1,1,4],f32>) -> !torch.vtensor<[2,3,4],f32> {
  %c2 = torch.constant.int 2
  %c3 = torch.constant.int 3
  %c4 = torch.constant.int 4
  %false = torch.constant.bool false
  %size = torch.prim.ListConstruct %c2, %c3, %c4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %expand = torch.aten.expand %input, %size, %false : !torch.vtensor<[1,1,4],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[2,3,4],f32>
  return %expand : !torch.vtensor<[2,3,4],f32>
}

// CHECK-LABEL: func.func @convert_expand_with_dynamic_input_dim
// CHECK: mfuse.syminfo =
// CHECK-NOT: torch.bind_symbolic_shape
// CHECK: %[[BROADCAST2:.*]] = mfuse.broadcast_to
// CHECK-NOT: torch.aten.expand
func.func @convert_expand_with_dynamic_input_dim(%input: !torch.vtensor<[?,1],f32>) -> !torch.vtensor<[?,4],f32> {
  %s0 = torch.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  torch.bind_symbolic_shape %input, [%s0], affine_map<()[s0] -> (s0, 1)> : !torch.vtensor<[?,1],f32>
  %c_neg1 = torch.constant.int -1
  %c4 = torch.constant.int 4
  %false = torch.constant.bool false
  %size = torch.prim.ListConstruct %c_neg1, %c4 : (!torch.int, !torch.int) -> !torch.list<int>
  %expand = torch.aten.expand %input, %size, %false : !torch.vtensor<[?,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,4],f32>
  torch.bind_symbolic_shape %expand, [%s0], affine_map<()[s0] -> (s0, 4)> : !torch.vtensor<[?,4],f32>
  return %expand : !torch.vtensor<[?,4],f32>
}

// Negative test: size is not a constant int list, should NOT be converted
// CHECK-LABEL: func.func @convert_expand_non_const_size
// CHECK-NOT: mfuse.broadcast_to
// CHECK: torch.aten.expand
func.func @convert_expand_non_const_size(%input: !torch.vtensor<[3,1],f32>, %dim0: !torch.int) -> !torch.vtensor<[?,4],f32> {
  %c4 = torch.constant.int 4
  %false = torch.constant.bool false
  %size = torch.prim.ListConstruct %dim0, %c4 : (!torch.int, !torch.int) -> !torch.list<int>
  %expand = torch.aten.expand %input, %size, %false : !torch.vtensor<[3,1],f32>, !torch.list<int>, !torch.bool -> !torch.vtensor<[?,4],f32>
  return %expand : !torch.vtensor<[?,4],f32>
}
