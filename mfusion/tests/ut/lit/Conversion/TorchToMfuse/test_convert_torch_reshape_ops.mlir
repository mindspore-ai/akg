// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// torch.aten.view -> mfuse.reshape
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @convert_view_with_const_shape
// CHECK: %[[RESHAPE0:.*]] = mfuse.reshape
// CHECK-NOT: torch.aten.view
func.func @convert_view_with_const_shape(%input: !torch.vtensor<[2,3,4],bf16>) -> !torch.vtensor<[6,4],bf16> {
  %c6 = torch.constant.int 6
  %c4 = torch.constant.int 4
  %shape = torch.prim.ListConstruct %c6, %c4 : (!torch.int, !torch.int) -> !torch.list<int>
  %view = torch.aten.view %input, %shape : !torch.vtensor<[2,3,4],bf16>, !torch.list<int> -> !torch.vtensor<[6,4],bf16>
  return %view : !torch.vtensor<[6,4],bf16>
}

// CHECK-LABEL: func.func @convert_view_with_const_minus_one
// CHECK: mfuse.syminfo =
// CHECK-NOT: torch.bind_symbolic_shape
// CHECK: %[[RESHAPE1:.*]] = mfuse.reshape
// CHECK-NOT: torch.aten.view
func.func @convert_view_with_const_minus_one(%input: !torch.vtensor<[2,?,4],bf16>) -> !torch.vtensor<[?,4],bf16> {
  %s0 = torch.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  torch.bind_symbolic_shape %input, [%s0], affine_map<()[s0] -> (2, s0, 4)> : !torch.vtensor<[2,?,4],bf16>
  %cneg1 = torch.constant.int -1
  %c4 = torch.constant.int 4
  %shape = torch.prim.ListConstruct %cneg1, %c4 : (!torch.int, !torch.int) -> !torch.list<int>
  %view = torch.aten.view %input, %shape : !torch.vtensor<[2,?,4],bf16>, !torch.list<int> -> !torch.vtensor<[?,4],bf16>
  torch.bind_symbolic_shape %view, [%s0], affine_map<()[s0] -> ((2 * s0), 4)> : !torch.vtensor<[?,4],bf16>
  return %view : !torch.vtensor<[?,4],bf16>
}

// CHECK-LABEL: func.func @convert_view_with_computed_shape
// CHECK: mfuse.syminfo =
// CHECK-NOT: torch.bind_symbolic_shape
// CHECK: %[[RESHAPE2:.*]] = mfuse.reshape
// CHECK-NOT: torch.aten.mul.int
// CHECK-NOT: torch.aten.sub.int
// CHECK-NOT: torch.prim.ListConstruct
// CHECK-NOT: torch.aten.view
func.func @convert_view_with_computed_shape(%input: !torch.vtensor<[2,?,4],bf16>) -> !torch.vtensor<[?,4],bf16> {
  %s1 = torch.symbolic_int "s1" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  torch.bind_symbolic_shape %input, [%s1], affine_map<()[s1] -> (2, s1, 4)> : !torch.vtensor<[2,?,4],bf16>
  %c2 = torch.constant.int 2
  %c3 = torch.constant.int 3
  %c4 = torch.constant.int 4
  %c10 = torch.constant.int 10
  %m0 = torch.aten.mul.int %c2, %c3 : !torch.int, !torch.int -> !torch.int
  %m1 = torch.aten.sub.int %c10, %m0 : !torch.int, !torch.int -> !torch.int
  %neg1 = torch.aten.sub.int %c3, %c4 : !torch.int, !torch.int -> !torch.int
  %shape = torch.prim.ListConstruct %neg1, %m1 : (!torch.int, !torch.int) -> !torch.list<int>
  %view = torch.aten.view %input, %shape : !torch.vtensor<[2,?,4],bf16>, !torch.list<int> -> !torch.vtensor<[?,4],bf16>
  torch.bind_symbolic_shape %view, [%s1], affine_map<()[s1] -> ((2 * s1), 4)> : !torch.vtensor<[?,4],bf16>
  return %view : !torch.vtensor<[?,4],bf16>
}

// CHECK-LABEL: func.func @convert_view_dynamic_shape
// CHECK: mfuse.syminfo =
// CHECK-NOT: torch.bind_symbolic_shape
// CHECK: torch.aten.view
func.func @convert_view_dynamic_shape(%arg0: !torch.int, %arg1: !torch.int, %arg2: !torch.int, %arg3: !torch.int, %arg4: !torch.int, %arg5: !torch.vtensor<[?,?,?,?,?],bf16>, %arg6: !torch.int, %arg7: !torch.int) -> !torch.vtensor<[?,?,?],bf16> {
  %0 = torch.symbolic_int "s76" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  %1 = torch.symbolic_int "s67" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  %2 = torch.symbolic_int "s36" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  %3 = torch.symbolic_int "s90" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  %4 = torch.symbolic_int "s72" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  torch.bind_symbolic_shape %arg5, [%2, %1, %4, %0, %3], affine_map<()[s0, s1, s2, s3, s4] -> (s3, s1, s0, s4, s2)> : !torch.vtensor<[?,?,?,?,?],bf16>
  %5 = torch.aten.mul.int %arg1, %arg2 : !torch.int, !torch.int -> !torch.int
  %6 = torch.aten.mul.int %5, %arg3 : !torch.int, !torch.int -> !torch.int
  %7 = torch.prim.ListConstruct %arg0, %6, %arg4 : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  %8 = torch.aten.view %arg5, %7 : !torch.vtensor<[?,?,?,?,?],bf16>, !torch.list<int> -> !torch.vtensor<[?,?,?],bf16>
  torch.bind_symbolic_shape %8, [%2, %1, %4, %0, %3], affine_map<()[s0, s1, s2, s3, s4] -> (s3, (s0 * s1) * s4, s2)> : !torch.vtensor<[?,?,?],bf16>
  return %8 : !torch.vtensor<[?,?,?],bf16>
}

//===----------------------------------------------------------------------===//
// torch.aten.squeeze.dim -> mfuse.reshape
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @convert_squeeze_dim_basic
// CHECK: %[[RESHAPE3:.*]] = mfuse.reshape
// CHECK-NOT: torch.aten.squeeze.dim
func.func @convert_squeeze_dim_basic(%input: !torch.vtensor<[2,1,4],f32>) -> !torch.vtensor<[2,4],f32> {
  %c1 = torch.constant.int 1
  %squeeze = torch.aten.squeeze.dim %input, %c1 : !torch.vtensor<[2,1,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
  return %squeeze : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_squeeze_dim_negative
// CHECK: %[[RESHAPE4:.*]] = mfuse.reshape
// CHECK-NOT: torch.aten.squeeze.dim
func.func @convert_squeeze_dim_negative(%input: !torch.vtensor<[2,4,1],f32>) -> !torch.vtensor<[2,4],f32> {
  %cneg1 = torch.constant.int -1
  %squeeze = torch.aten.squeeze.dim %input, %cneg1 : !torch.vtensor<[2,4,1],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
  return %squeeze : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_squeeze_dim_with_dynamic
// CHECK: mfuse.syminfo =
// CHECK-NOT: torch.bind_symbolic_shape
// CHECK: %[[RESHAPE5:.*]] = mfuse.reshape
// CHECK-NOT: torch.aten.squeeze.dim
func.func @convert_squeeze_dim_with_dynamic(%input: !torch.vtensor<[?,1,4],f32>) -> !torch.vtensor<[?,4],f32> {
  %s0 = torch.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  torch.bind_symbolic_shape %input, [%s0], affine_map<()[s0] -> (s0, 1, 4)> : !torch.vtensor<[?,1,4],f32>
  %c1 = torch.constant.int 1
  %squeeze = torch.aten.squeeze.dim %input, %c1 : !torch.vtensor<[?,1,4],f32>, !torch.int -> !torch.vtensor<[?,4],f32>
  torch.bind_symbolic_shape %squeeze, [%s0], affine_map<()[s0] -> (s0, 4)> : !torch.vtensor<[?,4],f32>
  return %squeeze : !torch.vtensor<[?,4],f32>
}

//===----------------------------------------------------------------------===//
// torch.aten.unsqueeze -> mfuse.reshape
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @convert_unsqueeze_basic
// CHECK: %[[RESHAPE6:.*]] = mfuse.reshape
// CHECK-NOT: torch.aten.unsqueeze
func.func @convert_unsqueeze_basic(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,1,4],f32> {
  %c1 = torch.constant.int 1
  %unsqueeze = torch.aten.unsqueeze %input, %c1 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,1,4],f32>
  return %unsqueeze : !torch.vtensor<[2,1,4],f32>
}

// CHECK-LABEL: func.func @convert_unsqueeze_negative
// CHECK: %[[RESHAPE7:.*]] = mfuse.reshape
// CHECK-NOT: torch.aten.unsqueeze
func.func @convert_unsqueeze_negative(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4,1],f32> {
  %cneg1 = torch.constant.int -1
  %unsqueeze = torch.aten.unsqueeze %input, %cneg1 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4,1],f32>
  return %unsqueeze : !torch.vtensor<[2,4,1],f32>
}

// CHECK-LABEL: func.func @convert_unsqueeze_with_dynamic
// CHECK: mfuse.syminfo =
// CHECK-NOT: torch.bind_symbolic_shape
// CHECK: %[[RESHAPE8:.*]] = mfuse.reshape
// CHECK-NOT: torch.aten.unsqueeze
func.func @convert_unsqueeze_with_dynamic(%input: !torch.vtensor<[?,4],f32>) -> !torch.vtensor<[?,1,4],f32> {
  %s0 = torch.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  torch.bind_symbolic_shape %input, [%s0], affine_map<()[s0] -> (s0, 4)> : !torch.vtensor<[?,4],f32>
  %c1 = torch.constant.int 1
  %unsqueeze = torch.aten.unsqueeze %input, %c1 : !torch.vtensor<[?,4],f32>, !torch.int -> !torch.vtensor<[?,1,4],f32>
  torch.bind_symbolic_shape %unsqueeze, [%s0], affine_map<()[s0] -> (s0, 1, 4)> : !torch.vtensor<[?,1,4],f32>
  return %unsqueeze : !torch.vtensor<[?,1,4],f32>
}

// CHECK-LABEL: func.func @convert_unsqueeze_at_front
// CHECK: %[[RESHAPE9:.*]] = mfuse.reshape
// CHECK-NOT: torch.aten.unsqueeze
func.func @convert_unsqueeze_at_front(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[1,2,4],f32> {
  %c0 = torch.constant.int 0
  %unsqueeze = torch.aten.unsqueeze %input, %c0 : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[1,2,4],f32>
  return %unsqueeze : !torch.vtensor<[1,2,4],f32>
}
