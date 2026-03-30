// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --canonicalize | FileCheck %s

//===----------------------------------------------------------------------===//
// torch.prims.convert_element_type -> mfuse.cast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @convert_prims_convert_element_type_f16_to_f32
// CHECK: %[[CAST0:.*]] = mfuse.cast
// CHECK-NOT: torch.prims.convert_element_type
func.func @convert_prims_convert_element_type_f16_to_f32(%input: !torch.vtensor<[2,4],f16>) -> !torch.vtensor<[2,4],f32> {
  %dtype = torch.constant.int 6
  %cast = torch.prims.convert_element_type %input, %dtype : !torch.vtensor<[2,4],f16>, !torch.int -> !torch.vtensor<[2,4],f32>
  return %cast : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_prims_convert_element_type_f32_to_f16
// CHECK: %[[CAST1:.*]] = mfuse.cast
// CHECK-NOT: torch.prims.convert_element_type
func.func @convert_prims_convert_element_type_f32_to_f16(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f16> {
  %dtype = torch.constant.int 5
  %cast = torch.prims.convert_element_type %input, %dtype : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f16>
  return %cast : !torch.vtensor<[2,4],f16>
}

// CHECK-LABEL: func.func @convert_prims_convert_element_type_with_dynamic
// CHECK: mfuse.syminfo =
// CHECK-NOT: torch.bind_symbolic_shape
// CHECK: %[[CAST2:.*]] = mfuse.cast
// CHECK-NOT: torch.prims.convert_element_type
func.func @convert_prims_convert_element_type_with_dynamic(%input: !torch.vtensor<[?,4],f16>) -> !torch.vtensor<[?,4],f32> {
  %s0 = torch.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  torch.bind_symbolic_shape %input, [%s0], affine_map<()[s0] -> (s0, 4)> : !torch.vtensor<[?,4],f16>
  %dtype = torch.constant.int 6
  %cast = torch.prims.convert_element_type %input, %dtype : !torch.vtensor<[?,4],f16>, !torch.int -> !torch.vtensor<[?,4],f32>
  torch.bind_symbolic_shape %cast, [%s0], affine_map<()[s0] -> (s0, 4)> : !torch.vtensor<[?,4],f32>
  return %cast : !torch.vtensor<[?,4],f32>
}

// CHECK-LABEL: func.func @convert_prims_convert_element_type_f32_to_bf16
// CHECK: %[[CAST3:.*]] = mfuse.cast
// CHECK-NOT: torch.prims.convert_element_type
func.func @convert_prims_convert_element_type_f32_to_bf16(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],bf16> {
  %dtype = torch.constant.int 17
  %cast = torch.prims.convert_element_type %input, %dtype : !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],bf16>
  return %cast : !torch.vtensor<[2,4],bf16>
}

// CHECK-LABEL: func.func @convert_prims_convert_element_type_bf16_to_f32
// CHECK: %[[CAST4:.*]] = mfuse.cast
// CHECK-NOT: torch.prims.convert_element_type
func.func @convert_prims_convert_element_type_bf16_to_f32(%input: !torch.vtensor<[2,4],bf16>) -> !torch.vtensor<[2,4],f32> {
  %dtype = torch.constant.int 6
  %cast = torch.prims.convert_element_type %input, %dtype : !torch.vtensor<[2,4],bf16>, !torch.int -> !torch.vtensor<[2,4],f32>
  return %cast : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_prims_convert_element_type_3d
// CHECK: %[[CAST5:.*]] = mfuse.cast
// CHECK-NOT: torch.prims.convert_element_type
func.func @convert_prims_convert_element_type_3d(%input: !torch.vtensor<[2,3,4],f16>) -> !torch.vtensor<[2,3,4],f32> {
  %dtype = torch.constant.int 6
  %cast = torch.prims.convert_element_type %input, %dtype : !torch.vtensor<[2,3,4],f16>, !torch.int -> !torch.vtensor<[2,3,4],f32>
  return %cast : !torch.vtensor<[2,3,4],f32>
}

//===----------------------------------------------------------------------===//
// torch.npu._npu_dtype_cast -> mfuse.cast
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @convert_npu_dtype_cast_f16_to_f32
// CHECK: %[[CAST6:.*]] = mfuse.cast
// CHECK-NOT: torch.operator
func.func @convert_npu_dtype_cast_f16_to_f32(%input: !torch.vtensor<[2,4],f16>) -> !torch.vtensor<[2,4],f32> {
  %cast = torch.operator "torch.npu._npu_dtype_cast"(%input) : (!torch.vtensor<[2,4],f16>) -> !torch.vtensor<[2,4],f32>
  return %cast : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @convert_npu_dtype_cast_f32_to_f16
// CHECK: %[[CAST7:.*]] = mfuse.cast
// CHECK-NOT: torch.operator
func.func @convert_npu_dtype_cast_f32_to_f16(%input: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f16> {
  %cast = torch.operator "torch.npu._npu_dtype_cast"(%input) : (!torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f16>
  return %cast : !torch.vtensor<[2,4],f16>
}

// CHECK-LABEL: func.func @convert_npu_dtype_cast_with_dynamic
// CHECK: mfuse.syminfo =
// CHECK-NOT: torch.bind_symbolic_shape
// CHECK: %[[CAST8:.*]] = mfuse.cast
// CHECK-NOT: torch.operator
func.func @convert_npu_dtype_cast_with_dynamic(%input: !torch.vtensor<[?,4],f16>) -> !torch.vtensor<[?,4],f32> {
  %s0 = torch.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  torch.bind_symbolic_shape %input, [%s0], affine_map<()[s0] -> (s0, 4)> : !torch.vtensor<[?,4],f16>
  %cast = torch.operator "torch.npu._npu_dtype_cast"(%input) : (!torch.vtensor<[?,4],f16>) -> !torch.vtensor<[?,4],f32>
  torch.bind_symbolic_shape %cast, [%s0], affine_map<()[s0] -> (s0, 4)> : !torch.vtensor<[?,4],f32>
  return %cast : !torch.vtensor<[?,4],f32>
}
