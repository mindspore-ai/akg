// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse | FileCheck %s
// Guard: binary ops with mixed (f16, f32) are converted by convert-torch-to-mfuse
// (meta ops use promoteBinaryOperands; add/sub go to aclnn).

// CHECK-LABEL: func.func @main_mul
// CHECK: mfuse.mul
// CHECK-NOT: torch.aten.mul.Tensor
func.func @main_mul(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @main_add
// CHECK: mfuse.aclnn.add
// CHECK-NOT: torch.aten.add.Tensor
func.func @main_add(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @main_sub
// CHECK: mfuse.aclnn.sub
// CHECK-NOT: torch.aten.sub.Tensor
func.func @main_sub(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %int1 = torch.constant.int 1
  %0 = torch.aten.sub.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32>, !torch.int -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @main_div
// CHECK: mfuse.div
// CHECK-NOT: torch.aten.div.Tensor
func.func @main_div(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @main_div_rank0_float
// CHECK: %[[RHS_IN:.*]] = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[],f64> to tensor<f64>
// CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[80,204,204],f32> to tensor<80x204x204xf32>
// CHECK: %[[RHS:.*]] = mfuse.cast %[[RHS_IN]] : (tensor<f64>) -> tensor<f32>
// CHECK: mfuse.div %[[LHS]], %[[RHS]] : (tensor<80x204x204xf32>, tensor<f32>) -> tensor<80x204x204xf32>
// CHECK-NOT: torch.aten.div.Tensor
func.func @main_div_rank0_float(%arg0: !torch.vtensor<[80,204,204],f32>, %arg1: !torch.vtensor<[],f64>) -> !torch.vtensor<[80,204,204],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.div.Tensor %arg0, %arg1 : !torch.vtensor<[80,204,204],f32>, !torch.vtensor<[],f64> -> !torch.vtensor<[80,204,204],f32>
  return %0 : !torch.vtensor<[80,204,204],f32>
}

// CHECK-LABEL: func.func @main_maximum
// CHECK: mfuse.maximum
// CHECK-NOT: torch.aten.maximum
func.func @main_maximum(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.maximum %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @main_minimum
// CHECK: mfuse.minimum
// CHECK-NOT: torch.aten.minimum
func.func @main_minimum(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.minimum %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}

// CHECK-LABEL: func.func @main_eq
// CHECK: mfuse.eq
// CHECK-NOT: torch.aten.eq.Tensor
func.func @main_eq(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],i1> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.eq.Tensor %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],i1>
  return %0 : !torch.vtensor<[2,4],i1>
}

// CHECK-LABEL: func.func @main_ne
// CHECK: mfuse.ne
// CHECK-NOT: torch.aten.ne.Tensor
func.func @main_ne(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],i1> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.ne.Tensor %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],i1>
  return %0 : !torch.vtensor<[2,4],i1>
}

// CHECK-LABEL: func.func @main_ge
// CHECK: mfuse.ge
// CHECK-NOT: torch.aten.ge.Tensor
func.func @main_ge(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],i1> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.ge.Tensor %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],i1>
  return %0 : !torch.vtensor<[2,4],i1>
}

// CHECK-LABEL: func.func @main_gt
// CHECK: mfuse.gt
// CHECK-NOT: torch.aten.gt.Tensor
func.func @main_gt(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],i1> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.gt.Tensor %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],i1>
  return %0 : !torch.vtensor<[2,4],i1>
}

// CHECK-LABEL: func.func @main_le
// CHECK: mfuse.le
// CHECK-NOT: torch.aten.le.Tensor
func.func @main_le(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],i1> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.le.Tensor %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],i1>
  return %0 : !torch.vtensor<[2,4],i1>
}

// CHECK-LABEL: func.func @main_lt
// CHECK: mfuse.lt
// CHECK-NOT: torch.aten.lt.Tensor
func.func @main_lt(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],i1> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.lt.Tensor %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],i1>
  return %0 : !torch.vtensor<[2,4],i1>
}

// CHECK-LABEL: func.func @main_logical_and
// CHECK: mfuse.logical_and
// CHECK-NOT: torch.aten.logical_and
func.func @main_logical_and(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],i1> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.logical_and %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],i1>
  return %0 : !torch.vtensor<[2,4],i1>
}

// CHECK-LABEL: func.func @main_logical_or
// CHECK: mfuse.logical_or
// CHECK-NOT: torch.aten.logical_or
func.func @main_logical_or(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],i1> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.logical_or %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],i1>
  return %0 : !torch.vtensor<[2,4],i1>
}

// CHECK-LABEL: func.func @main_pow
// CHECK: mfuse.pow
// CHECK-NOT: torch.aten.pow.Tensor_Tensor
func.func @main_pow(%arg0: !torch.vtensor<[2,4],f16>, %arg1: !torch.vtensor<[2,4],f32>) -> !torch.vtensor<[2,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.pow.Tensor_Tensor %arg0, %arg1 : !torch.vtensor<[2,4],f16>, !torch.vtensor<[2,4],f32> -> !torch.vtensor<[2,4],f32>
  return %0 : !torch.vtensor<[2,4],f32>
}
