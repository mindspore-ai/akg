// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse -decompose="pattern-type=BEFORE_MANUAL_FUSION" --convert-mfuse-to-torch --canonicalize | FileCheck %s

// Test: f32 + f64_scalar -> f32, decompose, back to torch
// CHECK-LABEL: func.func @test_f32_div_f64_scalar
// CHECK: torch.aten.div.Scalar {{.*}} -> !torch.vtensor<[4,4],f32>
func.func @test_f32_div_f64_scalar(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.float) -> !torch.vtensor<[4,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.div.Scalar %arg0, %arg1 : !torch.vtensor<[4,4],f32>, !torch.float -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}

// Test: i32 + i64_scalar -> i32, decompose, back to torch
// CHECK-LABEL: func.func @test_i32_add_i64_scalar
// CHECK: torch.aten.add.Scalar {{.*}} -> !torch.vtensor<[4,4],si32>
func.func @test_i32_add_i64_scalar(%arg0: !torch.vtensor<[4,4],si32>, %arg1: !torch.int) -> !torch.vtensor<[4,4],si32> attributes {torch.assume_strict_symbolic_shapes} {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Scalar %arg0, %arg1, %int1 : !torch.vtensor<[4,4],si32>, !torch.int, !torch.int -> !torch.vtensor<[4,4],si32>
  return %0 : !torch.vtensor<[4,4],si32>
}

// Test: f16 + f32 -> f32 promotion, decompose, back to torch
// CHECK-LABEL: func.func @test_f16_mul_f32
// CHECK: torch.aten.mul.Tensor {{.*}} -> !torch.vtensor<[4,4],f32>
func.func @test_f16_mul_f32(%arg0: !torch.vtensor<[4,4],f16>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4,4],f16>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}

// Test: f32 + f64 -> f64 promotion, decompose, back to torch
// CHECK-LABEL: func.func @test_f32_mul_f64
// CHECK: torch.aten.mul.Tensor {{.*}} -> !torch.vtensor<[4,4],f64>
func.func @test_f32_mul_f64(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4,4],f64>) -> !torch.vtensor<[4,4],f64> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f64> -> !torch.vtensor<[4,4],f64>
  return %0 : !torch.vtensor<[4,4],f64>
}

// Test: i32 + i64 -> i64 promotion, decompose, back to torch
// CHECK-LABEL: func.func @test_i32_add_i64
// CHECK: torch.aten.add.Tensor {{.*}} -> !torch.vtensor<[4,4],si64>
func.func @test_i32_add_i64(%arg0: !torch.vtensor<[4,4],si32>, %arg1: !torch.vtensor<[4,4],si64>) -> !torch.vtensor<[4,4],si64> attributes {torch.assume_strict_symbolic_shapes} {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,4],si32>, !torch.vtensor<[4,4],si64>, !torch.int -> !torch.vtensor<[4,4],si64>
  return %0 : !torch.vtensor<[4,4],si64>
}

// Test: si64 + i64 -> si64 promotion, decompose, back to torch
// CHECK-LABEL: func.func @test_si64_add_i64
// CHECK: torch.aten.add.Scalar {{.*}} -> !torch.vtensor<[4,4],si64>
func.func @test_si64_add_i64(%arg0: !torch.vtensor<[4,4],si64>, %arg1: !torch.int) -> !torch.vtensor<[4,4],si64> attributes {torch.assume_strict_symbolic_shapes} {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Scalar %arg0, %arg1, %int1 : !torch.vtensor<[4,4],si64>, !torch.int, !torch.int -> !torch.vtensor<[4,4],si64>
  return %0 : !torch.vtensor<[4,4],si64>
}

// Test: int + float -> float promotion, decompose, back to torch
// CHECK-LABEL: func.func @test_int_mul_float
// CHECK: torch.aten.mul.Tensor {{.*}} -> !torch.vtensor<[4,4],f32>
func.func @test_int_mul_float(%arg0: !torch.vtensor<[4,4],si32>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4,4],si32>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}

// Test: bf16 + f32 -> f32 promotion, decompose, back to torch
// CHECK-LABEL: func.func @test_bf16_mul_f32
// CHECK: torch.aten.mul.Tensor {{.*}} -> !torch.vtensor<[4,4],f32>
func.func @test_bf16_mul_f32(%arg0: !torch.vtensor<[4,4],bf16>, %arg1: !torch.vtensor<[4,4],f32>) -> !torch.vtensor<[4,4],f32> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[4,4],bf16>, !torch.vtensor<[4,4],f32> -> !torch.vtensor<[4,4],f32>
  return %0 : !torch.vtensor<[4,4],f32>
}

// Test: comparison operation returns i1, decompose, back to torch
// CHECK-LABEL: func.func @test_compare_type
// CHECK: torch.aten.gt.Tensor {{.*}} -> !torch.vtensor<[4,4],i1>
func.func @test_compare_type(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.vtensor<[4,4],f16>) -> !torch.vtensor<[4,4],i1> attributes {torch.assume_strict_symbolic_shapes} {
  %0 = torch.aten.gt.Tensor %arg0, %arg1 : !torch.vtensor<[4,4],f32>, !torch.vtensor<[4,4],f16> -> !torch.vtensor<[4,4],i1>
  return %0 : !torch.vtensor<[4,4],i1>
}

// Test: ui8 + i16 -> i16 promotion (ui8 has lower priority than i16)
// CHECK-LABEL: func.func @test_ui8_add_i16
// CHECK: torch.aten.add.Tensor {{.*}} -> !torch.vtensor<[4,4],si16>
func.func @test_ui8_add_i16(%arg0: !torch.vtensor<[4,4],ui8>, %arg1: !torch.vtensor<[4,4],si16>) -> !torch.vtensor<[4,4],si16> attributes {torch.assume_strict_symbolic_shapes} {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,4],ui8>, !torch.vtensor<[4,4],si16>, !torch.int -> !torch.vtensor<[4,4],si16>
  return %0 : !torch.vtensor<[4,4],si16>
}

// Test: i8 + ui8 -> si16 (mixed signed/unsigned -> promoted to 16-bit signed)
func.func @test_i8_add_ui8(%arg0: !torch.vtensor<[4,4],si8>, %arg1: !torch.vtensor<[4,4],ui8>) -> !torch.vtensor<[4,4],si16> attributes {torch.assume_strict_symbolic_shapes} {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,4],si8>, !torch.vtensor<[4,4],ui8>, !torch.int -> !torch.vtensor<[4,4],si16>
  return %0 : !torch.vtensor<[4,4],si16>
}

// Test: i64 + bf16 -> bf16 promotion (i64 has lower priority than bf16)
// CHECK-LABEL: func.func @test_i64_add_bf16
// CHECK: torch.aten.add.Tensor {{.*}} -> !torch.vtensor<[4,4],bf16>
func.func @test_i64_add_bf16(%arg0: !torch.vtensor<[4,4],si64>, %arg1: !torch.vtensor<[4,4],bf16>) -> !torch.vtensor<[4,4],bf16> attributes {torch.assume_strict_symbolic_shapes} {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,4],si64>, !torch.vtensor<[4,4],bf16>, !torch.int -> !torch.vtensor<[4,4],bf16>
  return %0 : !torch.vtensor<[4,4],bf16>
}

// Test: ui8 + i64 -> i64 promotion
// CHECK-LABEL: func.func @test_ui8_add_i64
// CHECK: torch.aten.add.Tensor {{.*}} -> !torch.vtensor<[4,4],si64>
func.func @test_ui8_add_i64(%arg0: !torch.vtensor<[4,4],ui8>, %arg1: !torch.vtensor<[4,4],si64>) -> !torch.vtensor<[4,4],si64> attributes {torch.assume_strict_symbolic_shapes} {
  %int1 = torch.constant.int 1
  %0 = torch.aten.add.Tensor %arg0, %arg1, %int1 : !torch.vtensor<[4,4],ui8>, !torch.vtensor<[4,4],si64>, !torch.int -> !torch.vtensor<[4,4],si64>
  return %0 : !torch.vtensor<[4,4],si64>
}
