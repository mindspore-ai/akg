// RUN: mfusion-opt %s --canonicalize | FileCheck %s
//
// Redundant mfuse.cast chains: CastOp::fold (identity), CastOp::canonicalize
// (round-trip, same-target, precision-preserving cast(cast(x,T1),T2)->cast(x,T2)).

// CHECK-LABEL: func.func @same_input_same_type_cse
// CHECK: %[[CAST:.*]] = mfuse.cast %arg0 : (tensor<1x4xf32>) -> tensor<1x4xf16>
// CHECK-NOT: mfuse.cast
// CHECK: return %[[CAST]], %[[CAST]] : tensor<1x4xf16>, tensor<1x4xf16>
func.func @same_input_same_type_cse(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf16>, tensor<1x4xf16>) {
  %a = mfuse.cast %arg0 : (tensor<1x4xf32>) -> tensor<1x4xf16>
  %b = mfuse.cast %arg0 : (tensor<1x4xf32>) -> tensor<1x4xf16>
  return %a, %b : tensor<1x4xf16>, tensor<1x4xf16>
}

// CHECK-LABEL: func.func @same_input_different_type_no_cse
// CHECK-COUNT-2: mfuse.cast
func.func @same_input_different_type_no_cse(%arg0: tensor<1x4xf32>)
    -> (tensor<1x4xf16>, tensor<1x4xbf16>) {
  %a = mfuse.cast %arg0 : (tensor<1x4xf32>) -> tensor<1x4xf16>
  %b = mfuse.cast %arg0 : (tensor<1x4xf32>) -> tensor<1x4xbf16>
  return %a, %b : tensor<1x4xf16>, tensor<1x4xbf16>
}

// CHECK-LABEL: func.func @round_trip
// CHECK-NOT: mfuse.cast
// CHECK: {{\s*return %arg0 : tensor<1x4xf32>}}
func.func @round_trip(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %a = mfuse.cast %arg0 : (tensor<1x4xf32>) -> tensor<1x4xf16>
  %b = mfuse.cast %a : (tensor<1x4xf16>) -> tensor<1x4xf32>
  return %b : tensor<1x4xf32>
}

// Inner cast is still used elsewhere: only the outer round-trip cast is removed; inner must stay.
// CHECK-LABEL: func.func @round_trip_inner_shared
// CHECK-DAG: {{\s*%[0-9]+ = mfuse\.cast %arg0 : \(tensor<1x4xf32>\) -> tensor<1x4xf16>}}
// CHECK-DAG: {{\s*return %arg0, %[0-9]+ : tensor<1x4xf32>, tensor<1x4xf16>}}
func.func @round_trip_inner_shared(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<1x4xf16>) {
  %a = mfuse.cast %arg0 : (tensor<1x4xf32>) -> tensor<1x4xf16>
  %b = mfuse.cast %a : (tensor<1x4xf16>) -> tensor<1x4xf32>
  return %b, %a : tensor<1x4xf32>, tensor<1x4xf16>
}

// CHECK-LABEL: func.func @same_target
// CHECK-COUNT-1: mfuse.cast
func.func @same_target(%arg0: tensor<1x4xf32>) -> tensor<1x4xf16> {
  %a = mfuse.cast %arg0 : (tensor<1x4xf32>) -> tensor<1x4xf16>
  %b = mfuse.cast %a : (tensor<1x4xf16>) -> tensor<1x4xf16>
  return %b : tensor<1x4xf16>
}

// CHECK-LABEL: func.func @identity
// CHECK-NOT: mfuse.cast
// CHECK: {{\s*return %arg0 : tensor<1x4xf32>}}
func.func @identity(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %y = mfuse.cast %arg0 : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %y : tensor<1x4xf32>
}

// CHECK-LABEL: func.func @compose_widening_int
// CHECK: {{\s*%[0-9]+ = mfuse\.cast %arg0 : \(tensor<2x3xi8>\) -> tensor<2x3xi64>}}
// CHECK-NEXT: {{\s*return %[0-9]+ : tensor<2x3xi64>}}
func.func @compose_widening_int(%arg0: tensor<2x3xi8>) -> tensor<2x3xi64> {
  %a = mfuse.cast %arg0 : (tensor<2x3xi8>) -> tensor<2x3xi32>
  %b = mfuse.cast %a : (tensor<2x3xi32>) -> tensor<2x3xi64>
  return %b : tensor<2x3xi64>
}

// CHECK-LABEL: func.func @compose_widening_float
// CHECK: {{\s*%[0-9]+ = mfuse\.cast %arg0 : \(tensor<1x4xf16>\) -> tensor<1x4xf64>}}
// CHECK-NEXT: {{\s*return %[0-9]+ : tensor<1x4xf64>}}
func.func @compose_widening_float(%arg0: tensor<1x4xf16>) -> tensor<1x4xf64> {
  %a = mfuse.cast %arg0 : (tensor<1x4xf16>) -> tensor<1x4xf32>
  %b = mfuse.cast %a : (tensor<1x4xf32>) -> tensor<1x4xf64>
  return %b : tensor<1x4xf64>
}

// Narrowing middle: must not fold to a single cast (would change semantics vs two-step).
// CHECK-LABEL: func.func @no_compose_narrow_mid
// CHECK-COUNT-2: mfuse.cast
func.func @no_compose_narrow_mid(%arg0: tensor<1x4xf32>) -> tensor<1x4xf64> {
  %a = mfuse.cast %arg0 : (tensor<1x4xf32>) -> tensor<1x4xf16>
  %b = mfuse.cast %a : (tensor<1x4xf16>) -> tensor<1x4xf64>
  return %b : tensor<1x4xf64>
}

//===----------------------------------------------------------------------===//
// Constant cast folding: foldConstantCast
//===----------------------------------------------------------------------===//

// Constant f32 -> f16: should fold constant and produce tensor<1x6xf16>
// CHECK-LABEL: func.func @constant_cast_f32_to_f16
// CHECK: %[[CST:.*]] = mfuse.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<1x6xf16>
// CHECK-NOT: mfuse.cast
// CHECK: return %[[CST]] : tensor<1x6xf16>
func.func @constant_cast_f32_to_f16() -> tensor<1x6xf16> {
  %cst = mfuse.constant dense<[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]> : tensor<1x6xf32>
  %cast = mfuse.cast %cst : (tensor<1x6xf32>) -> tensor<1x6xf16>
  return %cast : tensor<1x6xf16>
}

// Constant f64 -> f32: should fold and produce tensor<1x2xf32>
// CHECK-LABEL: func.func @constant_cast_f64_to_f32
// CHECK: %[[CST:.*]] = mfuse.constant dense<{{\[\[}}1.500000e+00, 2.500000e+00]]> : tensor<1x2xf32>
// CHECK-NOT: mfuse.cast
// CHECK: return %[[CST]] : tensor<1x2xf32>
func.func @constant_cast_f64_to_f32() -> tensor<1x2xf32> {
  %cst = mfuse.constant dense<[[1.5, 2.5]]> : tensor<1x2xf64>
  %cast = mfuse.cast %cst : (tensor<1x2xf64>) -> tensor<1x2xf32>
  return %cast : tensor<1x2xf32>
}

// Constant f64 -> f16: should fold and produce tensor<3x1xf16>
// CHECK-LABEL: func.func @constant_cast_f64_to_f16
// CHECK: %[[CST:.*]] = mfuse.constant dense<{{\[\[}}1.000000e+00], [2.000000e+00], [3.000000e+00]]> : tensor<3x1xf16>
// CHECK-NOT: mfuse.cast
// CHECK: return %[[CST]] : tensor<3x1xf16>
func.func @constant_cast_f64_to_f16() -> tensor<3x1xf16> {
  %cst = mfuse.constant dense<[[1.0], [2.0], [3.0]]> : tensor<3x1xf64>
  %cast = mfuse.cast %cst : (tensor<3x1xf64>) -> tensor<3x1xf16>
  return %cast : tensor<3x1xf16>
}

// Constant i64 -> i32: should fold and produce tensor<2x2xi32>
// CHECK-LABEL: func.func @constant_cast_i64_to_i32
// CHECK: %[[CST:.*]] = mfuse.constant dense<{{\[\[}}1, 2], [3, 4]]> : tensor<2x2xi32>
// CHECK-NOT: mfuse.cast
// CHECK: return %[[CST]] : tensor<2x2xi32>
func.func @constant_cast_i64_to_i32() -> tensor<2x2xi32> {
  %cst = mfuse.constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi64>
  %cast = mfuse.cast %cst : (tensor<2x2xi64>) -> tensor<2x2xi32>
  return %cast : tensor<2x2xi32>
}

// Constant i32 -> i64: should fold and produce tensor<1x4xi64>
// CHECK-LABEL: func.func @constant_cast_i32_to_i64
// CHECK: %[[CST:.*]] = mfuse.constant dense<{{\[\[}}10, 20, 30, 40]]> : tensor<1x4xi64>
// CHECK-NOT: mfuse.cast
// CHECK: return %[[CST]] : tensor<1x4xi64>
func.func @constant_cast_i32_to_i64() -> tensor<1x4xi64> {
  %cst = mfuse.constant dense<[[10, 20, 30, 40]]> : tensor<1x4xi32>
  %cast = mfuse.cast %cst : (tensor<1x4xi32>) -> tensor<1x4xi64>
  return %cast : tensor<1x4xi64>
}

// Constant f64 -> i64: should fold and produce tensor<2x1xi64>
// CHECK-LABEL: func.func @constant_cast_f64_to_i64
// CHECK: %[[CST:.*]] = mfuse.constant dense<{{\[\[}}1], [2]]> : tensor<2x1xi64>
// CHECK-NOT: mfuse.cast
// CHECK: return %[[CST]] : tensor<2x1xi64>
func.func @constant_cast_f64_to_i64() -> tensor<2x1xi64> {
  %cst = mfuse.constant dense<[[1.5], [2.5]]> : tensor<2x1xf64>
  %cast = mfuse.cast %cst : (tensor<2x1xf64>) -> tensor<2x1xi64>
  return %cast : tensor<2x1xi64>
}

// Constant i64 -> f64: should fold and produce tensor<1x3xf64>
// CHECK-LABEL: func.func @constant_cast_i64_to_f64
// CHECK: %[[CST:.*]] = mfuse.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00]]> : tensor<1x3xf64>
// CHECK-NOT: mfuse.cast
// CHECK: return %[[CST]] : tensor<1x3xf64>
func.func @constant_cast_i64_to_f64() -> tensor<1x3xf64> {
  %cst = mfuse.constant dense<[[1, 2, 3]]> : tensor<1x3xi64>
  %cast = mfuse.cast %cst : (tensor<1x3xi64>) -> tensor<1x3xf64>
  return %cast : tensor<1x3xf64>
}

// Constant i64 -> f32: should fold and produce tensor<3x2xf32>
// CHECK-LABEL: func.func @constant_cast_i64_to_f32
// CHECK: %[[CST:.*]] = mfuse.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]]> : tensor<3x2xf32>
// CHECK-NOT: mfuse.cast
// CHECK: return %[[CST]] : tensor<3x2xf32>
func.func @constant_cast_i64_to_f32() -> tensor<3x2xf32> {
  %cst = mfuse.constant dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>
  %cast = mfuse.cast %cst : (tensor<3x2xi64>) -> tensor<3x2xf32>
  return %cast : tensor<3x2xf32>
}
