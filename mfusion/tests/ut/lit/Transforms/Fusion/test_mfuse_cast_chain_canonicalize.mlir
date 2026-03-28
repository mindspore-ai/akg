// RUN: mfusion-opt %s --canonicalize | FileCheck %s
//
// Redundant mfuse.cast chains: CastOp::fold (identity), CastOp::canonicalize
// (round-trip, same-target, precision-preserving cast(cast(x,T1),T2)->cast(x,T2)).

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
