// RUN: akg-opt %s --broadcast-loop-hoist | FileCheck %s

// CHECK-LABEL:  module {
// CHECK:  func.func @MODEL_NAME(%arg0: memref<8192xi64, {SymShapeAttr = ["s0"]}>, %arg1: memref<64xf32, {SymShapeAttr = ["s1"]}>) -> memref<8192x2x64xf32, {SymShapeAttr = ["s0", "s13", "s1"]}> attributes {OperatorType = "Broadcast", hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK:    %cst = arith.constant 0.000000e+00 : f32
// CHECK:    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8192x2x64xf32, {SymShapeAttr = ["s0", "s13", "s1"]}>
// CHECK:    affine.for %arg2 = 0 to 8192 {
// CHECK:      %0 = affine.load %arg0[%arg2] : memref<8192xi64, {SymShapeAttr = ["s0"]}>
// CHECK:      %1 = arith.sitofp %0 : i64 to f32
// CHECK:      %2 = arith.addf %1, %cst : f32
// CHECK:      affine.for %arg3 = 0 to 2 {
// CHECK:        affine.for %arg4 = 0 to 64 {
// CHECK:          %3 = affine.load %arg1[%arg4] : memref<64xf32, {SymShapeAttr = ["s1"]}>
// CHECK:          %4 = arith.mulf %2, %3 : f32
// CHECK:          affine.store %4, %alloc[%arg2, %arg3, %arg4] : memref<8192x2x64xf32, {SymShapeAttr = ["s0", "s13", "s1"]}>
// CHECK:        } {broadcast}
// CHECK:      } {broadcast}
// CHECK:    } {broadcast}
// CHECK:    return %alloc : memref<8192x2x64xf32, {SymShapeAttr = ["s0", "s13", "s1"]}>
// CHECK:  }
// CHECK:}


func.func @MODEL_NAME(%arg0: memref<8192xi64, {SymShapeAttr = ["s0"]}>, %arg1: memref<64xf32, {SymShapeAttr = ["s1"]}>) -> memref<8192x2x64xf32, {SymShapeAttr = ["s0", "s13", "s1"]}> attributes {OperatorType = "Broadcast", hacc.function_kind = #hacc.function_kind<HOST>} {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8192x2x64xf32, {SymShapeAttr = ["s0", "s13", "s1"]}>
  affine.for %arg2 = 0 to 8192 {
    %0 = affine.load %arg0[%arg2] : memref<8192xi64, {SymShapeAttr = ["s0"]}>
    %1 = arith.sitofp %0 : i64 to f32
    %2 = arith.addf %1, %cst : f32
    affine.for %arg3 = 0 to 64 {
      %3 = affine.load %arg1[%arg3] : memref<64xf32, {SymShapeAttr = ["s1"]}>
      %4 = arith.mulf %2, %3 : f32
      affine.for %arg4 = 0 to 2 {
        affine.store %4, %alloc[%arg2, %arg4, %arg3] : memref<8192x2x64xf32, {SymShapeAttr = ["s0", "s13", "s1"]}>
      } {broadcast}
    } {broadcast}
  } {broadcast}
  return %alloc : memref<8192x2x64xf32, {SymShapeAttr = ["s0", "s13", "s1"]}>
}