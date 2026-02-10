// RUN: akg-opt %s --normalize | FileCheck %s

// CHECK-LABEL: func.func @Fused_Pow_fusion_14397231247954220179(%arg0: memref<1xbf16>) -> memref<1xbf16> attributes {OperatorType = "Default", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
// CHECK:  %c-1_i32 = arith.constant -1 : i32
// CHECK:  %c31_i32 = arith.constant 31 : i32
// CHECK:  %cst = arith.constant 1.88016488E+14 : f32
// CHECK:  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xbf16>
// CHECK:  %0 = affine.load %arg0[0] : memref<1xbf16>
// CHECK:  %1 = arith.extf %0 : bf16 to f32
// CHECK:  %2 = arith.bitcast %1 : f32 to i32
// CHECK:  %3 = arith.shrsi %2, %c31_i32 : i32
// CHECK:  %4 = arith.cmpi eq, %3, %c-1_i32 : i32
// CHECK:  %5 = math.absf %1 : f32
// CHECK:  %6 = math.log %5 : f32
// CHECK:  %7 = arith.mulf %6, %cst : f32
// CHECK:  %8 = math.exp %7 : f32
// CHECK:  %9 = math.absf %1 : f32
// CHECK:  %10 = math.log %9 : f32
// CHECK:  %11 = arith.mulf %10, %cst : f32
// CHECK:  %12 = math.exp %11 : f32
// CHECK:  %13 = arith.select %4, %8, %12 : f32
// CHECK:  %14 = arith.truncf %13 : f32 to bf16
// CHECK:  affine.store %14, %alloc[0] : memref<1xbf16>
// CHECK:  return %alloc : memref<1xbf16>
// CHECK:}

func.func @Fused_Pow_fusion_14397231247954220179(%arg0: memref<1xbf16>) -> memref<1xbf16> attributes {OperatorType = "Default", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
%cst = arith.constant 1.88016488E+14 : f32
%alloc = memref.alloc() {alignment = 64 : i64} : memref<1xbf16>
%0 = affine.load %arg0[0] : memref<1xbf16>
%1 = arith.extf %0 : bf16 to f32
%2 = math.powf %1, %cst : f32
%3 = arith.truncf %2 : f32 to bf16
affine.store %3, %alloc[0] : memref<1xbf16>
return %alloc : memref<1xbf16>
}