
//RUN: akg-opt %s -linalg-ext-lower | FileCheck %s
//CHECK: memref.alloc() {alignment = 64 : i64} : memref<4x8x9x6x7xf32>
//CHECK: affine.for %arg2 = 0 to 8 {
//CHECK: affine.for %arg3 = 0 to 9 {
//CHECK: memref.load %arg1[%arg2, %arg3] : memref<8x9xi32>
//CHECK: arith.index_cast %0 : i32 to index
//CHECK: affine.for %arg4 = 0 to 4 {
//CHECK: affine.for %arg5 = 0 to 6 {
//CHECK: affine.for %arg6 = 0 to 7 {
//CHECK: memref.load %arg0[%arg4, %1, %arg5, %arg6] : memref<4x5x6x7xf32>
//CHECK: memref.store %2, %alloc[%arg4, %arg2, %arg3, %arg5, %arg6] : memref<4x8x9x6x7xf32>

module {
  func.func @gather_float(%arg0: memref<4x5x6x7xf32>, %arg1: memref<8x9xi32>) -> memref<4x8x9x6x7xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x8x9x6x7xf32>
    linalgExt.gather %arg0, %arg1, %alloc {axis = 1 : i64} : memref<4x5x6x7xf32>, memref<8x9xi32>, memref<4x8x9x6x7xf32>
    return %alloc : memref<4x8x9x6x7xf32>
  }
}