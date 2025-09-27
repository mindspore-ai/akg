
//RUN: akg-opt %s -linalg-ext-lower | FileCheck %s
//CHECK: affine_map<(d0, d1)[s0] -> (8, d0 - d1)>
//CHECK: memref.alloc() {alignment = 64 : i64} : memref<20x6x23xf32>
//CHECK: affine.for %arg2 = 0 to 4 {
//CHECK: affine.for %arg3 = 0 to 5 {
//CHECK: memref.load %arg1[%arg2, %arg3] : memref<4x5xi32>
//CHECK: arith.index_cast %0 : i32 to index
//CHECK: affine.for %arg4 = 0 to 6 {
//CHECK: affine.for %arg5 = 0 to 23 step 8 {
//CHECK: affine.min #map(%c23, %arg5)[%c8]
//CHECK: vector.create_mask %2 : vector<8xi1>
//CHECK: vector.transfer_read %arg0[%arg2, %arg3, %arg4, %arg5], %cst, %3 : memref<4x5x6x23xf32>, vector<8xf32>
//CHECK: vector.transfer_read %alloc[%1, %arg4, %arg5], %cst, %3 : memref<20x6x23xf32>, vector<8xf32>
//CHECK: arith.addf %5, %4 : vector<8xf32>
//CHECK: vector.transfer_write %6, %alloc[%1, %arg4, %arg5], %3 : vector<8xf32>, memref<20x6x23xf32>

module {
  func.func @unsorted_segment_sum(%arg0: memref<4x5x6x23xf32>, %arg1: memref<4x5xi32>) -> memref<20x6x23xf32> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<20x6x23xf32>
    linalgExt.unsorted_segment_sum %arg0, %arg1, %alloc {num_segments = 20 : i64} : memref<4x5x6x23xf32>, memref<4x5xi32>, memref<20x6x23xf32>
    return %alloc : memref<20x6x23xf32>
  }
}
