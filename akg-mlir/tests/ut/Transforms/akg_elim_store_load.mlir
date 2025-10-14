// RUN: akg-opt --elim-store-load   -split-input-file %s | FileCheck %s

// Test StoreLoadElimPass
//===----------------------------------------------------------------------===//
// Test removing an immediate affine store load pair from the MLIR code
//===----------------------------------------------------------------------===//


func.func @test_removal(%arg0: memref<128x768xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x768xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128xf32>
  affine.for %arg3 = 0 to 128 {
    affine.for %arg4 = 0 to 768 {
      %0 = affine.load %arg0[%arg3, %arg4] : memref<128x768xf32>
      %1 = affine.load %arg1[%arg3] : memref<128xf32>
      %2 = arith.subf %0, %1 : f32
      %3 = arith.mulf %2, %2 : f32
      affine.store %3, %alloc[%arg3, %arg4] : memref<128x768xf32>
      %4 = affine.load %alloc[%arg3, %arg4] : memref<128x768xf32>
      %5 = affine.load %alloc_0[%arg3] : memref<128xf32>
      %6 = arith.addf %4, %5 {reduction_axes = [2 : index], reduction_type = "x"} : f32
      affine.store %6, %alloc_0[%arg3] : memref<128xf32>
    }
  }
  memref.copy %alloc_0, %arg2 : memref<128xf32> to memref<128xf32>
  return
}
// CHECK-LABEL: @test_removal
// CHECK-NOT: affine.store {{.*}}[%alloc]
// CHECK-NOT: affine.load {{.*}}[%alloc]
