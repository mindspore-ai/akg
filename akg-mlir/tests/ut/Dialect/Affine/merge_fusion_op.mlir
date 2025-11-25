// RUN: akg-opt %s --merge-fusion-op="merge-specific-op=true" | FileCheck %s

// CHECK: #set = affine_set<(d0) : (-d0 + 14 == 0)>
// CHECK: #set1 = affine_set<(d0) : (-d0 + 31 == 0)>
// CHECK-LABEL: func.func @Fused_ReduceSum_Mul_split_14504185833015688388(%arg0: memref<32x15xf32>, %arg1: memref<f32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
// CHECK-NEXT:   %cst = arith.constant -3.125000e-02 : f32
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
// CHECK-NEXT:   %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<f32>
// CHECK-NEXT:   affine.for %arg2 = 0 to 32 {
// CHECK-NEXT:     affine.for %arg3 = 0 to 15 {
// CHECK-NEXT:       %0 = affine.load %arg0[%arg2, %arg3] : memref<32x15xf32>
// CHECK-NEXT:       %1 = affine.load %alloc[] : memref<f32>
// CHECK-NEXT:       %2 = arith.addf %0, %1 {reduction_axes = [0 : index, 1 : index], reduction_type = "all"} : f32
// CHECK-NEXT:       affine.store %2, %alloc[] : memref<f32>
// CHECK-NEXT:       affine.if #set(%arg3) {
// CHECK-NEXT:         affine.if #set1(%arg2) {
// CHECK-NEXT:           %3 = affine.load %alloc[] : memref<f32>
// CHECK-NEXT:           %4 = arith.mulf %3, %cst : f32
// CHECK-NEXT:           affine.store %4, %alloc_0[] : memref<f32>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   memref.copy %alloc_0, %arg1 : memref<f32> to memref<f32>
// CHECK-NEXT:   return
// CHECK-NEXT: }



func.func @Fused_ReduceSum_Mul_split_14504185833015688388(%arg0: memref<32x15xf32>, %arg1: memref<f32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
  %cst = arith.constant -3.125000e-02 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<f32>
  affine.for %arg2 = 0 to 32 {
    affine.for %arg3 = 0 to 15 {
      %2 = affine.load %arg0[%arg2, %arg3] : memref<32x15xf32>
      %3 = affine.load %alloc[] : memref<f32>
      %4 = arith.addf %2, %3 {reduction_axes = [0 : index, 1 : index], reduction_type = "all"} : f32
      affine.store %4, %alloc[] : memref<f32>
    }
  }
  %0 = affine.load %alloc[] : memref<f32>
  %1 = arith.mulf %0, %cst : f32
  affine.store %1, %alloc_0[] : memref<f32>
  memref.copy %alloc_0, %arg1 : memref<f32> to memref<f32>
  return
}