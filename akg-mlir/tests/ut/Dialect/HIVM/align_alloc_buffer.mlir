// RUN: akg-opt %s --align-alloc-buffer | FileCheck %s

// `align-alloc-buffer` is anchored on `func::FuncOp`. Nested `module { func ... }`
// puts the function under a child module, so the pass never runs on it (IR unchanged).
// Keep `func.func` as direct children of the implicit top-level module.

// Last-axis 2D vtranspose: 32-byte BL marks; 4x15 f32 keeps base collect -> src pad 8x16, dst pad 16x8.
// CHECK-LABEL: func.func @transpose_alloc_mark
// CHECK-DAG: memref.alloc() : memref<8x16xf32>
// CHECK-DAG: memref.alloc() : memref<16x8xf32>
// CHECK-DAG: memref.subview
// CHECK-DAG: memref.subview
func.func @transpose_alloc_mark() {
  %src = memref.alloc() : memref<4x15xf32>
  %dst = memref.alloc() : memref<15x4xf32>
  hivm.hir.vtranspose ins(%src : memref<4x15xf32>) outs(%dst : memref<15x4xf32>) permutation = [1, 0]
  return
}

// i32 -> i8 vcast path marks root allocs (i32/i16 -> i8 only); expect pad + subview.
// Dst rank-1 uses pushVcastDstMarks: dstHwAlign = (32/elemBytes)^2 bytes → 1024 for i8, so alloc is 1024xi8.
// CHECK-LABEL: func.func @vcast_alloc_mark
// CHECK-DAG: memref.alloc() : memref<16xi32>
// CHECK-DAG: memref.alloc() : memref<1024xi8>
// CHECK-DAG: memref.subview
// CHECK-DAG: memref.subview
func.func @vcast_alloc_mark() {
  %src = memref.alloc() : memref<15xi32>
  %dst = memref.alloc() : memref<15xi8>
  hivm.hir.vcast ins(%src : memref<15xi32>) outs(%dst : memref<15xi8>) round_mode = <truncwithoverflow>
  return
}

// CHECK-NOT: akg.alloc_align_dims
