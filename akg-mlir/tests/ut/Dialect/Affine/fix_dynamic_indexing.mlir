// RUN: akg-opt %s  -split-input-file -fix-dynamic-indexing | FileCheck %s


func.func @elem_broadcast_last_5(%arg0: memref<4096x?xf32>, %arg1: memref<?xf32>, %arg2: memref<1x?xf32>, %arg3: memref<4096x?xf32>) 
  attributes {OperatorType = "Reshape", enable_atomic_add = false, mindspore_kernel, need_fix = [[0, 1], [1], [0, 1]], process = "cuda"} {
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg2, %c1 : memref<1x?xf32>
  %expand_shape = memref.expand_shape %arg0 [[0], [1, 2]] : memref<4096x?xf32> into memref<4096x1x?xf32>
  %expand_shape_0 = memref.expand_shape %arg1 [[0, 1]] : memref<?xf32> into memref<1x?xf32>
  %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<4096x1x?xf32>
  %dim_1 = memref.dim %arg2, %c1 : memref<1x?xf32>
  affine.for %arg4 = 0 to 4096 {
    affine.for %arg5 = 0 to 1 {
      affine.for %arg6 = 0 to %dim_1 {
        //CHECK: affine.if
        %0 = affine.load %arg2[%arg5, %arg6] : memref<1x?xf32>
        //CHECK: affine.if
        %1 = affine.load %expand_shape[%arg4, %arg5, %arg6] : memref<4096x1x?xf32>
        //CHECK: affine.if
        %2 = affine.load %expand_shape_0[%arg5, %arg6] : memref<1x?xf32>
        %3 = arith.addf %1, %2 : f32
        %4 = arith.mulf %0, %3 : f32
        affine.store %4, %alloc[%arg4, %arg5, %arg6] : memref<4096x1x?xf32>
      }
    }
  }
  %collapse_shape = memref.collapse_shape %alloc [[0], [1, 2]] : memref<4096x1x?xf32> into memref<4096x?xf32>
  memref.copy %collapse_shape, %arg3 : memref<4096x?xf32> to memref<4096x?xf32>
  return
}

//---

func.func @Fused_Add_Sub_fusion_2263688444547730434(%arg0: memref<4096x7680xf32>, %arg1: memref<?x?xf32>, %arg2: memref<4096x7680xf32>, %arg3: memref<4096x7680xf32>)
 attributes {OperatorType = "Elementwise", mindspore_kernel, process = "gpu", need_fix = [[0, 0], [1, 1], [0, 0]]} {
  affine.for %arg4 = 0 to 4096 {
    affine.for %arg5 = 0 to 7680 {
      %0 = memref.load %arg0[%arg4, %arg5] : memref<4096x7680xf32>
      //CHECK: affine.if
      %1 = memref.load %arg1[%arg4, %arg5] : memref<?x?xf32>
      %2 = memref.load %arg2[%arg4, %arg5] : memref<4096x7680xf32>
      %3 = arith.addf %0, %1 : f32
      %4 = arith.subf %3, %2 : f32
      memref.store %4, %arg3[%arg4, %arg5] : memref<4096x7680xf32>
    }
  }
  return
}
