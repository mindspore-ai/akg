// RUN: akg-opt --alloc-buffer-shrink -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @alloc_buffer_shrink(%arg0: memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>, %arg1: memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>) -> (memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>) attributes {OperatorType = "Reduction"} {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %cst_0 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:   %cst_1 = arith.constant 9.9999999999999995E-7 : f64
// CHECK-NEXT:   %cst_2 = arith.constant 6.400000e+01 : f32
// CHECK-NEXT:   %cst_3 = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:   %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:   %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:   %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x256xf32, {SymShapeAttr = ["1", "s3"]}>
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %subview = memref.subview %alloc_6[0, 0] [1, 64] [1, 1] : memref<1x256xf32, {SymShapeAttr = ["1", "s3"]}> to memref<1x64xf32, strided<[256, 1]>, {SymShapeAttr = ["1", "s3"]}>
// CHECK-NEXT:   %memspacecast = memref.memory_space_cast %subview : memref<1x64xf32, strided<[256, 1]>, {SymShapeAttr = ["1", "s3"]}> to memref<1x64xf32, strided<[256, 1]>, {SymShapeAttr = ["1", "s1"]}>
// CHECK-NEXT:   %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x64xf32, {SymShapeAttr = ["1", "s1"]}>
// CHECK-NEXT:   %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:   %expand_shape = memref.expand_shape %alloc_4 {{\[\[0, 1\]\]}} output_shape [14646, 1] : memref<14646xf32, {SymShapeAttr = ["s0"]}> into memref<14646x1xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:   %memspacecast_9 = memref.memory_space_cast %expand_shape : memref<14646x1xf32, {SymShapeAttr = ["s0"]}> to memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>
// CHECK-NEXT:   %expand_shape_10 = memref.expand_shape %alloc_5 {{\[\[0, 1\]\]}} output_shape [14646, 1] : memref<14646xf32, {SymShapeAttr = ["s0"]}> into memref<14646x1xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:   %memspacecast_11 = memref.memory_space_cast %expand_shape_10 : memref<14646x1xf32, {SymShapeAttr = ["s0"]}> to memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>
// CHECK-NEXT:   %expand_shape_12 = memref.expand_shape %alloc {{\[\[0, 1\]\]}} output_shape [14646, 1] : memref<14646xf32, {SymShapeAttr = ["s0"]}> into memref<14646x1xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:   %memspacecast_13 = memref.memory_space_cast %expand_shape_12 : memref<14646x1xf32, {SymShapeAttr = ["s0"]}> to memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>
// CHECK-NEXT:   affine.for %arg2 = 0 to 14646 {
// CHECK-NEXT:     %0 = affine.load %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %1 = affine.for %arg3 = 0 to 64 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:       %10 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:       %11 = arith.addf %10, %arg4 {reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK-NEXT:       affine.yield %11 : f32
// CHECK-NEXT:     } {reduction}
// CHECK-NEXT:     affine.store %1, %alloc_4[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %2 = affine.load %alloc_4[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %3 = arith.divf %2, %cst_2 : f32
// CHECK-NEXT:     affine.store %3, %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %4 = affine.for %arg3 = 0 to 64 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:       %10 = affine.load %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:       %11 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:       %12 = arith.subf %11, %10 : f32
// CHECK-NEXT:       %13 = math.powf %12, %cst_3 : f32
// CHECK-NEXT:       %14 = arith.addf %13, %arg4 {reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK-NEXT:       affine.yield %14 : f32
// CHECK-NEXT:     } {reduction}
// CHECK-NEXT:     affine.store %4, %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %5 = affine.load %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %6 = arith.divf %5, %cst_2 : f32
// CHECK-NEXT:     %7 = arith.truncf %cst_1 : f64 to f32
// CHECK-NEXT:     %8 = arith.addf %6, %7 : f32
// CHECK-NEXT:     %9 = math.rsqrt %8 : f32
// CHECK-NEXT:     affine.for %arg3 = 0 to 64 {
// CHECK-NEXT:       %10 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:       %11 = affine.load %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:       %12 = arith.subf %10, %11 : f32
// CHECK-NEXT:       %13 = arith.mulf %12, %9 : f32
// CHECK-NEXT:       affine.store %13, %alloc_7[%c0, %arg3] : memref<1x64xf32, {SymShapeAttr = ["1", "s1"]}>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg3 = 0 to 256 {
// CHECK-NEXT:       %10 = affine.load %arg1[%arg2, %arg3] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
// CHECK-NEXT:       %11 = arith.negf %10 : f32
// CHECK-NEXT:       %12 = math.exp %11 : f32
// CHECK-NEXT:       %13 = arith.addf %12, %cst_0 : f32
// CHECK-NEXT:       %14 = arith.divf %cst_0, %13 : f32
// CHECK-NEXT:       %15 = affine.load %arg1[%arg2, %arg3] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
// CHECK-NEXT:       %16 = arith.mulf %15, %14 : f32
// CHECK-NEXT:       affine.store %16, %alloc_6[%c0, %arg3] : memref<1x256xf32, {SymShapeAttr = ["1", "s3"]}>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg3 = 0 to 64 {
// CHECK-NEXT:       %10 = affine.load %memspacecast[%c0, %arg3] : memref<1x64xf32, strided<[256, 1]>, {SymShapeAttr = ["1", "s1"]}>
// CHECK-NEXT:       %11 = affine.load %alloc_7[%c0, %arg3] : memref<1x64xf32, {SymShapeAttr = ["1", "s1"]}>
// CHECK-NEXT:       %12 = arith.mulf %10, %11 : f32
// CHECK-NEXT:       affine.store %12, %alloc_8[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %memspacecast_9, %memspacecast_11, %memspacecast_13, %alloc_8 : memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT: }


func.func @alloc_buffer_shrink(%arg0: memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>, %arg1: memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>) -> (memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>) attributes {OperatorType = "Reduction"} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 9.9999999999999995E-7 : f64
  %cst_2 = arith.constant 6.400000e+01 : f32
  %cst_3 = arith.constant 2.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<14646xf32, {SymShapeAttr = ["s0"]}>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<14646xf32, {SymShapeAttr = ["s0"]}>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<14646xf32, {SymShapeAttr = ["s0"]}>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
  %subview = memref.subview %alloc_6[0, 0] [14646, 64] [1, 1] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}> to memref<14646x64xf32, strided<[256, 1]>, {SymShapeAttr = ["s0", "s3"]}>
  %memspacecast = memref.memory_space_cast %subview : memref<14646x64xf32, strided<[256, 1]>, {SymShapeAttr = ["s0", "s3"]}> to memref<14646x64xf32, strided<[256, 1]>, {SymShapeAttr = ["s0", "s1"]}>
  %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
  %expand_shape = memref.expand_shape %alloc_4 [[0, 1]] output_shape [14646, 1] : memref<14646xf32, {SymShapeAttr = ["s0"]}> into memref<14646x1xf32, {SymShapeAttr = ["s0"]}>
  %memspacecast_9 = memref.memory_space_cast %expand_shape : memref<14646x1xf32, {SymShapeAttr = ["s0"]}> to memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>
  %expand_shape_10 = memref.expand_shape %alloc_5 [[0, 1]] output_shape [14646, 1] : memref<14646xf32, {SymShapeAttr = ["s0"]}> into memref<14646x1xf32, {SymShapeAttr = ["s0"]}>
  %memspacecast_11 = memref.memory_space_cast %expand_shape_10 : memref<14646x1xf32, {SymShapeAttr = ["s0"]}> to memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>
  %expand_shape_12 = memref.expand_shape %alloc [[0, 1]] output_shape [14646, 1] : memref<14646xf32, {SymShapeAttr = ["s0"]}> into memref<14646x1xf32, {SymShapeAttr = ["s0"]}>
  %memspacecast_13 = memref.memory_space_cast %expand_shape_12 : memref<14646x1xf32, {SymShapeAttr = ["s0"]}> to memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>
  affine.for %arg2 = 0 to 14646 {
    %0 = affine.load %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    %1 = affine.for %arg3 = 0 to 64 iter_args(%arg4 = %cst) -> (f32) {
      %10 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
      %11 = arith.addf %10, %arg4 {reduction_axes = [1 : index], reduction_type = "x"} : f32
      affine.yield %11 : f32
    } {reduction}
    affine.store %1, %alloc_4[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    %2 = affine.load %alloc_4[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    %3 = arith.divf %2, %cst_2 : f32
    affine.store %3, %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    %4 = affine.for %arg3 = 0 to 64 iter_args(%arg4 = %cst) -> (f32) {
      %10 = affine.load %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
      %11 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
      %12 = arith.subf %11, %10 : f32
      %13 = math.powf %12, %cst_3 : f32
      %14 = arith.addf %13, %arg4 {reduction_axes = [1 : index], reduction_type = "x"} : f32
      affine.yield %14 : f32
    } {reduction}
    affine.store %4, %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    %5 = affine.load %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    %6 = arith.divf %5, %cst_2 : f32
    %7 = arith.truncf %cst_1 : f64 to f32
    %8 = arith.addf %6, %7 : f32
    %9 = math.rsqrt %8 : f32
    affine.for %arg3 = 0 to 64 {
      %10 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
      %11 = affine.load %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
      %12 = arith.subf %10, %11 : f32
      %13 = arith.mulf %12, %9 : f32
      affine.store %13, %alloc_7[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
    }
    affine.for %arg3 = 0 to 256 {
      %10 = affine.load %arg1[%arg2, %arg3] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
      %11 = arith.negf %10 : f32
      %12 = math.exp %11 : f32
      %13 = arith.addf %12, %cst_0 : f32
      %14 = arith.divf %cst_0, %13 : f32
      %15 = affine.load %arg1[%arg2, %arg3] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
      %16 = arith.mulf %15, %14 : f32
      affine.store %16, %alloc_6[%arg2, %arg3] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
    }
    affine.for %arg3 = 0 to 64 {
      %10 = affine.load %memspacecast[%arg2, %arg3] : memref<14646x64xf32, strided<[256, 1]>, {SymShapeAttr = ["s0", "s1"]}>
      %11 = affine.load %alloc_7[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
      %12 = arith.mulf %10, %11 : f32
      affine.store %12, %alloc_8[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
    }
  }
  return %memspacecast_9, %memspacecast_11, %memspacecast_13, %alloc_8 : memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
}
