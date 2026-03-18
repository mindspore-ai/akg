// RUN: akg-opt %s --reduction-sibling-recompute | FileCheck %s

// CHECK-LABEL: func.func @akg_fused_add_mean_mul_pow_rsqrt_10_auto_fallback(%arg0: memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>, %arg1: memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>, %arg2: memref<2048xf32, {SymShapeAttr = ["s2"]}>) -> (memref<256x10x1xf32, {SymShapeAttr = ["s0", "s1", "1"]}>, memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>) attributes {OperatorType = "Reduction", hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-NEXT:    %cst = arith.constant 2.048000e+03 : f32
// CHECK-NEXT:    %cst_0 = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:    %cst_1 = arith.constant 1.000000e-05 : f64
// CHECK-NEXT:    %cst_2 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %collapse_shape = memref.collapse_shape %arg0 {{\[\[0, 1\], \[2\]\]}} : memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}> into memref<2560x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>
// CHECK-NEXT:    %memspacecast = memref.memory_space_cast %collapse_shape : memref<2560x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}> to memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:    %collapse_shape_3 = memref.collapse_shape %arg1 {{\[\[0, 1\], \[2\]\]}} : memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}> into memref<2560x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>
// CHECK-NEXT:    %memspacecast_4 = memref.memory_space_cast %collapse_shape_3 : memref<2560x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}> to memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<2560xf32, {SymShapeAttr = ["s33"]}>
// CHECK-NEXT:    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:    %expand_shape = memref.expand_shape %alloc_5 {{\[\[0, 1, 2\]\]}} output_shape [256, 10, 1] : memref<2560xf32, {SymShapeAttr = ["s33"]}> into memref<256x10x1xf32, {SymShapeAttr = ["s33"]}>
// CHECK-NEXT:    %memspacecast_7 = memref.memory_space_cast %expand_shape : memref<256x10x1xf32, {SymShapeAttr = ["s33"]}> to memref<256x10x1xf32, {SymShapeAttr = ["s0", "s1", "1"]}>
// CHECK-NEXT:    %expand_shape_8 = memref.expand_shape %alloc_6 {{\[\[0, 1\], \[2\]\]}} output_shape [256, 10, 2048] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}> into memref<256x10x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:    %memspacecast_9 = memref.memory_space_cast %expand_shape_8 : memref<256x10x2048xf32, {SymShapeAttr = ["s33", "s2"]}> to memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>
// CHECK-NEXT:    affine.for %arg3 = 0 to 2560 {
// CHECK-NEXT:      affine.store %cst_2, %alloc_5[%arg3] {reduction_init} : memref<2560xf32, {SymShapeAttr = ["s33"]}>
// CHECK-NEXT:      affine.for %arg4 = 0 to 2048 {
// CHECK-NEXT:        %5 = affine.load %memspacecast[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:        %6 = affine.load %memspacecast_4[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:        %7 = arith.addf %5, %6 : f32
// CHECK-NEXT:        affine.store %7, %alloc[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:        %8 = affine.load %alloc[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:        %9 = math.powf %8, %cst_0 : f32
// CHECK-NEXT:        %10 = affine.load %alloc_5[%arg3] : memref<2560xf32, {SymShapeAttr = ["s33"]}>
// CHECK-NEXT:        %11 = arith.addf %9, %10 {reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK-NEXT:        affine.store %11, %alloc_5[%arg3] : memref<2560xf32, {SymShapeAttr = ["s33"]}>
// CHECK-NEXT:      } {reduction}
// CHECK-NEXT:      %0 = affine.load %alloc_5[%arg3] : memref<2560xf32, {SymShapeAttr = ["s33"]}>
// CHECK-NEXT:      %1 = arith.divf %0, %cst : f32
// CHECK-NEXT:      %2 = arith.truncf %cst_1 : f64 to f32
// CHECK-NEXT:      %3 = arith.addf %1, %2 : f32
// CHECK-NEXT:      %4 = math.rsqrt %3 : f32
// CHECK-NEXT:      affine.for %arg4 = 0 to 2048 {
// CHECK-NEXT:        %5 = affine.load %memspacecast[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:        %6 = affine.load %memspacecast_4[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:        %7 = arith.addf %5, %6 : f32
// CHECK-NEXT:        %8 = arith.mulf %7, %4 : f32
// CHECK-NEXT:        %9 = affine.load %arg2[%arg4] : memref<2048xf32, {SymShapeAttr = ["s2"]}>
// CHECK-NEXT:        %10 = arith.mulf %9, %8 : f32
// CHECK-NEXT:        affine.store %10, %alloc_6[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return %memspacecast_7, %memspacecast_9 : memref<256x10x1xf32, {SymShapeAttr = ["s0", "s1", "1"]}>, memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>
// CHECK-NEXT:  }

func.func @akg_fused_add_mean_mul_pow_rsqrt_10_auto_fallback(%arg0: memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>, %arg1: memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>, %arg2: memref<2048xf32, {SymShapeAttr = ["s2"]}>) -> (memref<256x10x1xf32, {SymShapeAttr = ["s0", "s1", "1"]}>, memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>) attributes {OperatorType = "Reduction", hacc.function_kind = #hacc.function_kind<HOST>} {
  %cst = arith.constant 2.048000e+03 : f32
  %cst_0 = arith.constant 2.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e-05 : f64
  %cst_2 = arith.constant 0.000000e+00 : f32
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}> into memref<2560x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>
  %memspacecast = memref.memory_space_cast %collapse_shape : memref<2560x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}> to memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
  %collapse_shape_3 = memref.collapse_shape %arg1 [[0, 1], [2]] : memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}> into memref<2560x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>
  %memspacecast_4 = memref.memory_space_cast %collapse_shape_3 : memref<2560x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}> to memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<2560xf32, {SymShapeAttr = ["s33"]}>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
  %expand_shape = memref.expand_shape %alloc_5 [[0, 1, 2]] output_shape [256, 10, 1] : memref<2560xf32, {SymShapeAttr = ["s33"]}> into memref<256x10x1xf32, {SymShapeAttr = ["s33"]}>
  %memspacecast_7 = memref.memory_space_cast %expand_shape : memref<256x10x1xf32, {SymShapeAttr = ["s33"]}> to memref<256x10x1xf32, {SymShapeAttr = ["s0", "s1", "1"]}>
  %expand_shape_8 = memref.expand_shape %alloc_6 [[0, 1], [2]] output_shape [256, 10, 2048] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}> into memref<256x10x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
  %memspacecast_9 = memref.memory_space_cast %expand_shape_8 : memref<256x10x2048xf32, {SymShapeAttr = ["s33", "s2"]}> to memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>
  affine.for %arg3 = 0 to 2560 {
    affine.store %cst_2, %alloc_5[%arg3] {reduction_init} : memref<2560xf32, {SymShapeAttr = ["s33"]}>
    affine.for %arg4 = 0 to 2048 {
      %5 = affine.load %memspacecast[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
      %6 = affine.load %memspacecast_4[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
      %7 = arith.addf %5, %6 : f32
      affine.store %7, %alloc[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
      %8 = affine.load %alloc[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
      %9 = math.powf %8, %cst_0 : f32
      %10 = affine.load %alloc_5[%arg3] : memref<2560xf32, {SymShapeAttr = ["s33"]}>
      %11 = arith.addf %9, %10 {reduction_axes = [1 : index], reduction_type = "x"} : f32
      affine.store %11, %alloc_5[%arg3] : memref<2560xf32, {SymShapeAttr = ["s33"]}>
    } {reduction}
    %0 = affine.load %alloc_5[%arg3] : memref<2560xf32, {SymShapeAttr = ["s33"]}>
    %1 = arith.divf %0, %cst : f32
    %2 = arith.truncf %cst_1 : f64 to f32
    %3 = arith.addf %1, %2 : f32
    %4 = math.rsqrt %3 : f32
    affine.for %arg4 = 0 to 2048 {
      %5 = affine.load %alloc[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
      %6 = arith.mulf %5, %4 : f32
      %7 = affine.load %arg2[%arg4] : memref<2048xf32, {SymShapeAttr = ["s2"]}>
      %8 = arith.mulf %7, %6 : f32
      affine.store %8, %alloc_6[%arg3, %arg4] : memref<2560x2048xf32, {SymShapeAttr = ["s33", "s2"]}>
    }
  }
  return %memspacecast_7, %memspacecast_9 : memref<256x10x1xf32, {SymShapeAttr = ["s0", "s1", "1"]}>, memref<256x10x2048xf32, {SymShapeAttr = ["s0", "s1", "s2"]}>
}
