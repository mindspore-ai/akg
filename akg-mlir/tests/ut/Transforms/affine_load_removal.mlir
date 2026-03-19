// RUN: akg-opt %s --affine-load-removal | FileCheck %s

// CHECK-LABEL:  func.func @reduction_x(%arg0: memref<256x512x1024xf32>, %arg1: memref<256x512xf32>) attributes {OperatorType = "Reduction"} {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     affine.for %arg2 = 0 to 256 {
// CHECK-NEXT:       affine.for %arg3 = 0 to 512 {
// CHECK-NEXT:         %0 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %cst) -> (f32) {
// CHECK-NEXT:           %1 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<256x512x1024xf32>
// CHECK-NEXT:           %2 = arith.addf %arg5, %1 {reduction_axes = [2 : index], reduction_type = "x"} : f32
// CHECK-NEXT:           affine.yield %2 : f32
// CHECK-NEXT:         }
// CHECK-NEXT:         affine.store %0, %arg1[%arg2, %arg3] : memref<256x512xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


func.func @reduction_x(%in: memref<256x512x1024xf32>, %out: memref<256x512xf32>) attributes {OperatorType = "Reduction"} {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   affine.for %j = 0 to 512 {
     affine.store %cst, %out[%i, %j] {reduction_init} : memref<256x512xf32>
     affine.for %k = 0 to 1024 {
       %ld = affine.load %in[%i, %j, %k] : memref<256x512x1024xf32>
       %sum = affine.load %out[%i, %j] : memref<256x512xf32>
       %add = arith.addf %sum, %ld {reduction_axes = [2:index], reduction_type = "x"}: f32
       affine.store %add, %out[%i, %j] : memref<256x512xf32>
     } {reduction}
   }
 }
 return
}


// CHECK-LABEL:  func.func @reduction_y(%arg0: memref<5128x12288xf32, {SymShapeAttr = ["s0", "s1"]}>) -> memref<12288xf32, {SymShapeAttr = ["s1"]}> attributes {OperatorType = "Reduction"} {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<12288xf32, {SymShapeAttr = ["s1"]}>
// CHECK-NEXT:     affine.for %arg1 = 0 to 12288 {
// CHECK-NEXT:       %0 = affine.for %arg2 = 0 to 5128 iter_args(%arg3 = %cst) -> (f32) {
// CHECK-NEXT:         %1 = affine.load %arg0[%arg2, %arg1] : memref<5128x12288xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:         %2 = arith.addf %1, %arg3 {reduction_axes = [0 : index], reduction_type = "y"} : f32
// CHECK-NEXT:         affine.yield %2 : f32
// CHECK-NEXT:       } {reduction}
// CHECK-NEXT:       affine.store %0, %alloc[%arg1] : memref<12288xf32, {SymShapeAttr = ["s1"]}>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %alloc : memref<12288xf32, {SymShapeAttr = ["s1"]}>
// CHECK-NEXT:   }


func.func @reduction_y(%arg0: memref<5128x12288xf32, {SymShapeAttr = ["s0", "s1"]}>) -> memref<12288xf32, {SymShapeAttr = ["s1"]}> attributes {OperatorType = "Reduction"} {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<12288xf32, {SymShapeAttr = ["s1"]}>
  affine.for %arg1 = 0 to 12288 {
    affine.store %cst, %alloc[%arg1] {reduction_init} : memref<12288xf32, {SymShapeAttr = ["s1"]}>
  }
  affine.for %arg1 = 0 to 5128 {
    affine.for %arg2 = 0 to 12288 {
      %0 = affine.load %arg0[%arg1, %arg2] : memref<5128x12288xf32, {SymShapeAttr = ["s0", "s1"]}>
      %1 = affine.load %alloc[%arg2] : memref<12288xf32, {SymShapeAttr = ["s1"]}>
      %2 = arith.addf %0, %1 {reduction_axes = [0 : index], reduction_type = "y"} : f32
      affine.store %2, %alloc[%arg2] : memref<12288xf32, {SymShapeAttr = ["s1"]}>
    }
  } {reduction}
  return %alloc : memref<12288xf32, {SymShapeAttr = ["s1"]}>
}


// CHECK-LABEL:  func.func @multi_reduce_x(%arg0: memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>, %arg1: memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>) -> (memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>) attributes {OperatorType = "Reduction"} {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_1 = arith.constant 9.9999999999999995E-7 : f64
// CHECK-NEXT:     %cst_2 = arith.constant 6.400000e+01 : f32
// CHECK-NEXT:     %cst_3 = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
// CHECK-NEXT:     %subview = memref.subview %alloc_6[0, 0] [14646, 64] [1, 1] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}> to memref<14646x64xf32, strided<[256, 1]>, {SymShapeAttr = ["s0", "s3"]}>
// CHECK-NEXT:     %memspacecast = memref.memory_space_cast %subview : memref<14646x64xf32, strided<[256, 1]>, {SymShapeAttr = ["s0", "s3"]}> to memref<14646x64xf32, strided<[256, 1]>, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:     %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:     %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:     %expand_shape = memref.expand_shape %alloc_4 {{\[\[0, 1\]\]}} output_shape [14646, 1] : memref<14646xf32, {SymShapeAttr = ["s0"]}> into memref<14646x1xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %memspacecast_9 = memref.memory_space_cast %expand_shape : memref<14646x1xf32, {SymShapeAttr = ["s0"]}> to memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>
// CHECK-NEXT:     %expand_shape_10 = memref.expand_shape %alloc_5 {{\[\[0, 1\]\]}} output_shape [14646, 1] : memref<14646xf32, {SymShapeAttr = ["s0"]}> into memref<14646x1xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %memspacecast_11 = memref.memory_space_cast %expand_shape_10 : memref<14646x1xf32, {SymShapeAttr = ["s0"]}> to memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>
// CHECK-NEXT:     %expand_shape_12 = memref.expand_shape %alloc {{\[\[0, 1\]\]}} output_shape [14646, 1] : memref<14646xf32, {SymShapeAttr = ["s0"]}> into memref<14646x1xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:     %memspacecast_13 = memref.memory_space_cast %expand_shape_12 : memref<14646x1xf32, {SymShapeAttr = ["s0"]}> to memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>
// CHECK-NEXT:     affine.for %arg2 = 0 to 14646 {
// CHECK-NEXT:       %0 = affine.load %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:       %1 = affine.for %arg3 = 0 to 64 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:         %10 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:         %11 = arith.addf %10, %arg4 {reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK-NEXT:         affine.yield %11 : f32
// CHECK-NEXT:       } {reduction}
// CHECK-NEXT:       affine.store %1, %alloc_4[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:       %2 = affine.load %alloc_4[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:       %3 = arith.divf %2, %cst_2 : f32
// CHECK-NEXT:       affine.store %3, %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:       %4 = affine.for %arg3 = 0 to 64 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:         %10 = affine.load %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:         %11 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:         %12 = arith.subf %11, %10 : f32
// CHECK-NEXT:         %13 = math.powf %12, %cst_3 : f32
// CHECK-NEXT:         %14 = arith.addf %13, %arg4 {reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK-NEXT:         affine.yield %14 : f32
// CHECK-NEXT:       } {reduction}
// CHECK-NEXT:       affine.store %4, %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:       %5 = affine.load %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:       %6 = arith.divf %5, %cst_2 : f32
// CHECK-NEXT:       %7 = arith.truncf %cst_1 : f64 to f32
// CHECK-NEXT:       %8 = arith.addf %6, %7 : f32
// CHECK-NEXT:       %9 = math.rsqrt %8 : f32
// CHECK-NEXT:       affine.for %arg3 = 0 to 64 {
// CHECK-NEXT:         %10 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:         %11 = affine.load %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
// CHECK-NEXT:         %12 = arith.subf %10, %11 : f32
// CHECK-NEXT:         %13 = arith.mulf %12, %9 : f32
// CHECK-NEXT:         affine.store %13, %alloc_7[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.for %arg3 = 0 to 256 {
// CHECK-NEXT:         %10 = affine.load %arg1[%arg2, %arg3] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
// CHECK-NEXT:         %11 = arith.negf %10 : f32
// CHECK-NEXT:         %12 = math.exp %11 : f32
// CHECK-NEXT:         %13 = arith.addf %12, %cst_0 : f32
// CHECK-NEXT:         %14 = arith.divf %cst_0, %13 : f32
// CHECK-NEXT:         %15 = affine.load %arg1[%arg2, %arg3] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
// CHECK-NEXT:         %16 = arith.mulf %15, %14 : f32
// CHECK-NEXT:         affine.store %16, %alloc_6[%arg2, %arg3] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.for %arg3 = 0 to 64 {
// CHECK-NEXT:         %10 = affine.load %memspacecast[%arg2, %arg3] : memref<14646x64xf32, strided<[256, 1]>, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:         %11 = affine.load %alloc_7[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:         %12 = arith.mulf %10, %11 : f32
// CHECK-NEXT:         affine.store %12, %alloc_8[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return %memspacecast_9, %memspacecast_11, %memspacecast_13, %alloc_8 : memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK-NEXT:   }



func.func @multi_reduce_x(%arg0: memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>, %arg1: memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>) -> (memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>) attributes {OperatorType = "Reduction"} {
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
    affine.store %cst, %alloc[%arg2] {reduction_init} : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    %0 = affine.load %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    affine.store %0, %alloc_4[%arg2] {reduction_init} : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    affine.for %arg3 = 0 to 64 {
      %8 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
      %9 = affine.load %alloc_4[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
      %10 = arith.addf %8, %9 {reduction_axes = [1 : index], reduction_type = "x"} : f32
      affine.store %10, %alloc_4[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    } {reduction}
    %1 = affine.load %alloc_4[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    %2 = arith.divf %1, %cst_2 : f32
    affine.store %2, %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    affine.for %arg3 = 0 to 64 {
      %8 = affine.load %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
      %9 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
      %10 = arith.subf %9, %8 : f32
      %11 = math.powf %10, %cst_3 : f32
      %12 = affine.load %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
      %13 = arith.addf %11, %12 {reduction_axes = [1 : index], reduction_type = "x"} : f32
      affine.store %13, %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    } {reduction}
    %3 = affine.load %alloc[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
    %4 = arith.divf %3, %cst_2 : f32
    %5 = arith.truncf %cst_1 : f64 to f32
    %6 = arith.addf %4, %5 : f32
    %7 = math.rsqrt %6 : f32
    affine.for %arg3 = 0 to 64 {
      %8 = affine.load %arg0[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
      %9 = affine.load %alloc_5[%arg2] : memref<14646xf32, {SymShapeAttr = ["s0"]}>
      %10 = arith.subf %8, %9 : f32
      %11 = arith.mulf %10, %7 : f32
      affine.store %11, %alloc_7[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
    }
    affine.for %arg3 = 0 to 256 {
      %8 = affine.load %arg1[%arg2, %arg3] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
      %9 = arith.negf %8 : f32
      %10 = math.exp %9 : f32
      %11 = arith.addf %10, %cst_0 : f32
      %12 = arith.divf %cst_0, %11 : f32
      %13 = affine.load %arg1[%arg2, %arg3] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
      %14 = arith.mulf %13, %12 : f32
      affine.store %14, %alloc_6[%arg2, %arg3] : memref<14646x256xf32, {SymShapeAttr = ["s0", "s3"]}>
    }
    affine.for %arg3 = 0 to 64 {
      %8 = affine.load %memspacecast[%arg2, %arg3] : memref<14646x64xf32, strided<[256, 1]>, {SymShapeAttr = ["s0", "s1"]}>
      %9 = affine.load %alloc_7[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
      %10 = arith.mulf %8, %9 : f32
      affine.store %10, %alloc_8[%arg2, %arg3] : memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
    }
  }
  return %memspacecast_9, %memspacecast_11, %memspacecast_13, %alloc_8 : memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x1xf32, {SymShapeAttr = ["s0", "1"]}>, memref<14646x64xf32, {SymShapeAttr = ["s0", "s1"]}>
}
