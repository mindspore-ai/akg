// RUN: akg-opt %s -split-input-file --affine-loop-reorder="new-order=0,3,1,2" | FileCheck %s

// CHECK-LABEL:  #set = affine_set<(d0) : (d0 == 0)>
// CHECK-NEXT:   module {
// CHECK-NEXT:     func.func @Fused_Mul_ReduceSum_split_6886769834019553517(%arg0: memref<128x32x112x112xf32>, %arg1: memref<128x32x112x112xf32>, %arg2: memref<128x32xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
// CHECK-NEXT:       %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       affine.for %arg3 = 0 to 128 {
// CHECK-NEXT:         affine.for %arg4 = 0 to 112 {
// CHECK-NEXT:           affine.for %arg5 = 0 to 32 {
// CHECK-NEXT:             affine.for %arg6 = 0 to 112 {
// CHECK-NEXT:               affine.if #set(%arg4) {
// CHECK-NEXT:                 affine.if #set(%arg6) {
// CHECK-NEXT:                   affine.store %cst, %arg2[%arg3, %arg5] : memref<128x32xf32>
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:               %0 = affine.load %arg0[%arg3, %arg5, %arg6, %arg4] : memref<128x32x112x112xf32>
// CHECK-NEXT:               %1 = affine.load %arg1[%arg3, %arg5, %arg6, %arg4] : memref<128x32x112x112xf32>
// CHECK-NEXT:               %2 = arith.mulf %0, %1 : f32
// CHECK-NEXT:               %3 = affine.load %arg2[%arg3, %arg5] : memref<128x32xf32>
// CHECK-NEXT:               %4 = arith.addf %2, %3 {reduction_axes = [2 : index, 3 : index], reduction_type = "x"} : f32
// CHECK-NEXT:               affine.store %4, %arg2[%arg3, %arg5] : memref<128x32xf32>
// CHECK-NEXT:             } {tag2}
// CHECK-NEXT:           } {tag1}
// CHECK-NEXT:         } {tag3}
// CHECK-NEXT:       } {tag0}
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

#set = affine_set<(d0) : (d0 == 0)>
module {
  func.func @Fused_Mul_ReduceSum_split_6886769834019553517(%arg0: memref<128x32x112x112xf32>, %arg1: memref<128x32x112x112xf32>, %arg2: memref<128x32xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 128 {
      affine.for %arg4 = 0 to 32 {
        affine.for %arg5 = 0 to 112 {
          affine.for %arg6 = 0 to 112 {
            affine.if #set(%arg6) {
              affine.if #set(%arg5) {
                affine.store %cst, %arg2[%arg3, %arg4] : memref<128x32xf32>
              }
            }
            %0 = affine.load %arg0[%arg3, %arg4, %arg5, %arg6] : memref<128x32x112x112xf32>
            %1 = affine.load %arg1[%arg3, %arg4, %arg5, %arg6] : memref<128x32x112x112xf32>
            %2 = arith.mulf %0, %1 : f32
            %3 = affine.load %arg2[%arg3, %arg4] : memref<128x32xf32>
            %4 = arith.addf %2, %3 {reduction_axes = [2 : index, 3 : index], reduction_type = "x"} : f32
            affine.store %4, %arg2[%arg3, %arg4] : memref<128x32xf32>
          } {tag3}
        } {tag2}
      } {tag1}
    } {tag0}
    return
  }
}
