// RUN: akg-opt %s --extract-if | FileCheck %s

// CHECK-LABEL: func.func @Fused_ReduceSum_Mul_split_8579245076110957122(%arg0: memref<8x128x768xf32>, %arg1: memref<8x128xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
// CHECK-NEXT:   %cst = arith.constant 0.00130208337 : f32
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x128xf32>
// CHECK-NEXT:   %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x128xf32>
// CHECK-NEXT:   affine.for %arg2 = 0 to 8 {
// CHECK-NEXT:     affine.for %arg3 = 0 to 128 {
// CHECK-NEXT:       affine.for %arg4 = 0 to 768 {
// CHECK-NEXT:         affine.for %arg5 = 0 to 768 {
// CHECK-NEXT:           %2 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<8x128x768xf32>
// CHECK-NEXT:           %3 = affine.load %alloc[%arg2, %arg3] : memref<8x128xf32>
// CHECK-NEXT:           %4 = arith.addf %2, %3 {reduction_axes = [2 : index], reduction_type = "x"} : f32
// CHECK-NEXT:           affine.store %4, %alloc[%arg2, %arg3] : memref<8x128xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       %0 = affine.load %alloc[%arg2, %arg3] : memref<8x128xf32>
// CHECK-NEXT:       %1 = arith.mulf %0, %cst : f32
// CHECK-NEXT:       affine.store %1, %alloc_0[%arg2, %arg3] : memref<8x128xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   memref.copy %alloc_0, %arg1 : memref<8x128xf32> to memref<8x128xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }


#set = affine_set<(d0) : (-d0 + 767 == 0)>
func.func @Fused_ReduceSum_Mul_split_8579245076110957122(%arg0: memref<8x128x768xf32>, %arg1: memref<8x128xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
  %cst = arith.constant 0.00130208337 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x128xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<8x128xf32>
  affine.for %arg2 = 0 to 8 {
    affine.for %arg3 = 0 to 128 {
      affine.for %arg4 = 0 to 768 {
        affine.for %arg5 = 0 to 768 {
          %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<8x128x768xf32>
          %1 = affine.load %alloc[%arg2, %arg3] : memref<8x128xf32>
          %2 = arith.addf %0, %1 {reduction_axes = [2 : index], reduction_type = "x"} : f32
          affine.store %2, %alloc[%arg2, %arg3] : memref<8x128xf32>
          affine.if #set(%arg5) {
            affine.if #set(%arg4) {
              %3 = affine.load %alloc[%arg2, %arg3] : memref<8x128xf32>
              %4 = arith.mulf %3, %cst : f32
              affine.store %4, %alloc_0[%arg2, %arg3] : memref<8x128xf32>
            }
          }
        }
      }
    }
  }
  memref.copy %alloc_0, %arg1 : memref<8x128xf32> to memref<8x128xf32>
  return
}
