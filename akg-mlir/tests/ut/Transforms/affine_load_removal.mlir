// RUN: akg-opt %s --affine-load-removal | FileCheck %s

// CHECK-LABEL:  func.func @reduction(%arg0: memref<256x512x1024xf32>, %arg1: memref<256x512xf32>) attributes {OperatorType = "Reduce"} {
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


func.func @reduction(%in: memref<256x512x1024xf32>, %out: memref<256x512xf32>) attributes {OperatorType = "Reduce"} {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   affine.for %j = 0 to 512 {
     affine.store %cst, %out[%i, %j] : memref<256x512xf32>
     affine.for %k = 0 to 1024 {
       %ld = affine.load %in[%i, %j, %k] : memref<256x512x1024xf32>
       %sum = affine.load %out[%i, %j] : memref<256x512xf32>
       %add = arith.addf %sum, %ld {reduction_axes = [2:index], reduction_type = "x"}: f32
       affine.store %add, %out[%i, %j] : memref<256x512xf32>
     } {reduction_loop}
   }
 }
 return
}
