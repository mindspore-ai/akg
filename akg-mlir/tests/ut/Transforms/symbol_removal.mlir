// RUN: akg-opt %s -symbol-removal | FileCheck %s

// CHECK-LABEL:  func.func @Symbol_Removal(%arg0: memref<4x4x196x196xf32>, %arg1: memref<4x4x196xf32>) {
// CHECK-NEXT:     %cst = arith.constant -3.40282347E+38 : f32
// CHECK-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4x196xf32>
// CHECK-NEXT:     affine.for %arg2 = 0 to 4 {
// CHECK-NEXT:       affine.for %arg3 = 0 to 4 {
// CHECK-NEXT:         affine.for %arg4 = 0 to 196 {
// CHECK-NEXT:           affine.store %cst, %alloc[%arg2, %arg3, %arg4] : memref<4x4x196xf32>
// CHECK-NEXT:           affine.for %arg5 = 0 to 196 {
// CHECK-NEXT:             %0 = affine.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<4x4x196x196xf32>
// CHECK-NEXT:             %1 = affine.load %alloc[%arg2, %arg3, %arg4] : memref<4x4x196xf32>
// CHECK-NEXT:             %2 = arith.maxf %0, %1 : f32
// CHECK-NEXT:             affine.store %2, %alloc[%arg2, %arg3, %arg4] : memref<4x4x196xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     memref.copy %alloc, %arg1 : memref<4x4x196xf32> to memref<4x4x196xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }


func.func @Symbol_Removal(%arg0: memref<4x4x196x196xf32>, %arg1: memref<4x4x196xf32>) {
  %cst = arith.constant -3.40282347E+38 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x4x196xf32>
  affine.for %arg2 = 0 to 4 {
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 196 {
        affine.store %cst, %alloc[symbol(%arg2), symbol(%arg3), symbol(%arg4)] : memref<4x4x196xf32>
        affine.for %arg5 = 0 to 196 {
          %0 = affine.load %arg0[symbol(%arg2), symbol(%arg3), symbol(%arg4), symbol(%arg5)] : memref<4x4x196x196xf32>
          %1 = affine.load %alloc[symbol(%arg2), symbol(%arg3), symbol(%arg4)] : memref<4x4x196xf32>
          %2 = arith.maxf %0, %1 : f32
          affine.store %2, %alloc[symbol(%arg2), symbol(%arg3), symbol(%arg4)] : memref<4x4x196xf32>
        }
      }
    }
  }
  memref.copy %alloc, %arg1 : memref<4x4x196xf32> to memref<4x4x196xf32>
  return
}