// RUN: akg-opt %s --akg-affine-loop-tile="use-auto-tiling=true" -allow-unregistered-dialect | FileCheck %s 

// CHECK-LABEL:  #map = affine_map<(d0) -> (d0)>
// CHECK-NEXT:   #map1 = affine_map<(d0) -> (d0 + 32)>
// CHECK-NEXT:   module {
// CHECK-NEXT:     func.func @tiling_only_innermost(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>) attributes {OperatorType = "Elementwise", process = "aicore"} {
// CHECK-NEXT:       affine.for %arg2 = 0 to 256 step 32 {
// CHECK-NEXT:         affine.for %arg3 = 0 to 4000 step 32 {
// CHECK-NEXT:           affine.for %arg4 = #map(%arg2) to #map1(%arg2) {
// CHECK-NEXT:             affine.for %arg5 = #map(%arg3) to #map1(%arg3) {
// CHECK-NEXT:               %0 = affine.load %arg0[%arg4, %arg5] : memref<256x256xf32>
// CHECK-NEXT:               affine.store %0, %arg1[%arg4, %arg5] : memref<256x256xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

module {
  func.func @tiling_only_innermost(%A: memref<256x256xf32>, %B: memref<256x256xf32>) attributes {process = "aicore", OperatorType = "Elementwise"} {
    affine.for %i = 0 to 256 {
      affine.for %j = 0 to 4000 {
        %val = affine.load %A[%i, %j] : memref<256x256xf32>
        affine.store %val, %B[%i, %j] : memref<256x256xf32>
      }
    }
    return
  }
}