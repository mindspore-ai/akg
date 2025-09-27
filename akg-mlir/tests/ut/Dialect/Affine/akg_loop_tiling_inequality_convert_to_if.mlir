// RUN: akg-opt %s -split-input-file --akg-affine-loop-tile="tile-sizes=32,30,30 inequality-convert-to-if=true" | FileCheck %s


// CHECK-LABEL:  #map = affine_map<(d0) -> (d0)>
// CHECK-NEXT:   #map1 = affine_map<(d0) -> (d0 + 32)>
// CHECK-NEXT:   #map2 = affine_map<(d0) -> (d0 + 30)>
// CHECK-NEXT:   #set = affine_set<(d0, d1) : (-d0 + 1279 >= 0, -d1 + 767 >= 0)>
// CHECK-NEXT:   module {
// CHECK-NEXT:     func.func @elementwise(%arg0: memref<32x1280x768xf32>, %arg1: memref<32x1280x768xf32>, %arg2: memref<32x1280x768xf32>) attributes {OperatorType = "Elementwise", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
// CHECK-NEXT:       affine.for %arg3 = 0 to 32 step 32 {
// CHECK-NEXT:         affine.for %arg4 = 0 to 1280 step 30 {
// CHECK-NEXT:           affine.for %arg5 = 0 to 768 step 30 {
// CHECK-NEXT:             affine.for %arg6 = #map(%arg3) to #map1(%arg3) {
// CHECK-NEXT:               affine.for %arg7 = #map(%arg4) to #map2(%arg4) {
// CHECK-NEXT:                 affine.for %arg8 = #map(%arg5) to #map2(%arg5) {
// CHECK-NEXT:                   affine.if #set(%arg7, %arg8) {
// CHECK-NEXT:                     %0 = affine.load %arg0[%arg6, %arg7, %arg8] : memref<32x1280x768xf32>
// CHECK-NEXT:                     %1 = affine.load %arg1[%arg6, %arg7, %arg8] : memref<32x1280x768xf32>
// CHECK-NEXT:                     %2 = arith.mulf %0, %1 : f32
// CHECK-NEXT:                     affine.store %2, %arg2[%arg6, %arg7, %arg8] : memref<32x1280x768xf32>
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

 
module {
  func.func @elementwise(%arg0: memref<32x1280x768xf32>, %arg1: memref<32x1280x768xf32>, %arg2: memref<32x1280x768xf32>) attributes {OperatorType = "Elementwise", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
    affine.for %arg3 = 0 to 32 {
      affine.for %arg4 = 0 to 1280 {
        affine.for %arg5 = 0 to 768 {
          %0 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<32x1280x768xf32>
          %1 = affine.load %arg1[%arg3, %arg4, %arg5] : memref<32x1280x768xf32>
          %2 = arith.mulf %0, %1 : f32
          affine.store %2, %arg2[%arg3, %arg4, %arg5] : memref<32x1280x768xf32>
        }
      }
    }
    return
  }
}

