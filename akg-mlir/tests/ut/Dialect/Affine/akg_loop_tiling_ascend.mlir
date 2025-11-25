// RUN: akg-opt %s --akg-affine-loop-tile="use-auto-tiling=true" -allow-unregistered-dialect | FileCheck %s 


// CHECK-LABEL:  #map = affine_map<(d0) -> (d0)>
// CHECK-NEXT:   #map1 = affine_map<(d0) -> (d0 + 4096)>
// CHECK-NEXT:   #map2 = affine_map<(d0) -> (d0 + 512)>
// CHECK-NEXT:   #map3 = affine_map<(d0) -> (d0 + 416)>
// CHECK-NEXT:   #map4 = affine_map<(d0) -> (d0 + 1536)>
// CHECK-NEXT:   #map5 = affine_map<(d0) -> (d0 + 1696)>
// CHECK-NEXT:   #map6 = affine_map<(d0) -> (d0 + 160)>
// CHECK-NEXT:   module {
// CHECK-NEXT:     func.func @tiling_only_innermost(%arg0: memref<100000x4000xf32>, %arg1: memref<100000x4000xf32>) attributes {OperatorType = "Elementwise", process = "aicore"} {
// CHECK-NEXT:       affine.for %arg2 = 0 to 98304 step 4096 {
// CHECK-NEXT:         affine.for %arg3 = 0 to 3584 step 512 {
// CHECK-NEXT:           affine.for %arg4 = #map(%arg2) to #map1(%arg2) step 512 {
// CHECK-NEXT:             affine.for %arg5 = #map(%arg3) to #map2(%arg3) step 512 {
// CHECK-NEXT:              affine.for %arg6 = #map(%arg4) to #map2(%arg4) {
// CHECK-NEXT:                 affine.for %arg7 = #map(%arg5) to #map2(%arg5) {
// CHECK-NEXT:                   %0 = affine.load %arg0[%arg6, %arg7] : memref<100000x4000xf32>
// CHECK-NEXT:                   affine.store %0, %arg1[%arg6, %arg7] : memref<100000x4000xf32>
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:          }
// CHECK-NEXT:         }
// CHECK-NEXT:         affine.for %arg3 = 3584 to 4000 step 416 {
// CHECK-NEXT:           affine.for %arg4 = #map(%arg2) to #map1(%arg2) step 512 {
// CHECK-NEXT:             affine.for %arg5 = #map(%arg3) to #map3(%arg3) step 416 {
// CHECK-NEXT:               affine.for %arg6 = #map(%arg4) to #map2(%arg4) {
// CHECK-NEXT:                 affine.for %arg7 = #map(%arg5) to #map3(%arg5) {
// CHECK-NEXT:                   %0 = affine.load %arg0[%arg6, %arg7] : memref<100000x4000xf32>
// CHECK-NEXT:                   affine.store %0, %arg1[%arg6, %arg7] : memref<100000x4000xf32>
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.for %arg2 = 98304 to 100000 step 1696 {
// CHECK-NEXT:         affine.for %arg3 = 0 to 3584 step 512 {
// CHECK-NEXT:           affine.for %arg4 = #map(%arg2) to #map4(%arg2) step 512 {
// CHECK-NEXT:             affine.for %arg5 = #map(%arg3) to #map2(%arg3) step 512 {
// CHECK-NEXT:               affine.for %arg6 = #map(%arg4) to #map2(%arg4) {
// CHECK-NEXT:                 affine.for %arg7 = #map(%arg5) to #map2(%arg5) {
// CHECK-NEXT:                   %0 = affine.load %arg0[%arg6, %arg7] : memref<100000x4000xf32>
// CHECK-NEXT:                   affine.store %0, %arg1[%arg6, %arg7] : memref<100000x4000xf32>
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           affine.for %arg4 = #map4(%arg2) to #map5(%arg2) step 160 {
// CHECK-NEXT:             affine.for %arg5 = #map(%arg3) to #map2(%arg3) step 512 {
// CHECK-NEXT:               affine.for %arg6 = #map(%arg4) to #map6(%arg4) {
// CHECK-NEXT:                 affine.for %arg7 = #map(%arg5) to #map2(%arg5) {
// CHECK-NEXT:                   %0 = affine.load %arg0[%arg6, %arg7] : memref<100000x4000xf32>
// CHECK-NEXT:                   affine.store %0, %arg1[%arg6, %arg7] : memref<100000x4000xf32>
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:         affine.for %arg3 = 3584 to 4000 step 416 {
// CHECK-NEXT:           affine.for %arg4 = #map(%arg2) to #map4(%arg2) step 512 {
// CHECK-NEXT:             affine.for %arg5 = #map(%arg3) to #map3(%arg3) step 416 {
// CHECK-NEXT:               affine.for %arg6 = #map(%arg4) to #map2(%arg4) {
// CHECK-NEXT:                 affine.for %arg7 = #map(%arg5) to #map3(%arg5) {
// CHECK-NEXT:                   %0 = affine.load %arg0[%arg6, %arg7] : memref<100000x4000xf32>
// CHECK-NEXT:                   affine.store %0, %arg1[%arg6, %arg7] : memref<100000x4000xf32>
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           affine.for %arg4 = #map4(%arg2) to #map5(%arg2) step 160 {
// CHECK-NEXT:             affine.for %arg5 = #map(%arg3) to #map3(%arg3) step 416 {
// CHECK-NEXT:               affine.for %arg6 = #map(%arg4) to #map6(%arg4) {
// CHECK-NEXT:                 affine.for %arg7 = #map(%arg5) to #map3(%arg5) {
// CHECK-NEXT:                   %0 = affine.load %arg0[%arg6, %arg7] : memref<100000x4000xf32>
// CHECK-NEXT:                   affine.store %0, %arg1[%arg6, %arg7] : memref<100000x4000xf32>
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
  func.func @tiling_only_innermost(%A: memref<100000x4000xf32>, %B: memref<100000x4000xf32>) attributes {process = "aicore", OperatorType = "Elementwise"} {
    affine.for %i = 0 to 100000 {
      affine.for %j = 0 to 4000 {
        %val = affine.load %A[%i, %j] : memref<100000x4000xf32>
        affine.store %val, %B[%i, %j] : memref<100000x4000xf32>
      }
    }
    return
  }
}