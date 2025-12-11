// RUN: akg-opt %s --akg-affine-loop-tile="use-auto-tiling=true" -allow-unregistered-dialect | FileCheck %s 

// CHECK-LABEL:  ModelGraph Template : REDUCTION
// CHECK-NEXT:   #map = affine_map<(d0) -> (d0 * 513)>
// CHECK-NEXT:   #map1 = affine_map<(d0) -> (d0 * 513 + 513)>
// CHECK-NEXT:   #map2 = affine_map<(d0) -> (d0)>
// CHECK-NEXT:   #map3 = affine_map<(d0) -> (d0 + 513)>
// CHECK-NEXT:   #map4 = affine_map<(d0) -> (d0 + 512)>
// CHECK-NEXT:   #map5 = affine_map<(d0) -> (d0 + 45)>
// CHECK-NEXT:   #map6 = affine_map<(d0) -> (d0 + 180)>
// CHECK-NEXT:   #map7 = affine_map<(d0) -> (d0 + 493)>
// CHECK-NEXT:   #set = affine_set<(d0) : (-d0 + 38 >= 0)>
// CHECK-NEXT:   #set1 = affine_set<(d0) : (d0 - 39 == 0)>
// CHECK-NEXT:   module {
// CHECK-NEXT:     func.func @reduce_3d(%arg0: tensor<20500x5300x2605xf32>) -> tensor<20500xf32> attributes {OperatorType = "Reduce", process = "aicore"} {
// CHECK-NEXT:       %0 = bufferization.to_memref %arg0 : memref<20500x5300x2605xf32, strided<[?, ?, ?], offset: ?>>
// CHECK-NEXT:       %alloc = memref.alloc() {alignment = 64 : i64} : memref<20500xf32>
// CHECK-NEXT:       affine.for %arg1 = 0 to 40 {
// CHECK-NEXT:         affine.if #set(%arg1) {
// CHECK-NEXT:           affine.for %arg2 = #map(%arg1) to #map1(%arg1) step 513 {
// CHECK-NEXT:             affine.for %arg3 = 0 to 5120 step 512 {
// CHECK-NEXT:               affine.for %arg4 = 0 to 2560 step 512 {
// CHECK-NEXT:                 affine.for %arg5 = #map2(%arg2) to #map3(%arg2) step 513 {
// CHECK-NEXT:                   affine.for %arg6 = #map2(%arg3) to #map4(%arg3) step 512 {
// CHECK-NEXT:                     affine.for %arg7 = #map2(%arg4) to #map4(%arg4) step 512 {
// CHECK-NEXT:                       affine.for %arg8 = #map2(%arg5) to #map3(%arg5) {
// CHECK-NEXT:                         affine.for %arg9 = #map2(%arg6) to #map4(%arg6) {
// CHECK-NEXT:                           affine.for %arg10 = #map2(%arg7) to #map4(%arg7) {
// CHECK-NEXT:                             %2 = affine.load %0[%arg8, %arg9, %arg10] : memref<20500x5300x2605xf32, strided<[?, ?, ?], offset: ?>>
// CHECK-NEXT:                             %3 = affine.load %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                             %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:                             affine.store %4, %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                           } {vector}
// CHECK-NEXT:                         }
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:               affine.for %arg4 = 2560 to 2605 step 45 {
// CHECK-NEXT:                 affine.for %arg5 = #map2(%arg2) to #map3(%arg2) step 513 {
// CHECK-NEXT:                   affine.for %arg6 = #map2(%arg3) to #map4(%arg3) step 512 {
// CHECK-NEXT:                     affine.for %arg7 = #map2(%arg4) to #map5(%arg4) step 45 {
// CHECK-NEXT:                       affine.for %arg8 = #map2(%arg5) to #map3(%arg5) {
// CHECK-NEXT:                         affine.for %arg9 = #map2(%arg6) to #map4(%arg6) {
// CHECK-NEXT:                           affine.for %arg10 = #map2(%arg7) to #map5(%arg7) {
// CHECK-NEXT:                             %2 = affine.load %0[%arg8, %arg9, %arg10] : memref<20500x5300x2605xf32, strided<[?, ?, ?], offset: ?>>
// CHECK-NEXT:                             %3 = affine.load %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                             %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:                             affine.store %4, %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                           } {vector}
// CHECK-NEXT:                         }
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:             affine.for %arg3 = 5120 to 5300 step 180 {
// CHECK-NEXT:               affine.for %arg4 = 0 to 2560 step 512 {
// CHECK-NEXT:                 affine.for %arg5 = #map2(%arg2) to #map3(%arg2) step 513 {
// CHECK-NEXT:                   affine.for %arg6 = #map2(%arg3) to #map6(%arg3) step 180 {
// CHECK-NEXT:                     affine.for %arg7 = #map2(%arg4) to #map4(%arg4) step 512 {
// CHECK-NEXT:                       affine.for %arg8 = #map2(%arg5) to #map3(%arg5) {
// CHECK-NEXT:                         affine.for %arg9 = #map2(%arg6) to #map6(%arg6) {
// CHECK-NEXT:                           affine.for %arg10 = #map2(%arg7) to #map4(%arg7) {
// CHECK-NEXT:                             %2 = affine.load %0[%arg8, %arg9, %arg10] : memref<20500x5300x2605xf32, strided<[?, ?, ?], offset: ?>>
// CHECK-NEXT:                             %3 = affine.load %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                             %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:                             affine.store %4, %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                           } {vector}
// CHECK-NEXT:                         }
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:               affine.for %arg4 = 2560 to 2605 step 45 {
// CHECK-NEXT:                 affine.for %arg5 = #map2(%arg2) to #map3(%arg2) step 513 {
// CHECK-NEXT:                   affine.for %arg6 = #map2(%arg3) to #map6(%arg3) step 180 {
// CHECK-NEXT:                     affine.for %arg7 = #map2(%arg4) to #map5(%arg4) step 45 {
// CHECK-NEXT:                       affine.for %arg8 = #map2(%arg5) to #map3(%arg5) {
// CHECK-NEXT:                         affine.for %arg9 = #map2(%arg6) to #map6(%arg6) {
// CHECK-NEXT:                           affine.for %arg10 = #map2(%arg7) to #map5(%arg7) {
// CHECK-NEXT:                             %2 = affine.load %0[%arg8, %arg9, %arg10] : memref<20500x5300x2605xf32, strided<[?, ?, ?], offset: ?>>
// CHECK-NEXT:                             %3 = affine.load %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                             %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:                             affine.store %4, %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                           } {vector}
// CHECK-NEXT:                         }
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:         affine.if #set1(%arg1) {
// CHECK-NEXT:           affine.for %arg2 = 20007 to 20500 step 493 {
// CHECK-NEXT:             affine.for %arg3 = 0 to 5120 step 512 {
// CHECK-NEXT:               affine.for %arg4 = 0 to 2560 step 512 {
// CHECK-NEXT:                 affine.for %arg5 = #map2(%arg2) to #map7(%arg2) step 493 {
// CHECK-NEXT:                   affine.for %arg6 = #map2(%arg3) to #map4(%arg3) step 512 {
// CHECK-NEXT:                     affine.for %arg7 = #map2(%arg4) to #map4(%arg4) step 512 {
// CHECK-NEXT:                       affine.for %arg8 = #map2(%arg5) to #map7(%arg5) {
// CHECK-NEXT:                         affine.for %arg9 = #map2(%arg6) to #map4(%arg6) {
// CHECK-NEXT:                           affine.for %arg10 = #map2(%arg7) to #map4(%arg7) {
// CHECK-NEXT:                             %2 = affine.load %0[%arg8, %arg9, %arg10] : memref<20500x5300x2605xf32, strided<[?, ?, ?], offset: ?>>
// CHECK-NEXT:                             %3 = affine.load %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                             %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:                             affine.store %4, %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                           } {vector}
// CHECK-NEXT:                         }
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:               affine.for %arg4 = 2560 to 2605 step 45 {
// CHECK-NEXT:                 affine.for %arg5 = #map2(%arg2) to #map7(%arg2) step 493 {
// CHECK-NEXT:                   affine.for %arg6 = #map2(%arg3) to #map4(%arg3) step 512 {
// CHECK-NEXT:                     affine.for %arg7 = #map2(%arg4) to #map5(%arg4) step 45 {
// CHECK-NEXT:                       affine.for %arg8 = #map2(%arg5) to #map7(%arg5) {
// CHECK-NEXT:                         affine.for %arg9 = #map2(%arg6) to #map4(%arg6) {
// CHECK-NEXT:                           affine.for %arg10 = #map2(%arg7) to #map5(%arg7) {
// CHECK-NEXT:                             %2 = affine.load %0[%arg8, %arg9, %arg10] : memref<20500x5300x2605xf32, strided<[?, ?, ?], offset: ?>>
// CHECK-NEXT:                             %3 = affine.load %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                             %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:                             affine.store %4, %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                           } {vector}
// CHECK-NEXT:                         }
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:             affine.for %arg3 = 5120 to 5300 step 180 {
// CHECK-NEXT:               affine.for %arg4 = 0 to 2560 step 512 {
// CHECK-NEXT:                 affine.for %arg5 = #map2(%arg2) to #map7(%arg2) step 493 {
// CHECK-NEXT:                   affine.for %arg6 = #map2(%arg3) to #map6(%arg3) step 180 {
// CHECK-NEXT:                     affine.for %arg7 = #map2(%arg4) to #map4(%arg4) step 512 {
// CHECK-NEXT:                       affine.for %arg8 = #map2(%arg5) to #map7(%arg5) {
// CHECK-NEXT:                         affine.for %arg9 = #map2(%arg6) to #map6(%arg6) {
// CHECK-NEXT:                           affine.for %arg10 = #map2(%arg7) to #map4(%arg7) {
// CHECK-NEXT:                             %2 = affine.load %0[%arg8, %arg9, %arg10] : memref<20500x5300x2605xf32, strided<[?, ?, ?], offset: ?>>
// CHECK-NEXT:                             %3 = affine.load %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                             %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:                             affine.store %4, %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                           } {vector}
// CHECK-NEXT:                         }
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:               affine.for %arg4 = 2560 to 2605 step 45 {
// CHECK-NEXT:                 affine.for %arg5 = #map2(%arg2) to #map7(%arg2) step 493 {
// CHECK-NEXT:                   affine.for %arg6 = #map2(%arg3) to #map6(%arg3) step 180 {
// CHECK-NEXT:                     affine.for %arg7 = #map2(%arg4) to #map5(%arg4) step 45 {
// CHECK-NEXT:                       affine.for %arg8 = #map2(%arg5) to #map7(%arg5) {
// CHECK-NEXT:                         affine.for %arg9 = #map2(%arg6) to #map6(%arg6) {
// CHECK-NEXT:                           affine.for %arg10 = #map2(%arg7) to #map5(%arg7) {
// CHECK-NEXT:                             %2 = affine.load %0[%arg8, %arg9, %arg10] : memref<20500x5300x2605xf32, strided<[?, ?, ?], offset: ?>>
// CHECK-NEXT:                             %3 = affine.load %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                             %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:                             affine.store %4, %alloc[%arg8] : memref<20500xf32>
// CHECK-NEXT:                           } {vector}
// CHECK-NEXT:                         }
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       } {__tiled_for___1, map_for_to_forall}
// CHECK-NEXT:       %1 = bufferization.to_tensor %alloc : memref<20500xf32>
// CHECK-NEXT:       return %1 : tensor<20500xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }

module {
  func.func @reduce_3d(%arg0: tensor<20500x5300x2605xf32>)
      -> tensor<20500xf32>
      attributes {OperatorType = "Reduce", process = "aicore"} {
    %0 = bufferization.to_memref %arg0
         : memref<20500x5300x2605xf32, strided<[?, ?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64}
             : memref<20500xf32>
    affine.for %i = 0 to 20500 {
      affine.for %j = 0 to 5300 {
        affine.for %k = 0 to 2605 {
          %v0 = affine.load %0[%i, %j, %k]
                : memref<20500x5300x2605xf32, strided<[?, ?, ?], offset: ?>>
          %acc = affine.load %alloc[%i]
                 : memref<20500xf32>
          %sum = arith.addf %v0, %acc : f32
          affine.store %sum, %alloc[%i] : memref<20500xf32>
        }
      }
    }
    %1 = bufferization.to_tensor %alloc : memref<20500xf32>
    return %1 : tensor<20500xf32>
  }
}
