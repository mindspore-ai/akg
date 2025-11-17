// RUN: akg-opt %s --akg-affine-loop-tile="use-auto-tiling=true" -allow-unregistered-dialect | FileCheck %s --check-prefix=CHECK-ASCEND-AUTO

// 检查：自动切分会在外层生成 tile 循环，步长由 solver 决定。
// CHECK-ASCEND-AUTO-LABEL: func.func @tiling_only_innermost
// CHECK-ASCEND-AUTO: affine.for %[[I_TILE:.*]] = 0 to 256 step 32 {
// CHECK-ASCEND-AUTO:   affine.for %[[J_TILE:.*]] = 0 to 4000 step 32 {
// CHECK-ASCEND-AUTO:     affine.for %[[I_INNER:.*]] = #map(%[[I_TILE]]) to {{(min )?}}#map1(%[[I_TILE]]) {
// CHECK-ASCEND-AUTO:       affine.for %[[J_INNER:.*]] = #map(%[[J_TILE]]) to {{(min )?}}#map1(%[[J_TILE]]) {
// CHECK-ASCEND-AUTO:         affine.load %arg0[%[[I_INNER]], %[[J_INNER]]]
// CHECK-ASCEND-AUTO:       }
// CHECK-ASCEND-AUTO:     }
// CHECK-ASCEND-AUTO:   }
// CHECK-ASCEND-AUTO: }

func.func @tiling_only_innermost(%A: memref<256x256xf32>, %B: memref<256x256xf32>)
    attributes {process = "aicore", OperatorType = "PureElem"} {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 4000 {
      %val = affine.load %A[%i, %j] : memref<256x256xf32>
      affine.store %val, %B[%i, %j] : memref<256x256xf32>
    }
  }
  return
}

// CHECK-ASCEND-AUTO-LABEL: func.func @tiling_with_intermediate_store
// CHECK-ASCEND-AUTO: affine.for %[[I_TILE:.*]] = 0 to 8 step 4 {
// CHECK-ASCEND-AUTO:   affine.for %[[J_TILE:.*]] = 0 to 512 step 32 {
// CHECK-ASCEND-AUTO:     affine.for %[[I_INNER:.*]] = #map(%[[I_TILE]]) to #map2(%[[I_TILE]]) {
// CHECK-ASCEND-AUTO:       affine.for %[[J_INNER:.*]] = #map(%[[J_TILE]]) to #map1(%[[J_TILE]]) {
// CHECK-ASCEND-AUTO:         affine.store %{{.*}}, %arg2[%[[I_INNER]], %[[J_INNER]]] : memref<8x512xf32>
// CHECK-ASCEND-AUTO:         affine.for %[[K:.*]] = 0 to 4000 {
// CHECK-ASCEND-AUTO:           affine.store %{{.*}}, %arg1[%[[I_INNER]], %[[J_INNER]], %[[K]]] : memref<8x512x4000xf32>
// CHECK-ASCEND-AUTO:         }
// CHECK-ASCEND-AUTO:       }
// CHECK-ASCEND-AUTO:     }
// CHECK-ASCEND-AUTO:   }
// CHECK-ASCEND-AUTO: }

func.func @tiling_with_intermediate_store(%A: memref<8x512x4000xf32>, %B: memref<8x512x4000xf32>,
                                          %tmp: memref<8x512xf32>) attributes {process = "aicore", OperatorType = "PureElem"} {
  %cstf = arith.constant 0.000000e+00 : f32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 512 {
      affine.store %cstf, %tmp[%i, %j] : memref<8x512xf32>
      affine.for %k = 0 to 4000 {
        %val = affine.load %A[%i, %j, %k] : memref<8x512x4000xf32>
        affine.store %val, %B[%i, %j, %k] : memref<8x512x4000xf32>
      }
    }
  }
  return
}
