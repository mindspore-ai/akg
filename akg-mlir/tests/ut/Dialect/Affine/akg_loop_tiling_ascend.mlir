// RUN: akg-opt %s --akg-affine-loop-tile -allow-unregistered-dialect | FileCheck %s

// 检查：最内层 loop 被 512 步长切分，外层保持不变。
// CHECK-LABEL: func.func @tiling_only_innermost
// CHECK: affine.for %[[I:.*]] = 0 to 256 {
// CHECK:   affine.for %[[J_TILE:.*]] = 0 to 4000 step 512 {
// CHECK:     affine.for %[[J_INNER:.*]] = #map(%[[J_TILE]]) to min #map1(%[[J_TILE]]) {
// CHECK:       %[[VAL:.*]] = affine.load %arg0[%[[I]], %[[J_INNER]]]
// CHECK:       affine.store %[[VAL]], %arg1[%[[I]], %[[J_INNER]]]
// CHECK:     }
// CHECK:   }
// CHECK: }

func.func @tiling_only_innermost(%A: memref<256x256xf32>, %B: memref<256x256xf32>)
    attributes {process = "ascend"} {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 4000 {
      %val = affine.load %A[%i, %j] : memref<256x256xf32>
      affine.store %val, %B[%i, %j] : memref<256x256xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @tiling_with_intermediate_store
// CHECK: affine.for %[[I:.*]] = 0 to 8 {
// CHECK:   affine.for %[[J:.*]] = 0 to 512 {
// CHECK:     affine.store %{{.*}}, %arg2[%[[I]], %[[J]]] : memref<8x512xf32>
// CHECK:     affine.for %[[K_TILE:.*]] = 0 to 4000 step 512 {
// CHECK:       affine.for %[[K:.*]] = #map(%[[K_TILE]]) to min #map1(%[[K_TILE]]) {
// CHECK:         affine.store %{{.*}}, %arg1[%[[I]], %[[J]], %[[K]]] : memref<8x512x4000xf32>
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }

func.func @tiling_with_intermediate_store(%A: memref<8x512x4000xf32>, %B: memref<8x512x4000xf32>,
                                          %tmp: memref<8x512xf32>) attributes {process = "ascend"} {
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

