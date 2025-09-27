// RUN: akg-opt %s -split-input-file -linalg-lower-template-ops | FileCheck %s


#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @template_matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>, %m: index, %n: index, %k:index) {
    %0 = bufferization.to_tensor %arg2 : memref<?x?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x?xf32>
    %2 = bufferization.to_tensor %arg0 : memref<?x?xf32>
    %3 = bufferization.to_memref %2 : memref<?x?xf32>
    %4 = bufferization.to_memref %1 : memref<?x?xf32>
    %5 = bufferization.to_memref %0 : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %m step %c1 {
      scf.for %arg4 = %c0 to %n step %c1 {
        scf.for %arg5 = %c0 to %k step %c1 {
          %9 = memref.load %3[%arg3, %arg5] : memref<?x?xf32>
          %10 = memref.load %4[%arg5, %arg4] : memref<?x?xf32>
          %11 = memref.load %5[%arg3, %arg4] : memref<?x?xf32>
          %12 = arith.mulf %9, %10 : f32
          %13 = arith.addf %11, %12 : f32
          memref.store %13, %5[%arg3, %arg4] : memref<?x?xf32>
        }
      }
    }
    return
  }
  func.func @template_matmul_buffer(%arg0: memref<16x8xf32>, %arg1: memref<8x32xf32>, %arg2: memref<16x32xf32>) {
    %0 = bufferization.to_tensor %arg2 : memref<16x32xf32>
    %1 = bufferization.to_tensor %arg1 : memref<8x32xf32>
    %2 = bufferization.to_tensor %arg0 : memref<16x8xf32>
    %3 = bufferization.to_memref %1 : memref<8x32xf32>
    %4 = bufferization.to_memref %2 : memref<16x8xf32>
    %5 = bufferization.to_memref %0 : memref<16x32xf32>
    %6 = memref.alloc() {alignment = 128 : i64} : memref<16x32xf32>
    memref.copy %5, %6 : memref<16x32xf32> to memref<16x32xf32>
    %7 = bufferization.to_tensor %6 : memref<16x32xf32>
    linalg.template {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%4, %3 : memref<16x8xf32>, memref<8x32xf32>) outs(%6 : memref<16x32xf32>) attrs =  {template_func = @template_matmul} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %9 = arith.mulf %arg3, %arg4 : f32
      %10 = arith.addf %arg5, %9 : f32
      linalg.yield %10 : f32
    }
    %8 = bufferization.to_tensor %6 : memref<16x32xf32>
    return
  }
}

// CHECK-NOT: linalg.template
// CHECK: call @template_matmul(%[[ARG0:.+]], %[[ARG1:.+]], %[[ARG2:.+]]
// -----
