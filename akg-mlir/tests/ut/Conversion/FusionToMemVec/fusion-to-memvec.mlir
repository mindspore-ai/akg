// RUN: akg-opt %s -split-input-file -lower-fusion | FileCheck %s


#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @template_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %m: index, %n: index, %k:index) {
    %0 = bufferization.to_memref %arg0 : memref<?x?xf32>
    %1 = bufferization.to_memref %arg1 : memref<?x?xf32>
    %2 = bufferization.to_memref %arg2 : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %m step %c1 {
      scf.for %arg4 = %c0 to %n step %c1 {
        scf.for %arg5 = %c0 to %k step %c1 {
          %6 = fusion.load %0[%arg3, %arg5] : memref<?x?xf32>, f32
          %7 = memref.load %1[%arg5, %arg4] : memref<?x?xf32>
          %8 = memref.load %2[%arg3, %arg4] : memref<?x?xf32>
          %9 = fusion.insert %0, %6 : memref<?x?xf32>, f32 to f32
          %10 = arith.mulf %9, %7 : f32
          %11 = arith.addf %8, %10 : f32
          fusion.store %11, %2[%arg3, %arg4] : f32, memref<?x?xf32>
        }
      }
    }
    return
  }
  func.func @template_matmul_buffer(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) {
    %0 = linalg.template {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul} {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %1 = arith.mulf %arg3, %arg4 : f32
      %2 = arith.addf %arg5, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<16x32xf32>
    return
  }
}

// CHECK-NOT: fusion.load
// CHECK-NOT: fusion.insert
// CHECK-NOT: fusion.store

// -----
 
func.func @empty(%arg0: memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = fusion.load %arg0[%c0, %c0] : memref<?x?xf32>, f32
  %2 = fusion.load %arg0[%c0, %c0] : memref<?x?xf32>, vector<8x1xf32>
  %3 = fusion.load %arg0[%c0, %c0], %cst : memref<?x?xf32>, vector<8x1xf32>
  %4 = fusion.load %arg0[%c0, %c0], %cst {in_bounds = [false, true]} : memref<?x?xf32>, vector<8x1xf32>
  fusion.store %1, %arg0[%c0, %c0] : f32, memref<?x?xf32>
  fusion.store %2, %arg0[%c0, %c0] : vector<8x1xf32>, memref<?x?xf32>
  return
}
 
// CHECK-NOT: fusion.load
// CHECK-NOT: fusion.store
// CHECK: %[[ARG0:.+]] = memref.load
// CHECK: %[[ARG1:.+]] = vector.transfer_read
// CHECK: %[[ARG2:.+]] = vector.transfer_read
// CHECK: %[[ARG3:.+]] = vector.transfer_read
// CHECK-NEXT: memref.store
// CHECK-NEXT: vector.transfer_write

// -----
 
func.func @empty(%arg0 : index, %arg1 : index, %arg2 : index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = memref.alloc() : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>>
    %1 = fusion.subview %0[%c0, %c0, %c0][%arg0, %arg1, %arg2][%c1, %c1, %c1]
            : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
              memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
    return
}
 
// CHECK-NOT: fusion.subview
// CHECK: %[[ARG0:.+]] = memref.subview

// -----
 
func.func @empty(%a: f32, %b: vector<f32>, %c: vector<16xf32>, %d: vector<1x16xf32>, %e: vector<8x1xf32>) -> vector<8x16xf32> {
  %0 = fusion.broadcast %a : f32 to vector<f32>
  %1 = fusion.broadcast %b : vector<f32> to vector<4xf32>
  %2 = fusion.broadcast %a : f32 to vector<16xf32>
  %3 = fusion.broadcast %c : vector<16xf32> to vector<8x16xf32>
  %4 = fusion.broadcast %d : vector<1x16xf32> to vector<8x16xf32>
  %5 = fusion.broadcast %e : vector<8x1xf32> to vector<8x16xf32>
  return %4 : vector<8x16xf32>
}
 
// CHECK-NOT: fusion.broadcast
// CHECK: %[[ARG0:.+]] = vector.broadcast

// -----
 
func.func @empty(%arg0: vector<i32>) -> vector<i32> {
  %0 = fusion.transpose %arg0, [] : vector<i32> to vector<i32>
  return %0 : vector<i32>
}
 
// CHECK-NOT: fusion.transpose
// CHECK: %[[X:.*]] = vector.transpose %{{.*}}, [] : vector<i32> to vector<i32>

// -----

func.func @test_multi_load_lower(%arg : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %1 = fusion.multi_load %arg[%c0, %c0] : memref<?x?xf32>, f32
  %2 = fusion.multi_load %arg[%c0, %c0] : memref<?x?xf32>, vector<8x1xf32>
  return
}

// CHECK-NOT: fusion.multi_load