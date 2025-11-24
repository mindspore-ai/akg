// RUN: akg-opt %s -split-input-file -affine-memory-promotion | FileCheck %s


// CHECK-LABEL: #map = affine_map<(d0) -> (d0)>
// CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 + 28)>
// CHECK-NEXT:  #set = affine_set<(d0) : (d0 == 0)>
// CHECK-NEXT:  #set1 = affine_set<(d0) : (-d0 + 27 == 0)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    func.func @allreduce_post(%arg0: memref<28x28xf32>, %arg1: memref<f32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %cst_0 = arith.constant 0.00127551018 : f32
// CHECK-NEXT:      %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32, 5>
// CHECK-NEXT:      %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<f32>
// CHECK-NEXT:      affine.for %arg2 = 0 to 28 step 28 {
// CHECK-NEXT:        affine.for %arg3 = 0 to 28 step 28 {
// CHECK-NEXT:          affine.for %arg4 = #map(%arg2) to #map1(%arg2) {
// CHECK-NEXT:            affine.for %arg5 = #map(%arg3) to #map1(%arg3) {
// CHECK-NEXT:              affine.if #set(%arg5) {
// CHECK-NEXT:                affine.if #set(%arg4) {
// CHECK-NEXT:                  affine.store %cst, %alloc[] : memref<f32, 5>
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:              %0 = affine.load %arg0[%arg4, %arg5] : memref<28x28xf32>
// CHECK-NEXT:              %1 = affine.load %alloc[] : memref<f32, 5>
// CHECK-NEXT:              %2 = arith.addf %0, %1 {enable_atomic_add = false, gpu_parallel_reduce = true, reduction_axes = [0 : index, 1 : index], reduction_type = "all"} : f32
// CHECK-NEXT:              affine.store %2, %alloc[] : memref<f32, 5>
// CHECK-NEXT:              affine.if #set1(%arg5) {
// CHECK-NEXT:                affine.if #set1(%arg4) {
// CHECK-NEXT:                  %3 = affine.load %alloc[] : memref<f32, 5>
// CHECK-NEXT:                  %4 = arith.mulf %3, %cst_0 : f32
// CHECK-NEXT:                  affine.store %4, %alloc_1[] : memref<f32>
// CHECK-NEXT:                }
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.copy %alloc_1, %arg1 : memref<f32> to memref<f32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }




#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 28)>
#set = affine_set<(d0) : (d0 == 0)>
#set1 = affine_set<(d0) : (-d0 + 27 == 0)>
module {
  func.func @allreduce_post(%arg0: memref<28x28xf32>, %arg1: memref<f32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.00127551018 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    affine.for %arg2 = 0 to 28 step 28 {
      affine.for %arg3 = 0 to 28 step 28 {
        affine.for %arg4 = #map(%arg2) to #map1(%arg2) {
          affine.for %arg5 = #map(%arg3) to #map1(%arg3) {
            affine.if #set(%arg5) {
              affine.if #set(%arg4) {
                affine.store %cst, %alloc[] : memref<f32>
              }
            }
            %0 = affine.load %arg0[%arg4, %arg5] : memref<28x28xf32>
            %1 = affine.load %alloc[] : memref<f32>
            %2 = arith.addf %0, %1 {enable_atomic_add = false, gpu_parallel_reduce = true, reduction_axes = [0 : index, 1 : index], reduction_type = "all"} : f32
            affine.store %2, %alloc[] : memref<f32>
            affine.if #set1(%arg5) {
              affine.if #set1(%arg4) {
                %3 = affine.load %alloc[] : memref<f32>
                %4 = arith.mulf %3, %cst_0 : f32
                affine.store %4, %alloc_1[] : memref<f32>
              }
            }
          }
        }
      }
    }
    memref.copy %alloc_1, %arg1 : memref<f32> to memref<f32>
    return
  }
}


