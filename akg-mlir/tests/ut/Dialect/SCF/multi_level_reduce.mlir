// RUN: akg-opt %s -split-input-file --rewrite-reduce-in-multi-level-memory -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL:  module {
// CHECK-NEXT:     func.func @allreduce(%arg0: memref<320xf32>, %arg1: memref<f32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored, thread_x = 10 : i32} {
// CHECK-NEXT:       %c32 = arith.constant 32 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       %c10 = arith.constant 10 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
// CHECK-NEXT:       scf.parallel (%arg2) = (%c0) to (%c10) step (%c1) {
// CHECK-NEXT:         %0 = arith.muli %arg2, %c32 : index
// CHECK-NEXT:         %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:         %alloc_1 = memref.alloc() : memref<f32, 5>
// CHECK-NEXT:         memref.store %cst_0, %alloc_1[] : memref<f32, 5>
// CHECK-NEXT:         scf.parallel (%arg3) = (%c0) to (%c32) step (%c1) {
// CHECK-NEXT:           %4 = arith.addi %arg3, %0 : index
// CHECK-NEXT:           %alloc_3 = memref.alloc() : memref<1xf32, 5>
// CHECK-NEXT:           %5 = memref.load %arg0[%4] : memref<320xf32>
// CHECK-NEXT:           memref.store %5, %alloc_3[%c0] : memref<1xf32, 5>
// CHECK-NEXT:           %6 = memref.load %alloc_3[%c0] : memref<1xf32, 5>
// CHECK-NEXT:           %7 = memref.load %alloc_1[] : memref<f32, 5>
// CHECK-NEXT:           %8 = arith.addf %6, %7 : f32
// CHECK-NEXT:           memref.store %8, %alloc_1[] : memref<f32, 5>
// CHECK-NEXT:           memref.dealloc %alloc_3 : memref<1xf32, 5>
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         } {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>], reduceLoop}
// CHECK-NEXT:         %1 = memref.load %alloc_1[] : memref<f32, 5>
// CHECK-NEXT:         %cst_2 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:         %2 = arith.addf %1, %cst_2 {gpu_parallel_reduce = true, reduction_axes = [0 : index], reduction_type = "all"} : f32
// CHECK-NEXT:         memref.store %2, %alloc_1[] : memref<f32, 5>
// CHECK-NEXT:         %3 = memref.load %alloc_1[] : memref<f32, 5>
// CHECK-NEXT:         memref.store %3, %alloc[] : memref<f32>
// CHECK-NEXT:         memref.dealloc %alloc_1 : memref<f32, 5>
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>], reduceLoop}
// CHECK-NEXT:       memref.copy %alloc, %arg1 : memref<f32> to memref<f32>
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }


module {
  func.func @allreduce(%arg0: memref<320xf32>, %arg1: memref<f32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored, thread_x = 10 : i32} {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    scf.parallel (%arg2) = (%c0) to (%c10) step (%c1) {
      %0 = arith.muli %arg2, %c32 : index
      memref.store %cst, %alloc[] : memref<f32>
      scf.parallel (%arg3) = (%c0) to (%c32) step (%c1) {
        %1 = arith.addi %arg3, %0 : index
        %alloc_0 = memref.alloc() : memref<1xf32, 5>
        %2 = memref.load %arg0[%1] : memref<320xf32>
        memref.store %2, %alloc_0[%c0] : memref<1xf32, 5>
        %alloc_1 = memref.alloc() : memref<f32, 5>
        %3 = memref.load %alloc[] : memref<f32>
        memref.store %3, %alloc_1[] : memref<f32, 5>
        %4 = memref.load %alloc_0[%c0] : memref<1xf32, 5>
        %5 = memref.load %alloc_1[] : memref<f32, 5>
        %6 = arith.addf %4, %5 {reduction_axes = [0 : index], reduction_type = "all", gpu_parallel_reduce = true} : f32
        memref.store %6, %alloc_1[] : memref<f32, 5>
        %7 = memref.load %alloc_1[] : memref<f32, 5>
        memref.store %7, %alloc[] : memref<f32>
        memref.dealloc %alloc_1 : memref<f32, 5>
        memref.dealloc %alloc_0 : memref<1xf32, 5>
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>], reduceLoop}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>], reduceLoop}
    memref.copy %alloc, %arg1 : memref<f32> to memref<f32>
    return
  }
}