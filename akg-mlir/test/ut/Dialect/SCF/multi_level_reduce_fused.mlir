// RUN: akg-opt %s -split-input-file --rewrite-reduce-in-multi-level-memory -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL:  module {
// CHECK-NEXT:     func.func @prefuse_reduce_bd_demo(%arg0: memref<1024x768xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) attributes {OperatorType = "Reduce", block_x = 1024 : i32, enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored, thread_x = 32 : i32} {
// CHECK-NEXT:       %c24 = arith.constant 24 : index
// CHECK-NEXT:       %c32 = arith.constant 32 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       %c1024 = arith.constant 1024 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
// CHECK-NEXT:       scf.parallel (%arg3) = (%c0) to (%c1024) step (%c1) {
// CHECK-NEXT:         scf.parallel (%arg4) = (%c0) to (%c32) step (%c1) {
// CHECK-NEXT:           %0 = arith.muli %arg4, %c24 : index
// CHECK-NEXT:           scf.parallel (%arg5) = (%c0) to (%c1) step (%c1) {
// CHECK-NEXT:             %1 = arith.addi %arg5, %arg3 : index
// CHECK-NEXT:             %alloc_0 = memref.alloc() : memref<1xf32, 5>
// CHECK-NEXT:             %2 = memref.load %arg1[%1] : memref<1024xf32>
// CHECK-NEXT:             memref.store %2, %alloc_0[%c0] : memref<1xf32, 5>
// CHECK-NEXT:             %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:             %alloc_1 = memref.alloc() : memref<1xf32, 5>
// CHECK-NEXT:             memref.store %cst, %alloc_1[%c0] : memref<1xf32, 5>
// CHECK-NEXT:             scf.parallel (%arg6) = (%c0) to (%c24) step (%c1) {
// CHECK-NEXT:               %6 = arith.addi %arg6, %0 : index
// CHECK-NEXT:               %alloc_3 = memref.alloc() : memref<1x1xf32, 5>
// CHECK-NEXT:               %7 = memref.load %arg0[%1, %6] : memref<1024x768xf32>
// CHECK-NEXT:               memref.store %7, %alloc_3[%c0, %c0] : memref<1x1xf32, 5>
// CHECK-NEXT:               %8 = memref.load %alloc_3[%c0, %c0] : memref<1x1xf32, 5>
// CHECK-NEXT:               %9 = memref.load %alloc_0[%c0] : memref<1xf32, 5>
// CHECK-NEXT:               %10 = arith.addf %8, %9 : f32
// CHECK-NEXT:               %11 = memref.load %alloc_1[%c0] : memref<1xf32, 5>
// CHECK-NEXT:               %12 = arith.addf %10, %11 : f32
// CHECK-NEXT:               memref.store %12, %alloc_1[%c0] : memref<1xf32, 5>
// CHECK-NEXT:               memref.dealloc %alloc_3 : memref<1x1xf32, 5>
// CHECK-NEXT:               scf.yield
// CHECK-NEXT:             } {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>], reduceLoop}
// CHECK-NEXT:             %3 = memref.load %alloc_1[%c0] : memref<1xf32, 5>
// CHECK-NEXT:             %cst_2 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:             %4 = arith.addf %3, %cst_2 {gpu_parallel_reduce = true, reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK-NEXT:             memref.store %4, %alloc_1[%c0] : memref<1xf32, 5>
// CHECK-NEXT:             %5 = memref.load %alloc_1[%c0] : memref<1xf32, 5>
// CHECK-NEXT:             memref.store %5, %alloc[%1] : memref<1024xf32>
// CHECK-NEXT:             memref.dealloc %alloc_1 : memref<1xf32, 5>
// CHECK-NEXT:             memref.dealloc %alloc_0 : memref<1xf32, 5>
// CHECK-NEXT:             scf.yield
// CHECK-NEXT:           } {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:           scf.yield
// CHECK-NEXT:         } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>], reduceLoop}
// CHECK-NEXT:         scf.yield
// CHECK-NEXT:       } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:       memref.copy %alloc, %arg2 : memref<1024xf32> to memref<1024xf32>
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

module {
  func.func @prefuse_reduce_bd_demo(%arg0: memref<1024x768xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>) attributes {OperatorType = "Reduce", block_x = 1024 : i32, enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored, thread_x = 32 : i32} {
    %c24 = arith.constant 24 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024xf32>
    scf.parallel (%arg3) = (%c0) to (%c1024) step (%c1) {
      scf.parallel (%arg4) = (%c0) to (%c32) step (%c1) {
        %0 = arith.muli %arg4, %c24 : index
        scf.parallel (%arg5) = (%c0) to (%c1) step (%c1) {
          %1 = arith.addi %arg5, %arg3 : index
          %alloc_0 = memref.alloc() : memref<1xf32, 5>
          %2 = memref.load %arg1[%1] : memref<1024xf32>
          memref.store %2, %alloc_0[%c0] : memref<1xf32, 5>
          %alloc_1 = memref.alloc() : memref<1xf32, 5>
          %3 = memref.load %alloc[%1] : memref<1024xf32>
          memref.store %3, %alloc_1[%c0] : memref<1xf32, 5>
          scf.parallel (%arg6) = (%c0) to (%c24) step (%c1) {
            %5 = arith.addi %arg6, %0 : index
            %alloc_2 = memref.alloc() : memref<1x1xf32, 5>
            %6 = memref.load %arg0[%1, %5] : memref<1024x768xf32>
            memref.store %6, %alloc_2[%c0, %c0] : memref<1x1xf32, 5>
            %7 = memref.load %alloc_2[%c0, %c0] : memref<1x1xf32, 5>
            %8 = memref.load %alloc_0[%c0] : memref<1xf32, 5>
            %9 = arith.addf %7, %8 : f32
            %10 = memref.load %alloc_1[%c0] : memref<1xf32, 5>
            %11 = arith.addf %9, %10 {reduction_axes = [1 : index], reduction_type = "x", gpu_parallel_reduce = true} : f32
            memref.store %11, %alloc_1[%c0] : memref<1xf32, 5>
            memref.dealloc %alloc_2 : memref<1x1xf32, 5>
            scf.yield
          } {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>], reduceLoop}
          %4 = memref.load %alloc_1[%c0] : memref<1xf32, 5>
          memref.store %4, %alloc[%1] : memref<1024xf32>
          memref.dealloc %alloc_1 : memref<1xf32, 5>
          memref.dealloc %alloc_0 : memref<1xf32, 5>
          scf.yield
        } {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
        scf.yield
      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>], reduceLoop}
      scf.yield
    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
    memref.copy %alloc, %arg2 : memref<1024xf32> to memref<1024xf32>
    return
  }
}