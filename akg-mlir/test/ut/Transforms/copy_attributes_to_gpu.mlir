// RUN: akg-opt %s -copy-attributes-to-gpu | FileCheck %s

// CHECK-LABEL: module attributes {gpu.container_module} {
// CHECK-NEXT:  func.func @Fused_BiasAdd_17839785675172125010(%arg0: memref<?x768xf32>, %arg1: memref<768xf32>, %arg2: memref<?x768xf32>) attributes {OperatorType = "Reshape", block_x = 24 : i32, enable_atomic_add = false, mindspore_kernel, process = "cuda", thread_x = 32 : i32} {
// CHECK-NEXT:    %c24 = arith.constant 24 : index
// CHECK-NEXT:    %c32 = arith.constant 32 : index
// CHECK-NEXT:    %c-32 = arith.constant -32 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %dim = memref.dim %arg0, %c0 : memref<?x768xf32>
// CHECK-NEXT:    %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<?x768xf32>
// CHECK-NEXT:    %dim_0 = memref.dim %arg0, %c0 : memref<?x768xf32>
// CHECK-NEXT:    %0 = arith.cmpi sle, %dim_0, %c0 : index
// CHECK-NEXT:    %1 = arith.subi %c0, %dim_0 : index
// CHECK-NEXT:    %2 = arith.subi %dim_0, %c1 : index
// CHECK-NEXT:    %3 = arith.select %0, %1, %2 : index
// CHECK-NEXT:    %4 = arith.divsi %3, %c32 : index
// CHECK-NEXT:    %5 = arith.subi %c0, %4 : index
// CHECK-NEXT:    %6 = arith.addi %4, %c1 : index
// CHECK-NEXT:    %7 = arith.select %0, %5, %6 : index
// CHECK-NEXT:    gpu.launch_func  @Fused_BiasAdd_17839785675172125010_kernel::@Fused_BiasAdd_17839785675172125010_kernel blocks in (%c24, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<?x768xf32>, %arg1 : memref<768xf32>, %alloc : memref<?x768xf32>, %7 : index)
// CHECK-NEXT:    memref.copy %alloc, %arg2 : memref<?x768xf32> to memref<?x768xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  gpu.module @Fused_BiasAdd_17839785675172125010_kernel {
// CHECK-NEXT:    gpu.func @Fused_BiasAdd_17839785675172125010_kernel(%arg0: memref<?x768xf32>, %arg1: memref<768xf32>, %arg2: memref<?x768xf32>, %arg3: index) kernel attributes {OperatorType = "Reshape", block_x = 24 : i32, enable_atomic_add = false, gpu.known_block_size = array<i32: 32, 1, 1>, gpu.known_grid_size = array<i32: 24, 1, 1>, mindspore_kernel, process = "cuda", thread_x = 32 : i32} {
// CHECK-NEXT:      gpu.return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:}


module attributes {gpu.container_module} {
  func.func @Fused_BiasAdd_17839785675172125010(%arg0: memref<?x768xf32>, %arg1: memref<768xf32>, %arg2: memref<?x768xf32>) attributes {OperatorType = "Reshape", block_x = 24 : i32, enable_atomic_add = false, mindspore_kernel, process = "cuda", thread_x = 32 : i32} {
    %c24 = arith.constant 24 : index
    %c32 = arith.constant 32 : index
    %c-32 = arith.constant -32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x768xf32>
    %alloc = memref.alloc(%dim) {alignment = 64 : i64} : memref<?x768xf32>
    %dim_0 = memref.dim %arg0, %c0 : memref<?x768xf32>
    %0 = arith.cmpi sle, %dim_0, %c0 : index
    %1 = arith.subi %c0, %dim_0 : index
    %2 = arith.subi %dim_0, %c1 : index
    %3 = arith.select %0, %1, %2 : index
    %4 = arith.divsi %3, %c32 : index
    %5 = arith.subi %c0, %4 : index
    %6 = arith.addi %4, %c1 : index
    %7 = arith.select %0, %5, %6 : index
    gpu.launch_func  @Fused_BiasAdd_17839785675172125010_kernel::@Fused_BiasAdd_17839785675172125010_kernel blocks in (%c24, %c1, %c1) threads in (%c32, %c1, %c1) args(%arg0 : memref<?x768xf32>, %arg1 : memref<768xf32>, %alloc : memref<?x768xf32>, %7 : index)
    memref.copy %alloc, %arg2 : memref<?x768xf32> to memref<?x768xf32>
    return
  }
  gpu.module @Fused_BiasAdd_17839785675172125010_kernel {
    gpu.func @Fused_BiasAdd_17839785675172125010_kernel(%arg0: memref<?x768xf32>, %arg1: memref<768xf32>, %arg2: memref<?x768xf32>, %arg3: index) kernel attributes {mindspore_kernel, gpu.known_block_size = array<i32: 32, 1, 1>, gpu.known_grid_size = array<i32: 24, 1, 1>} {
      gpu.return
    }
  }
}

