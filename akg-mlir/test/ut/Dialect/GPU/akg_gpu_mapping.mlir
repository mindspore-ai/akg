// RUN: akg-opt %s -split-input-file -akg-gpu-map-parallel-loops -allow-unregistered-dialect | FileCheck %s


// CHECK-LABEL: func.func @one_d_case(%arg0: memref<32xf32>, %arg1: memref<1xf32>, %arg2: memref<32xf32>) attributes {mindspore_kernel, thread_x = 32 : i32} {
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c32 = arith.constant 32 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %collapse_shape = memref.collapse_shape %arg1 [] : memref<1xf32> into memref<f32>
// CHECK-NEXT:      %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
// CHECK-NEXT:      scf.parallel (%arg3) = (%c0) to (%c32) step (%c1) {
// CHECK-NEXT:          scf.parallel (%arg4) = (%c0) to (%c1) step (%c1) {
// CHECK-NEXT:              %0 = arith.addi %arg4, %arg3 : index
// CHECK-NEXT:              %1 = memref.load %arg0[%0] : memref<32xf32>
// CHECK-NEXT:              %2 = memref.load %collapse_shape[] : memref<f32>
// CHECK-NEXT:              %3 = arith.addf %1, %2 : f32
// CHECK-NEXT:              memref.store %3, %alloc[%0] : memref<32xf32>
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:          } {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:      } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:      memref.dealloc %alloc : memref<32xf32>
// CHECK-NEXT:      memref.copy %alloc, %arg2 : memref<32xf32> to memref<32xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:  }


func.func @one_d_case(%arg0: memref<32xf32>, %arg1: memref<1xf32>, %arg2: memref<32xf32>) attributes {mindspore_kernel} {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %collapse_shape = memref.collapse_shape %arg1 [] : memref<1xf32> into memref<f32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
    scf.parallel (%arg3) = (%c0) to (%c32) step (%c1) {
        scf.parallel (%arg4) = (%c0) to (%c1) step (%c1) {
            %0 = arith.addi %arg4, %arg3 : index
            %1 = memref.load %arg0[%0] : memref<32xf32>
            %2 = memref.load %collapse_shape[] : memref<f32>
            %3 = arith.addf %1, %2 : f32
            memref.store %3, %alloc[%0] : memref<32xf32>
            scf.yield
        }
        scf.yield
    }
    memref.dealloc %alloc : memref<32xf32>
    memref.copy %alloc, %arg2 : memref<32xf32> to memref<32xf32>
    return
}


// -----

// CHECK-LABEL:  func.func @dynamic_shape_demo(%arg0: memref<?x768xf32>, %arg1: memref<768xf32>, %arg2: memref<?x768xf32>) attributes {OperatorType = "Reshape", block_x = -1 : i32, block_y = 24 : i32, enable_atomic_add = false, mindspore_kernel, process = "cuda", thread_x = 32 : i32, thread_y = 32 : i32} {
// CHECK-NEXT:    %c-32 = arith.constant -32 : index
// CHECK-NEXT:    %c24 = arith.constant 24 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c32 = arith.constant 32 : index
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
// CHECK-NEXT:    scf.parallel (%arg3) = (%c0) to (%7) step (%c1) {
// CHECK-NEXT:      %8 = arith.muli %arg3, %c32 : index
// CHECK-NEXT:      scf.parallel (%arg4) = (%c0) to (%c24) step (%c1) {
// CHECK-NEXT:        %9 = arith.muli %arg4, %c32 : index
// CHECK-NEXT:        %10 = arith.muli %arg3, %c-32 : index
// CHECK-NEXT:        %11 = arith.addi %10, %dim_0 : index
// CHECK-NEXT:        %12 = arith.cmpi sgt, %11, %c32 : index
// CHECK-NEXT:        %13 = arith.select %12, %c32, %11 : index
// CHECK-NEXT:        scf.parallel (%arg5) = (%c0) to (%13) step (%c1) {
// CHECK-NEXT:          %14 = arith.addi %arg5, %8 : index
// CHECK-NEXT:          scf.parallel (%arg6) = (%c0) to (%c32) step (%c1) {
// CHECK-NEXT:            %15 = arith.addi %arg6, %9 : index
// CHECK-NEXT:            %16 = memref.load %arg0[%14, %15] : memref<?x768xf32>
// CHECK-NEXT:            %17 = memref.load %arg1[%15] : memref<768xf32>
// CHECK-NEXT:            %18 = arith.addf %16, %17 : f32
// CHECK-NEXT:            memref.store %18, %alloc[%14, %15] : memref<?x768xf32>
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        } {mapping = [#gpu.loop_dim_map<processor = thread_y, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      } {mapping = [#gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:    memref.copy %alloc, %arg2 : memref<?x768xf32> to memref<?x768xf32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }


func.func @dynamic_shape_demo(%arg0: memref<?x768xf32>, %arg1: memref<768xf32>, %arg2: memref<?x768xf32>) attributes {OperatorType = "Reshape", enable_atomic_add = false, mindspore_kernel, process = "cuda"} {
    %c-32 = arith.constant -32 : index
    %c24 = arith.constant 24 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
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
    scf.parallel (%arg3) = (%c0) to (%7) step (%c1) {
      %8 = arith.muli %arg3, %c32 : index
      scf.parallel (%arg4) = (%c0) to (%c24) step (%c1) {
        %9 = arith.muli %arg4, %c32 : index
        %10 = arith.muli %arg3, %c-32 : index
        %11 = arith.addi %10, %dim_0 : index
        %12 = arith.cmpi sgt, %11, %c32 : index
        %13 = arith.select %12, %c32, %11 : index
        scf.parallel (%arg5) = (%c0) to (%13) step (%c1) {
          %14 = arith.addi %arg5, %8 : index
          scf.parallel (%arg6) = (%c0) to (%c32) step (%c1) {
            %15 = arith.addi %arg6, %9 : index
            %16 = memref.load %arg0[%14, %15] : memref<?x768xf32>
            %17 = memref.load %arg1[%15] : memref<768xf32>
            %18 = arith.addf %16, %17 : f32
            memref.store %18, %alloc[%14, %15] : memref<?x768xf32>
            scf.yield
          }
          scf.yield
        }
        scf.yield
      }
      scf.yield
    }
    memref.copy %alloc, %arg2 : memref<?x768xf32> to memref<?x768xf32>
    return
}


// -----

// CHECK-LABEL:  func.func @elem_broadcast_last_5(%arg0: memref<4096x?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<4096x?xf32>) attributes {OperatorType = "Reshape", block_x = 16 : i32, block_y = -1 : i32, block_z = 256 : i32, enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored, thread_x = 32 : i32} {
// CHECK-NEXT:      %c-2 = arith.constant -2 : index
// CHECK-NEXT:      %c-32 = arith.constant -32 : index
// CHECK-NEXT:      %c32 = arith.constant 32 : index
// CHECK-NEXT:      %c256 = arith.constant 256 : index
// CHECK-NEXT:      %c16 = arith.constant 16 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %dim = memref.dim %arg0, %c1 : memref<4096x?xf32>
// CHECK-NEXT:      %dim_0 = memref.dim %arg1, %c0 : memref<?xf32>
// CHECK-NEXT:      %dim_1 = memref.dim %arg2, %c0 : memref<?xf32>
// CHECK-NEXT:      %dim_2 = memref.dim %arg3, %c1 : memref<4096x?xf32>
// CHECK-NEXT:      scf.parallel (%arg4) = (%c0) to (%c1) step (%c1) {
// CHECK-NEXT:        scf.parallel (%arg5) = (%c0) to (%c16) step (%c1) {
// CHECK-NEXT:          %0 = arith.muli %arg5, %c256 : index
// CHECK-NEXT:          %1 = arith.cmpi sle, %dim_2, %c0 : index
// CHECK-NEXT:          %2 = arith.subi %c0, %dim_2 : index
// CHECK-NEXT:          %3 = arith.subi %dim_2, %c1 : index
// CHECK-NEXT:          %4 = arith.select %1, %2, %3 : index
// CHECK-NEXT:          %5 = arith.divsi %4, %c32 : index
// CHECK-NEXT:          %6 = arith.subi %c0, %5 : index
// CHECK-NEXT:          %7 = arith.addi %5, %c1 : index
// CHECK-NEXT:          %8 = arith.select %1, %6, %7 : index
// CHECK-NEXT:          scf.parallel (%arg6) = (%c0) to (%8) step (%c1) {
// CHECK-NEXT:            %9 = arith.muli %arg6, %c32 : index
// CHECK-NEXT:            scf.parallel (%arg7) = (%c0) to (%c256) step (%c1) {
// CHECK-NEXT:              %10 = arith.addi %arg7, %0 : index
// CHECK-NEXT:              %11 = arith.muli %arg6, %c-32 : index
// CHECK-NEXT:              %12 = arith.addi %11, %dim_2 : index
// CHECK-NEXT:              %13 = arith.cmpi sgt, %12, %c32 : index
// CHECK-NEXT:              %14 = arith.select %13, %c32, %12 : index
// CHECK-NEXT:              scf.parallel (%arg8) = (%c0) to (%14) step (%c1) {
// CHECK-NEXT:                %15 = arith.addi %arg8, %9 : index
// CHECK-NEXT:                %16 = arith.addi %dim_1, %c-2 : index
// CHECK-NEXT:                %17 = arith.cmpi eq, %16, %c0 : index
// CHECK-NEXT:                %18 = scf.if %17 -> (index) {
// CHECK-NEXT:                  scf.yield %c0 : index
// CHECK-NEXT:                } else {
// CHECK-NEXT:                  scf.yield %15 : index
// CHECK-NEXT:                }
// CHECK-NEXT:                %19 = arith.addi %dim, %c-2 : index
// CHECK-NEXT:                %20 = arith.cmpi eq, %19, %c0 : index
// CHECK-NEXT:                %21 = scf.if %20 -> (index) {
// CHECK-NEXT:                  scf.yield %c0 : index
// CHECK-NEXT:                } else {
// CHECK-NEXT:                  scf.yield %15 : index
// CHECK-NEXT:                }
// CHECK-NEXT:                %22 = arith.addi %dim_0, %c-2 : index
// CHECK-NEXT:                %23 = arith.cmpi eq, %22, %c0 : index
// CHECK-NEXT:                %24 = scf.if %23 -> (index) {
// CHECK-NEXT:                  scf.yield %c0 : index
// CHECK-NEXT:                } else {
// CHECK-NEXT:                  scf.yield %15 : index
// CHECK-NEXT:                }
// CHECK-NEXT:                %25 = memref.load %arg2[%18] : memref<?xf32>
// CHECK-NEXT:                %26 = memref.load %arg0[%10, %21] : memref<4096x?xf32>
// CHECK-NEXT:                %27 = memref.load %arg1[%24] : memref<?xf32>
// CHECK-NEXT:                %28 = arith.addf %26, %27 : f32
// CHECK-NEXT:                %29 = arith.mulf %25, %28 : f32
// CHECK-NEXT:                memref.store %29, %arg3[%10, %15] : memref<4096x?xf32>
// CHECK-NEXT:                scf.yield
// CHECK-NEXT:              } {mapping = [#gpu.loop_dim_map<processor = thread_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:              scf.yield
// CHECK-NEXT:            } {mapping = [#gpu.loop_dim_map<processor = block_z, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:            scf.yield
// CHECK-NEXT:          } {mapping = [#gpu.loop_dim_map<processor = block_y, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:          scf.yield
// CHECK-NEXT:        } {mapping = [#gpu.loop_dim_map<processor = block_x, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:        scf.yield
// CHECK-NEXT:      } {mapping = [#gpu.loop_dim_map<processor = sequential, map = (d0) -> (d0), bound = (d0) -> (d0)>]}
// CHECK-NEXT:      return
// CHECK-NEXT:  }


func.func @elem_broadcast_last_5(%arg0: memref<4096x?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<4096x?xf32>) attributes {OperatorType = "Reshape", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored} {
    %c-2 = arith.constant -2 : index
    %c-32 = arith.constant -32 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = memref.dim %arg0, %c1 : memref<4096x?xf32>
    %dim_0 = memref.dim %arg1, %c0 : memref<?xf32>
    %dim_1 = memref.dim %arg2, %c0 : memref<?xf32>
    %dim_2 = memref.dim %arg3, %c1 : memref<4096x?xf32>
    scf.for %arg4 = %c0 to %c1 step %c1 {
      scf.for %arg5 = %c0 to %c16 step %c1 {
        %0 = arith.muli %arg5, %c256 : index
        %1 = arith.cmpi sle, %dim_2, %c0 : index
        %2 = arith.subi %c0, %dim_2 : index
        %3 = arith.subi %dim_2, %c1 : index
        %4 = arith.select %1, %2, %3 : index
        %5 = arith.divsi %4, %c32 : index
        %6 = arith.subi %c0, %5 : index
        %7 = arith.addi %5, %c1 : index
        %8 = arith.select %1, %6, %7 : index
        scf.for %arg6 = %c0 to %8 step %c1 {
          %9 = arith.muli %arg6, %c32 : index
          scf.for %arg7 = %c0 to %c256 step %c1 {
            %10 = arith.addi %arg7, %0 : index
            %11 = arith.muli %arg6, %c-32 : index
            %12 = arith.addi %11, %dim_2 : index
            %13 = arith.cmpi sgt, %12, %c32 : index
            %14 = arith.select %13, %c32, %12 : index
            scf.for %arg8 = %c0 to %14 step %c1 {
              %15 = arith.addi %arg8, %9 : index
              %16 = arith.addi %dim_1, %c-2 : index
              %17 = arith.cmpi eq, %16, %c0 : index
              %18 = scf.if %17 -> (index) {
                scf.yield %c0 : index
              } else {
                scf.yield %15 : index
              }
              %19 = arith.addi %dim, %c-2 : index
              %20 = arith.cmpi eq, %19, %c0 : index
              %21 = scf.if %20 -> (index) {
                scf.yield %c0 : index
              } else {
                scf.yield %15 : index
              }
              %22 = arith.addi %dim_0, %c-2 : index
              %23 = arith.cmpi eq, %22, %c0 : index
              %24 = scf.if %23 -> (index) {
                scf.yield %c0 : index
              } else {
                scf.yield %15 : index
              }
              %25 = memref.load %arg2[%18] : memref<?xf32>
              %26 = memref.load %arg0[%10, %21] : memref<4096x?xf32>
              %27 = memref.load %arg1[%24] : memref<?xf32>
              %28 = arith.addf %26, %27 : f32
              %29 = arith.mulf %25, %28 : f32
              memref.store %29, %arg3[%10, %15] : memref<4096x?xf32>
            }
          }
        }
      }
    }
    return
}

