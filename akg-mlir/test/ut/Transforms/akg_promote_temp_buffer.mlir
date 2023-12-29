// RUN: akg-opt %s -split-input-file -promote-temp-buffer | FileCheck %s


module attributes {gpu.container_module} {
  func.func @Fused_Mul_Mul_Add_RealDiv_Mul_Mul_Mul_Add_RealDiv_Add_Rsqrt_Mul_Mul_Add_Mul_Redu_more_split_7756657581262259816(%arg0: memref<15xf32>, %arg1: memref<15xf32>, %arg2: memref<15xf32>, %arg3: memref<15xf32>, %arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<15xf32>, %arg7: memref<f32>, %arg8: memref<15xf32>, %arg9: memref<15xf32>, %arg10: memref<15xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored, thread_x = 15 : i32} {
    %c15 = arith.constant 15 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.100000024 : f32
    %cst_0 = arith.constant 0.899999976 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 9.99987125E-4 : f32
    %cst_3 = arith.constant 9.990000e-01 : f32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %cst_5 = arith.constant 9.99999997E-7 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<15xf32>
    gpu.launch_func  @Fused_Mul_Mul_Add_RealDiv_Mul_Mul_Mul_Add_RealDiv_Add_Rsqrt_Mul_Mul_Add_Mul_Redu_more_split_7756657581262259816_kernel::@Fused_Mul_Mul_Add_RealDiv_Mul_Mul_Mul_Add_RealDiv_Add_Rsqrt_Mul_Mul_Add_Mul_Redu_more_split_7756657581262259816_kernel blocks in (%c1, %c1, %c1) threads in (%c15, %c1, %c1) args(%arg2 : memref<15xf32>, %arg4 : memref<f32>, %alloc : memref<f32>, %alloc_6 : memref<15xf32>)
    memref.copy %alloc_6, %arg8 : memref<15xf32> to memref<15xf32>
    return
  }
  gpu.module @Fused_Mul_Mul_Add_RealDiv_Mul_Mul_Mul_Add_RealDiv_Add_Rsqrt_Mul_Mul_Add_Mul_Redu_more_split_7756657581262259816_kernel {
    gpu.func @Fused_Mul_Mul_Add_RealDiv_Mul_Mul_Mul_Add_RealDiv_Add_Rsqrt_Mul_Mul_Add_Mul_Redu_more_split_7756657581262259816_kernel(%arg0: memref<15xf32>, %arg1: memref<f32>, %arg2: memref<f32>, %arg3: memref<15xf32>) kernel attributes {gpu.known_block_size = array<i32: 15, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
      %0 = gpu.block_id  x
      %1 = gpu.block_id  y
      %2 = gpu.block_id  z
      %3 = gpu.thread_id  x
      %4 = gpu.thread_id  y
      %5 = gpu.thread_id  z
      %6 = gpu.grid_dim  x
      %7 = gpu.grid_dim  y
      %8 = gpu.grid_dim  z
      %9 = gpu.block_dim  x
      %10 = gpu.block_dim  y
      %11 = gpu.block_dim  z
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c15 = arith.constant 15 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.899999976 : f32
      %cst_0 = arith.constant 0.100000024 : f32
      %cst_1 = arith.constant 1.000000e+00 : f32
      %cst_2 = arith.constant 9.990000e-01 : f32
      %cst_3 = arith.constant 9.99987125E-4 : f32
      %cst_4 = arith.constant 0.000000e+00 : f32
      %cst_5 = arith.constant 9.99999997E-7 : f32
      scf.for %arg5 = %c0 to %c1 step %c1 {
        %12 = arith.muli %arg5, %c15 : index
        %13 = arith.muli %3, %c1 : index
        %14 = arith.addi %13, %c0 : index
        %15 = arith.addi %14, %12 : index
        %16 = memref.load %arg0[%15] : memref<15xf32>
        %17 = arith.muli %arg5, %c15 : index
        %18 = arith.addi %14, %17 : index
        %19 = arith.cmpi eq, %18, %c0 : index
        scf.if %19 {
          %22 = memref.load %arg1[] : memref<f32>
          %23 = arith.divf %cst_1, %22 : f32
          memref.store %23, %arg2[] : memref<f32>
        }
        %20 = memref.load %arg2[] : memref<f32>
        %21 = arith.mulf %16, %20 : f32
        memref.store %21, %arg3[%15] : memref<15xf32>
      }
      gpu.return
    }
  }
}


// CHECK-LABEL: module attributes {gpu.container_module} {
// CHECK-NEXT:   func.func @Fused_Mul_Mul_Add_RealDiv_Mul_Mul_Mul_Add_RealDiv_Add_Rsqrt_Mul_Mul_Add_Mul_Redu_more_split_7756657581262259816(%arg0: memref<15xf32>, %arg1: memref<15xf32>, %arg2: memref<15xf32>, %arg3: memref<15xf32>, %arg4: memref<f32>, %arg5: memref<f32>, %arg6: memref<15xf32>, %arg7: memref<f32>, %arg8: memref<15xf32>, %arg9: memref<15xf32>, %arg10: memref<15xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda", scop.ignored, thread_x = 15 : i32} {
// CHECK-NEXT:     %c15 = arith.constant 15 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %cst = arith.constant 0.100000024 : f32
// CHECK-NEXT:     %cst_0 = arith.constant 0.899999976 : f32
// CHECK-NEXT:     %cst_1 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_2 = arith.constant 9.99987125E-4 : f32
// CHECK-NEXT:     %cst_3 = arith.constant 9.990000e-01 : f32
// CHECK-NEXT:     %cst_4 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %cst_5 = arith.constant 9.99999997E-7 : f32
// CHECK-NEXT:     %alloc = memref.alloc() {alignment = 64 : i64} : memref<15xf32>
// CHECK-NEXT:     gpu.launch_func  @Fused_Mul_Mul_Add_RealDiv_Mul_Mul_Mul_Add_RealDiv_Add_Rsqrt_Mul_Mul_Add_Mul_Redu_more_split_7756657581262259816_kernel::@Fused_Mul_Mul_Add_RealDiv_Mul_Mul_Mul_Add_RealDiv_Add_Rsqrt_Mul_Mul_Add_Mul_Redu_more_split_7756657581262259816_kernel blocks in (%c1, %c1, %c1) threads in (%c15, %c1, %c1) args(%arg2 : memref<15xf32>, %arg4 : memref<f32>, %alloc : memref<15xf32>)
// CHECK-NEXT:     memref.copy %alloc, %arg8 : memref<15xf32> to memref<15xf32>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   gpu.module @Fused_Mul_Mul_Add_RealDiv_Mul_Mul_Mul_Add_RealDiv_Add_Rsqrt_Mul_Mul_Add_Mul_Redu_more_split_7756657581262259816_kernel {
// CHECK-NEXT:     gpu.func @Fused_Mul_Mul_Add_RealDiv_Mul_Mul_Mul_Add_RealDiv_Add_Rsqrt_Mul_Mul_Add_Mul_Redu_more_split_7756657581262259816_kernel(%arg0: memref<15xf32>, %arg1: memref<f32>, %arg2: memref<15xf32>) workgroup(%arg3 : memref<f32, #gpu.address_space<workgroup>>) kernel attributes {gpu.known_block_size = array<i32: 15, 1, 1>, gpu.known_grid_size = array<i32: 1, 1, 1>} {
// CHECK-NEXT:       %0 = gpu.block_id  x
// CHECK-NEXT:       %1 = gpu.block_id  y
// CHECK-NEXT:       %2 = gpu.block_id  z
// CHECK-NEXT:       %3 = gpu.thread_id  x
// CHECK-NEXT:       %4 = gpu.thread_id  y
// CHECK-NEXT:       %5 = gpu.thread_id  z
// CHECK-NEXT:       %6 = gpu.grid_dim  x
// CHECK-NEXT:       %7 = gpu.grid_dim  y
// CHECK-NEXT:       %8 = gpu.grid_dim  z
// CHECK-NEXT:       %9 = gpu.block_dim  x
// CHECK-NEXT:       %10 = gpu.block_dim  y
// CHECK-NEXT:       %11 = gpu.block_dim  z
// CHECK-NEXT:       cf.br ^bb1
// CHECK-NEXT:     ^bb1:  // pred: ^bb0
// CHECK-NEXT:       %c15 = arith.constant 15 : index
// CHECK-NEXT:       %c1 = arith.constant 1 : index
// CHECK-NEXT:       %c0 = arith.constant 0 : index
// CHECK-NEXT:       %cst = arith.constant 0.899999976 : f32
// CHECK-NEXT:       %cst_0 = arith.constant 0.100000024 : f32
// CHECK-NEXT:       %cst_1 = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:       %cst_2 = arith.constant 9.990000e-01 : f32
// CHECK-NEXT:       %cst_3 = arith.constant 9.99987125E-4 : f32
// CHECK-NEXT:       %cst_4 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %cst_5 = arith.constant 9.99999997E-7 : f32
// CHECK-NEXT:       scf.for %arg4 = %c0 to %c1 step %c1 {
// CHECK-NEXT:         %12 = arith.muli %arg4, %c15 : index
// CHECK-NEXT:         %13 = arith.muli %3, %c1 : index
// CHECK-NEXT:         %14 = arith.addi %13, %c0 : index
// CHECK-NEXT:         %15 = arith.addi %14, %12 : index
// CHECK-NEXT:         %16 = memref.load %arg0[%15] : memref<15xf32>
// CHECK-NEXT:         %17 = arith.muli %arg4, %c15 : index
// CHECK-NEXT:         %18 = arith.addi %14, %17 : index
// CHECK-NEXT:         %19 = arith.cmpi eq, %18, %c0 : index
// CHECK-NEXT:         scf.if %19 {
// CHECK-NEXT:           %22 = memref.load %arg1[] : memref<f32>
// CHECK-NEXT:           %23 = arith.divf %cst_1, %22 : f32
// CHECK-NEXT:           memref.store %23, %arg3[] : memref<f32, #gpu.address_space<workgroup>>
// CHECK-NEXT:         }
// CHECK-NEXT:         %20 = memref.load %arg3[] : memref<f32, #gpu.address_space<workgroup>>
// CHECK-NEXT:         %21 = arith.mulf %16, %20 : f32
// CHECK-NEXT:         memref.store %21, %arg2[%15] : memref<15xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:       gpu.return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
