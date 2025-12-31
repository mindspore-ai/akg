// RUN: akg-opt %s --npu-auto-tiling | FileCheck %s

// CHECK-LABEL: module {
// CHECK-NEXT:  func.func @test_loop_tiling_static_get_tiling_struct_size_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<get_tiling_struct_size_function>} {
// CHECK-NEXT:    %c64_i64 = arith.constant 64 : i64
// CHECK-NEXT:    return %c64_i64 : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @test_loop_tiling_static_00_tiling_function(%arg0: memref<1024x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1024x1024xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, %arg2: !llvm.ptr {hacc.arg_type = #hacc.arg_type<tiling_key>}, %arg3: memref<64xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK-NEXT:    %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:    llvm.store %c0_i64, %arg2 : i64, !llvm.ptr
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c512_i64 = arith.constant 512 : i64
// CHECK-NEXT:    memref.store %c512_i64, %arg3[%c0] : memref<64xi64>
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c512_i64_0 = arith.constant 512 : i64
// CHECK-NEXT:    memref.store %c512_i64_0, %arg3[%c1] : memref<64xi64>
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c512_i64_1 = arith.constant 512 : i64
// CHECK-NEXT:    memref.store %c512_i64_1, %arg3[%c2] : memref<64xi64>
// CHECK-NEXT:    %c3 = arith.constant 3 : index
// CHECK-NEXT:    %c512_i64_2 = arith.constant 512 : i64
// CHECK-NEXT:    memref.store %c512_i64_2, %arg3[%c3] : memref<64xi64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @test_loop_tiling_static_tiling_function(%arg0: memref<1024x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1024x1024xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, %arg2: !llvm.ptr {hacc.arg_type = #hacc.arg_type<tiling_key>}, %arg3: memref<64xi64> {hacc.arg_type = #hacc.arg_type<tiling_struct>}) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
// CHECK-NEXT:    call @test_loop_tiling_static_00_tiling_function(%arg0, %arg1, %arg2, %arg3) : (memref<1024x1024xf32>, memref<1024x1024xf32>, !llvm.ptr, memref<64xi64>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @test_loop_tiling_static_00(%arg0: memref<1024x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1024x1024xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> memref<1024x1024xf32> attributes {OperatorType = "Elementwise", compute_capability = "", enable_auto_mark_buffer_size, hacc.block_dim = 40 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@test_loop_tiling_static_00_tiling_function>, hfusion.fusion_kind = "PURE_ELEMWISE", process = "aicore"} {
// CHECK-NEXT:    %c512 = arith.constant 512 : index
// CHECK-NEXT:    %c512_0 = arith.constant 512 : index
// CHECK-NEXT:    %c512_1 = arith.constant 512 : index
// CHECK-NEXT:    %c512_2 = arith.constant 512 : index
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1024 = arith.constant 1024 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_3 = arith.constant 0 : index
// CHECK-NEXT:    %c1_4 = arith.constant 1 : index
// CHECK-NEXT:    scf.for %arg2 = %c0 to %c1024 step %c512 {
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c1024 step %c512_0 {
// CHECK-NEXT:        %0 = arith.addi %arg2, %c512 : index
// CHECK-NEXT:        scf.for %arg4 = %arg2 to %0 step %c512_1 {
// CHECK-NEXT:          %1 = arith.addi %arg3, %c512_0 : index
// CHECK-NEXT:          scf.for %arg5 = %arg3 to %1 step %c512_2 {
// CHECK-NEXT:            %2 = arith.addi %arg4, %c512_1 : index
// CHECK-NEXT:            scf.for %arg6 = %arg4 to %2 step %c1_4 {
// CHECK-NEXT:              %3 = arith.addi %arg5, %c512_2 : index
// CHECK-NEXT:              scf.for %arg7 = %arg5 to %3 step %c1_4 {
// CHECK-NEXT:                %4 = memref.load %arg0[%arg6, %arg7] : memref<1024x1024xf32>
// CHECK-NEXT:                %5 = arith.addf %4, %4 : f32
// CHECK-NEXT:                memref.store %5, %arg1[%arg6, %arg7] : memref<1024x1024xf32>
// CHECK-NEXT:              } {vector = 4096 : i64}
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return %arg1 : memref<1024x1024xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @test_loop_tiling_static(%arg0: memref<1024x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1024x1024xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> memref<1024x1024xf32> attributes {OperatorType = "Elementwise", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
// CHECK-NEXT:    %0 = call @test_loop_tiling_static_00(%arg0, %arg1) : (memref<1024x1024xf32>, memref<1024x1024xf32>) -> memref<1024x1024xf32>
// CHECK-NEXT:    return %0 : memref<1024x1024xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:}

func.func @test_loop_tiling_static(
    %arg0: memref<1024x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>},
    %arg1: memref<1024x1024xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> memref<1024x1024xf32> 
    attributes {
      OperatorType = "Elementwise",
      compute_capability = "",
      hacc.function_kind = #hacc.function_kind<HOST>,
      mindspore_kernel,
      process = "aicore"
    } {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1024 step %c1 {
    scf.for %j = %c0 to %c1024 step %c1 {
      %val = memref.load %arg0[%i, %j] : memref<1024x1024xf32>
      %result = arith.addf %val, %val : f32
      memref.store %result, %arg1[%i, %j] : memref<1024x1024xf32>
    }
  }
  return %arg1 : memref<1024x1024xf32>
}
