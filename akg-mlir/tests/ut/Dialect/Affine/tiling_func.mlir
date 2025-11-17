// RUN: akg-opt %s --tiling-func | FileCheck %s

// CHECK-LABEL: #map = affine_map<(d0) -> (d0)>
// CHECK-NEXT:  #map1 = affine_map<(d0) -> (d0 + 512)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    func.func @Fused_Add_fusion_3324123131231234556_single_outlined_0_0_get_tiling_struct_size_function() -> i64 attributes {hacc.function_kind = "HOST"} {
// CHECK-NEXT:      %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:      return %c0_i64 : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @Fused_Add_fusion_3324123131231234556_single_outlined_0_0_tiling_function(%arg0: memref<171520xf32> {hacc.arg_type = "input", hacc.input_idx = 0 : i64}, %arg1: memref<171520xf32> {hacc.arg_type = "input", hacc.input_idx = 1 : i64}, %arg2: memref<171520xf32> {hacc.arg_type = "output", hacc.output_idx = 0 : i64}) -> (i64 {hacc.arg_type = "tiling_key"}, i64 {hacc.arg_type = "tiling_data"}, i64 {hacc.arg_type = "tiling_data"}, i64 {hacc.arg_type = "tiling_data"}, i64 {hacc.arg_type = "tiling_data"}) attributes {hacc.function_kind = "HOST", hacc.host_func_type = "tiling_function"} {
// CHECK-NEXT:      %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:      %c12280_i64 = arith.constant 12280 : i64
// CHECK-NEXT:      %c13_i64 = arith.constant 13 : i64
// CHECK-NEXT:      %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:      %c49120_i64 = arith.constant 49120 : i64
// CHECK-NEXT:      return %c0_i64, %c12280_i64, %c13_i64, %c1_i64, %c49120_i64 : i64, i64, i64, i64, i64
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @Fused_Add_fusion_3324123131231234556_single_outlined_0_0_0(%arg0: memref<171520xf32> {hacc.arg_type = "input", hacc.input_idx = 0 : i64}, %arg1: memref<171520xf32> {hacc.arg_type = "input", hacc.input_idx = 1 : i64}, %arg2: memref<171520xf32> {hacc.arg_type = "output", hacc.output_idx = 0 : i64}) attributes {enable_auto_mark_buffer_size, hacc.block_dim = 40 : i64, hacc.entry, hacc.function_kind = "DEVICE", hacc.tiling_function = @Fused_Add_fusion_3324123131231234556_single_outlined_0_0_tiling_function, hfusion.fusion_kind = "PURE_ELEMWISE"} {
// CHECK-NEXT:      %alloc = memref.alloc() {alignment = 64 : i64} : memref<171520xf32>
// CHECK-NEXT:      affine.for %arg3 = 0 to 171520 step 512 {
// CHECK-NEXT:        affine.for %arg4 = #map(%arg3) to #map1(%arg3) step 512 {
// CHECK-NEXT:          %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:          %subview = memref.subview %arg0[0] [512] [1] : memref<171520xf32> to memref<512xf32>
// CHECK-NEXT:          %0 = bufferization.to_tensor %subview restrict writable : memref<512xf32>
// CHECK-NEXT:          %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:          %subview_1 = memref.subview %arg1[0] [512] [1] : memref<171520xf32> to memref<512xf32>
// CHECK-NEXT:          %1 = bufferization.to_tensor %subview_1 restrict writable : memref<512xf32>
// CHECK-NEXT:          %2 = arith.addf %0, %1 : tensor<512xf32>
// CHECK-NEXT:          %subview_2 = memref.subview %alloc[0] [512] [1] : memref<171520xf32> to memref<512xf32>
// CHECK-NEXT:          %3 = bufferization.to_memref %2 : memref<512xf32>
// CHECK-NEXT:          memref.copy %3, %subview_2 : memref<512xf32> to memref<512xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.copy %alloc, %arg2 : memref<171520xf32> to memref<171520xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func @Fused_Add_fusion_3324123131231234556(%arg0: memref<171520xf32> {hacc.arg_type = "input", hacc.input_idx = 0 : i64}, %arg1: memref<171520xf32> {hacc.arg_type = "input", hacc.input_idx = 1 : i64}, %arg2: memref<171520xf32> {hacc.arg_type = "output", hacc.output_idx = 0 : i64}) -> memref<171520xf32> attributes {OperatorType = "Elementwise", compute_capability = "", mindspore_kernel, process = "aicore"} {
// CHECK-NEXT:      call @Fused_Add_fusion_3324123131231234556_single_outlined_0_0_0(%arg0, %arg1, %arg2) : (memref<171520xf32>, memref<171520xf32>, memref<171520xf32>) -> ()
// CHECK-NEXT:      return %arg2 : memref<171520xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 512)>
module {
  func.func @Fused_Add_fusion_3324123131231234556(%arg0: memref<171520xf32>, %arg1: memref<171520xf32>) -> memref<171520xf32> attributes {OperatorType = "Elementwise", compute_capability = "", mindspore_kernel, process = "aicore"} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<171520xf32>
    affine.for %arg2 = 0 to 171520 step 512 {
      affine.for %arg3 = #map(%arg2) to #map1(%arg2) step 512 {
        %cst = arith.constant 0.000000e+00 : f32
        %subview = memref.subview %arg0[0] [512] [1] : memref<171520xf32> to memref<512xf32>
        %0 = bufferization.to_tensor %subview restrict writable : memref<512xf32>
        %cst_0 = arith.constant 0.000000e+00 : f32
        %subview_1 = memref.subview %arg1[0] [512] [1] : memref<171520xf32> to memref<512xf32>
        %1 = bufferization.to_tensor %subview_1 restrict writable : memref<512xf32>
        %2 = arith.addf %0, %1 : tensor<512xf32>
        %subview_2 = memref.subview %alloc[0] [512] [1] : memref<171520xf32> to memref<512xf32>
        %3 = bufferization.to_memref %2 : memref<512xf32>
        memref.copy %3, %subview_2 : memref<512xf32> to memref<512xf32>
      }
    }
    return %alloc : memref<171520xf32>
  }
}

