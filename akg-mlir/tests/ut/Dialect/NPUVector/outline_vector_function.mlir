// RUN: akg-opt %s --outline-vector-function | FileCheck %s

// CHECK-LABEL: #map = affine_map<(d0) -> (3072, d0 * 64)>
// CHECK-NEXT:  #map1 = affine_map<(d0) -> (3072, d0 * 64 + 64)>
// CHECK-NEXT:  module {
// CHECK-NEXT:    func.func @Fused_Add_fusion_8550873487731602555(%arg0: memref<3072xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<3072xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<3072xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) attributes {OperatorType = "Elementwise", arch = "aarch20480", compute_capability = "", enable_auto_mark_buffer_size, hacc.block_dim = 48 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mindspore_kernel, process = "aicore"} {
// CHECK-NEXT:      %c1 = arith.constant 1 : index
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %c48 = arith.constant 48 : index
// CHECK-NEXT:      scf.for %arg3 = %c0 to %c48 step %c1 {
// CHECK-NEXT:        %0 = affine.min #map(%arg3)
// CHECK-NEXT:        %1 = affine.min #map1(%arg3)
// CHECK-NEXT:        %c64 = arith.constant 64 : index
// CHECK-NEXT:        %2 = arith.subi %1, %0 : index
// CHECK-NEXT:        %3 = arith.minsi %2, %c64 : index
// CHECK-NEXT:        %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:        %alloc = memref.alloc() : memref<64xf32>
// CHECK-NEXT:        %c0_0 = arith.constant 0 : index
// CHECK-NEXT:        %cst_1 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:        %c64_2 = arith.constant 64 : index
// CHECK-NEXT:        %4 = npuvector.transfer_read %arg0[%0] [%c64_2] [%c64_2], %cst_1 : memref<3072xf32>, !npuvector<64xf32>
// CHECK-NEXT:        npuvector.transfer_write %4, %alloc[%c0_0] : !npuvector<64xf32>, memref<64xf32>
// CHECK-NEXT:        %alloc_3 = memref.alloc() : memref<64xf32>
// CHECK-NEXT:        %c0_4 = arith.constant 0 : index
// CHECK-NEXT:        %cst_5 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:        %c64_6 = arith.constant 64 : index
// CHECK-NEXT:        %5 = npuvector.transfer_read %arg1[%0] [%c64_6] [%c64_6], %cst_5 : memref<3072xf32>, !npuvector<64xf32>
// CHECK-NEXT:        npuvector.transfer_write %5, %alloc_3[%c0_4] : !npuvector<64xf32>, memref<64xf32>
// CHECK-NEXT:        %alloc_7 = memref.alloc() : memref<64xf32>
// CHECK-NEXT:        %c0_8 = arith.constant 0 : index
// CHECK-NEXT:        func.call @Fused_Add_fusion_8550873487731602555_outlined_vf_0(%alloc, %3, %alloc_3, %alloc_7) {hivm.vector_function, no_inline} : (memref<64xf32>, index, memref<64xf32>, memref<64xf32>) -> ()
// CHECK-NEXT:        %c0_9 = arith.constant 0 : index
// CHECK-NEXT:        %c64_10 = arith.constant 64 : index
// CHECK-NEXT:        %cst_11 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:        %6 = npuvector.transfer_read %alloc_7[%c0_9] [%c64_10] [%c64_10], %cst_11 : memref<64xf32>, !npuvector<64xf32>
// CHECK-NEXT:        npuvector.transfer_write %6, %arg2[%0] : !npuvector<64xf32>, memref<3072xf32>
// CHECK-NEXT:      } {map_for_to_forall}
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:    func.func private @Fused_Add_fusion_8550873487731602555_outlined_vf_0(%arg0: memref<64xf32>, %arg1: index, %arg2: memref<64xf32>, %arg3: memref<64xf32>) attributes {hivm.vector_function, no_inline} {
// CHECK-NEXT:      %c0 = arith.constant 0 : index
// CHECK-NEXT:      %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %c64 = arith.constant 64 : index
// CHECK-NEXT:      %0 = npuvector.transfer_read %arg0[%c0] [%arg1] [%c64], %cst : memref<64xf32>, !npuvector<?xf32>
// CHECK-NEXT:      %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %1 = npuvector.transfer_read %arg2[%c0] [%arg1] [%c64], %cst_0 : memref<64xf32>, !npuvector<?xf32>
// CHECK-NEXT:      %2 = arith.addf %0, %1 : !npuvector<?xf32>
// CHECK-NEXT:      npuvector.transfer_write %2, %arg3[%c0] : !npuvector<?xf32>, memref<64xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }

#map = affine_map<(d0) -> (3072, d0 * 64)>
#map1 = affine_map<(d0) -> (3072, d0 * 64 + 64)>
module {
  func.func @Fused_Add_fusion_8550873487731602555(%arg0: memref<3072xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<3072xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<3072xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) attributes {OperatorType = "Elementwise", arch = "aarch20480", compute_capability = "", enable_auto_mark_buffer_size, hacc.block_dim = 48 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mindspore_kernel, process = "aicore"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c48 = arith.constant 48 : index
    scf.for %arg3 = %c0 to %c48 step %c1 {
      %0 = affine.min #map(%arg3)
      %1 = affine.min #map1(%arg3)
      %c64 = arith.constant 64 : index
      %2 = arith.subi %1, %0 : index
      %3 = arith.minsi %2, %c64 : index
      %cst = arith.constant 0.000000e+00 : f32
      %4 = npuvector.transfer_read %arg0[%0] [%3] [%c64], %cst : memref<3072xf32>, !npuvector<?xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %5 = npuvector.transfer_read %arg1[%0] [%3] [%c64], %cst_0 : memref<3072xf32>, !npuvector<?xf32>
      %6 = arith.addf %4, %5 : !npuvector<?xf32>
      npuvector.transfer_write %6, %arg2[%0] : !npuvector<?xf32>, memref<3072xf32>
    } {map_for_to_forall}
    return
  }
}
