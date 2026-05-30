// RUN: akg-opt %s -split-input-file --outline-vector-function | FileCheck %s

// Verify outline-vector-function:
//   1. Outlines npuvector compute into a private vf function.
//   2. Promotes GM memref args to UB buffers (GM->UB before call, UB->GM after).
//   3. UB inner-most dim is aligned to arch vector width (f32 case 1: 64, case 2: 80 -> 128).

// CHECK-LABEL: func.func @Fused_Add_fusion_8550873487731602555
// CHECK: scf.for
// CHECK-NOT: arith.addf
// CHECK: %[[ALLOC0:[a-zA-Z0-9_]+]] = memref.alloc() : memref<64xf32>
// CHECK: %[[RD0:[a-zA-Z0-9_]+]] = npuvector.transfer_read %arg0
// CHECK-SAME: memref<3072xf32>, !npuvector<64xf32>
// CHECK: npuvector.transfer_write %[[RD0]], %[[ALLOC0]]
// CHECK-SAME: !npuvector<64xf32>, memref<64xf32>
// CHECK: %[[ALLOC1:[a-zA-Z0-9_]+]] = memref.alloc() : memref<64xf32>
// CHECK: %[[RD1:[a-zA-Z0-9_]+]] = npuvector.transfer_read %arg1
// CHECK-SAME: memref<3072xf32>, !npuvector<64xf32>
// CHECK: npuvector.transfer_write %[[RD1]], %[[ALLOC1]]
// CHECK-SAME: !npuvector<64xf32>, memref<64xf32>
// CHECK: %[[ALLOC2:[a-zA-Z0-9_]+]] = memref.alloc() : memref<64xf32>
// CHECK: func.call @Fused_Add_fusion_8550873487731602555_outlined_vf_0(%[[ALLOC0]], %{{.*}}, %[[ALLOC1]], %[[ALLOC2]]) {hivm.vector_function, no_inline} : (memref<64xf32>, index, memref<64xf32>, memref<64xf32>) -> ()
// CHECK: %[[RD2:[a-zA-Z0-9_]+]] = npuvector.transfer_read %[[ALLOC2]]
// CHECK-SAME: memref<64xf32>, !npuvector<64xf32>
// CHECK: npuvector.transfer_write %[[RD2]], %arg2
// CHECK-SAME: !npuvector<64xf32>, memref<3072xf32>

// CHECK-LABEL: func.func private @Fused_Add_fusion_8550873487731602555_outlined_vf_0
// CHECK-SAME: memref<64xf32>
// CHECK-SAME: index
// CHECK-SAME: memref<64xf32>
// CHECK-SAME: memref<64xf32>
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: arith.constant 64 : index
// CHECK: %[[VW0A:[a-zA-Z0-9_]+]] = arith.constant 64 : index
// CHECK: npuvector.transfer_read %arg0[%{{.*}}] [%arg1] [%[[VW0A]]]
// CHECK-SAME: memref<64xf32>, !npuvector<?xf32>
// CHECK: %[[VW0B:[a-zA-Z0-9_]+]] = arith.constant 64 : index
// CHECK: npuvector.transfer_read %arg2[%{{.*}}] [%arg1] [%[[VW0B]]]
// CHECK-SAME: memref<64xf32>, !npuvector<?xf32>
// CHECK: arith.addf
// CHECK-SAME: !npuvector<?xf32>
// CHECK: npuvector.transfer_write %{{.*}}, %arg3
// CHECK-SAME: !npuvector<?xf32>, memref<64xf32>

#map = affine_map<(d0) -> (3072, d0 * 64)>
#map1 = affine_map<(d0) -> (3072, d0 * 64 + 64)>
module {
  func.func @Fused_Add_fusion_8550873487731602555(%arg0: memref<3072xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<3072xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<3072xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) attributes {OperatorType = "Elementwise", arch = "Ascend950PR_9599", compute_capability = "", enable_auto_mark_buffer_size, hacc.block_dim = 48 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mindspore_kernel, process = "aicore"} {
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

// -----

// Verify UB alignment: vector chunk 80 is padded to 128 (256B / 4B per f32),
// and the inner transfer_read width inside the outlined vf must be the aligned 128.

// CHECK-LABEL: func.func @Fused_Add_fusion_8550873487731000000
// CHECK: scf.for
// CHECK-NOT: arith.addf
// CHECK: %[[ALLOC0:[a-zA-Z0-9_]+]] = memref.alloc() : memref<128xf32>
// CHECK: %[[RD0:[a-zA-Z0-9_]+]] = npuvector.transfer_read %arg0
// CHECK-SAME: memref<4000xf32>, !npuvector<80xf32>
// CHECK: npuvector.transfer_write %[[RD0]], %[[ALLOC0]]
// CHECK-SAME: !npuvector<80xf32>, memref<128xf32>
// CHECK: %[[ALLOC1:[a-zA-Z0-9_]+]] = memref.alloc() : memref<128xf32>
// CHECK: %[[RD1:[a-zA-Z0-9_]+]] = npuvector.transfer_read %arg1
// CHECK-SAME: memref<4000xf32>, !npuvector<80xf32>
// CHECK: npuvector.transfer_write %[[RD1]], %[[ALLOC1]]
// CHECK-SAME: !npuvector<80xf32>, memref<128xf32>
// CHECK: %[[ALLOC2:[a-zA-Z0-9_]+]] = memref.alloc() : memref<128xf32>
// CHECK: func.call @Fused_Add_fusion_8550873487731000000_outlined_vf_0(%[[ALLOC0]], %{{.*}}, %[[ALLOC1]], %[[ALLOC2]]) {hivm.vector_function, no_inline} : (memref<128xf32>, index, memref<128xf32>, memref<128xf32>) -> ()
// CHECK: %[[RD2:[a-zA-Z0-9_]+]] = npuvector.transfer_read %[[ALLOC2]]
// CHECK-SAME: memref<128xf32>, !npuvector<80xf32>
// CHECK: npuvector.transfer_write %[[RD2]], %arg2
// CHECK-SAME: !npuvector<80xf32>, memref<4000xf32>

// CHECK-LABEL: func.func private @Fused_Add_fusion_8550873487731000000_outlined_vf_0
// CHECK-SAME: memref<128xf32>
// CHECK-SAME: index
// CHECK-SAME: memref<128xf32>
// CHECK-SAME: memref<128xf32>
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: arith.constant 80 : index
// CHECK: %[[VW1A:[a-zA-Z0-9_]+]] = arith.constant 128 : index
// CHECK: npuvector.transfer_read %arg0[%{{.*}}] [%arg1] [%[[VW1A]]]
// CHECK-SAME: memref<128xf32>, !npuvector<?xf32>
// CHECK: %[[VW1B:[a-zA-Z0-9_]+]] = arith.constant 128 : index
// CHECK: npuvector.transfer_read %arg2[%{{.*}}] [%arg1] [%[[VW1B]]]
// CHECK-SAME: memref<128xf32>, !npuvector<?xf32>
// CHECK: arith.addf
// CHECK-SAME: !npuvector<?xf32>
// CHECK: npuvector.transfer_write %{{.*}}, %arg3
// CHECK-SAME: !npuvector<?xf32>, memref<128xf32>

#map = affine_map<(d0) -> (4000, d0 * 80)>
#map1 = affine_map<(d0) -> (4000, d0 * 80 + 80)>
module {
  func.func @Fused_Add_fusion_8550873487731000000(%arg0: memref<4000xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<4000xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<4000xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) attributes {OperatorType = "Elementwise", arch = "Ascend950PR_9599", compute_capability = "", enable_auto_mark_buffer_size, hacc.block_dim = 50 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mindspore_kernel, process = "aicore"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c50 = arith.constant 50 : index
    scf.for %arg3 = %c0 to %c50 step %c1 {
      %0 = affine.min #map(%arg3)
      %1 = affine.min #map1(%arg3)
      %c80 = arith.constant 80 : index
      %2 = arith.subi %1, %0 : index
      %3 = arith.minsi %2, %c80 : index
      %cst = arith.constant 0.000000e+00 : f32
      %4 = npuvector.transfer_read %arg0[%0] [%3] [%c80], %cst : memref<4000xf32>, !npuvector<?xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %5 = npuvector.transfer_read %arg1[%0] [%3] [%c80], %cst_0 : memref<4000xf32>, !npuvector<?xf32>
      %6 = arith.addf %4, %5 : !npuvector<?xf32>
      npuvector.transfer_write %6, %arg2[%0] : !npuvector<?xf32>, memref<4000xf32>
    } {map_for_to_forall}
    return
  }
}
