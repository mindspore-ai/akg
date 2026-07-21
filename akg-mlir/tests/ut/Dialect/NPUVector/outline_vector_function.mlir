// RUN: akg-opt %s -split-input-file --outline-vector-function | FileCheck %s

// Verify outline-vector-function:
//   1. Outlines npuvector compute groups into private vf functions.
//   2. Promotes GM memref args to UB buffers (GM->UB before call, UB->GM after).
//   3. UB inner-most dim is aligned to arch vector width (f32 case 1: 64,
//      case 2: 80 -> 128).
//   4. UB allocations are hoisted to the start of the parent function body,
//      so each buffer is allocated once per kernel invocation rather than
//      once per loop iteration.
//   5. vf functions whose result is an npuvector value have the return
//      dropped: the value is routed through an extra UB memref out-param
//      written via npuvector.transfer_write inside the kernel and read back
//      via npuvector.transfer_read at the call site.

// CHECK-LABEL: func.func @Fused_Add_fusion_8550873487731602555
// All UB allocs appear at the very start of the parent func body, before
// any constants / scf.for.
// CHECK: %[[ALLOC_OUT:[a-zA-Z0-9_]+]] = memref.alloc() : memref<64xf32>
// CHECK-NEXT: %[[ALLOC_IN1:[a-zA-Z0-9_]+]] = memref.alloc() : memref<64xf32>
// CHECK-NEXT: %[[ALLOC_IN0:[a-zA-Z0-9_]+]] = memref.alloc() : memref<64xf32>
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK-NOT: arith.addf
// CHECK: %[[RD0:[a-zA-Z0-9_]+]] = npuvector.transfer_read %arg0
// CHECK-SAME: memref<3072xf32>, !npuvector<?xf32>
// CHECK: %[[SUB0:[a-zA-Z0-9_]+]] = memref.subview %[[ALLOC_IN0]]
// CHECK: npuvector.transfer_write %[[RD0]], %[[SUB0]]
// CHECK-SAME: !npuvector<?xf32>, memref<?xf32, strided<[1], offset: ?>>
// CHECK: %[[RD1:[a-zA-Z0-9_]+]] = npuvector.transfer_read %arg1
// CHECK-SAME: memref<3072xf32>, !npuvector<?xf32>
// CHECK: %[[SUB1:[a-zA-Z0-9_]+]] = memref.subview %[[ALLOC_IN1]]
// CHECK: npuvector.transfer_write %[[RD1]], %[[SUB1]]
// CHECK-SAME: !npuvector<?xf32>, memref<?xf32, strided<[1], offset: ?>>
// CHECK: func.call @Fused_Add_fusion_8550873487731602555_outlined_vf_0(%[[ALLOC_IN0]], %{{.*}}, %[[ALLOC_IN1]], %[[ALLOC_OUT]]) {hivm.vector_function, no_inline} : (memref<64xf32>, index, memref<64xf32>, memref<64xf32>) -> ()
// CHECK: %[[SUB2:[a-zA-Z0-9_]+]] = memref.subview %[[ALLOC_OUT]]
// CHECK: %[[RD2:[a-zA-Z0-9_]+]] = npuvector.transfer_read %[[SUB2]]
// CHECK-SAME: memref<?xf32, strided<[1], offset: ?>>, !npuvector<?xf32>
// CHECK: npuvector.transfer_write %[[RD2]], %arg2
// CHECK-SAME: !npuvector<?xf32>, memref<3072xf32>

// CHECK-LABEL: func.func private @Fused_Add_fusion_8550873487731602555_outlined_vf_0
// CHECK-SAME: memref<64xf32>
// CHECK-SAME: index
// CHECK-SAME: memref<64xf32>
// CHECK-SAME: memref<64xf32>
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: arith.constant 64 : index
// CHECK: arith.constant 64 : index
// CHECK: npuvector.transfer_read %arg0[%c0] [%arg1] [%{{.*}}]
// CHECK-SAME: memref<64xf32>, !npuvector<?xf32>
// CHECK: arith.constant 64 : index
// CHECK: npuvector.transfer_read %arg2[%c0] [%arg1] [%{{.*}}]
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
// The UB allocations are still hoisted to the start of the parent function.

// CHECK-LABEL: func.func @Fused_Add_fusion_8550873487731000000
// CHECK: %[[ALLOC_OUT:[a-zA-Z0-9_]+]] = memref.alloc() : memref<128xf32>
// CHECK-NEXT: %[[ALLOC_IN1:[a-zA-Z0-9_]+]] = memref.alloc() : memref<128xf32>
// CHECK-NEXT: %[[ALLOC_IN0:[a-zA-Z0-9_]+]] = memref.alloc() : memref<128xf32>
// CHECK-NOT: memref.alloc
// CHECK: scf.for
// CHECK-NOT: arith.addf
// CHECK: %[[RD0:[a-zA-Z0-9_]+]] = npuvector.transfer_read %arg0
// CHECK-SAME: memref<4000xf32>, !npuvector<?xf32>
// CHECK: %[[SUB0:[a-zA-Z0-9_]+]] = memref.subview %[[ALLOC_IN0]]
// CHECK: npuvector.transfer_write %[[RD0]], %[[SUB0]]
// CHECK-SAME: !npuvector<?xf32>, memref<?xf32, strided<[1], offset: ?>>
// CHECK: %[[RD1:[a-zA-Z0-9_]+]] = npuvector.transfer_read %arg1
// CHECK-SAME: memref<4000xf32>, !npuvector<?xf32>
// CHECK: %[[SUB1:[a-zA-Z0-9_]+]] = memref.subview %[[ALLOC_IN1]]
// CHECK: npuvector.transfer_write %[[RD1]], %[[SUB1]]
// CHECK-SAME: !npuvector<?xf32>, memref<?xf32, strided<[1], offset: ?>>
// CHECK: func.call @Fused_Add_fusion_8550873487731000000_outlined_vf_0(%[[ALLOC_IN0]], %{{.*}}, %[[ALLOC_IN1]], %[[ALLOC_OUT]]) {hivm.vector_function, no_inline} : (memref<128xf32>, index, memref<128xf32>, memref<128xf32>) -> ()
// CHECK: %[[SUB2:[a-zA-Z0-9_]+]] = memref.subview %[[ALLOC_OUT]]
// CHECK: %[[RD2:[a-zA-Z0-9_]+]] = npuvector.transfer_read %[[SUB2]]
// CHECK-SAME: memref<?xf32, strided<[1], offset: ?>>, !npuvector<?xf32>
// CHECK: npuvector.transfer_write %[[RD2]], %arg2
// CHECK-SAME: !npuvector<?xf32>, memref<4000xf32>

// CHECK-LABEL: func.func private @Fused_Add_fusion_8550873487731000000_outlined_vf_0
// CHECK-SAME: memref<128xf32>
// CHECK-SAME: index
// CHECK-SAME: memref<128xf32>
// CHECK-SAME: memref<128xf32>
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: npuvector.transfer_read %arg0[%c0] [%arg1] [%{{.*}}]
// CHECK-SAME: memref<128xf32>, !npuvector<?xf32>
// CHECK: npuvector.transfer_read %arg2[%c0] [%arg1] [%{{.*}}]
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

// -----

// Verify vf-returning-npuvector handling: the scalar broadcast prep before
// the scf.for is extracted into `_outlined_vf_1`, which originally returned
// an `!npuvector<1xf32>` value. The pass drops that return and routes the
// value through an extra UB `memref<64xf32>` out-param. That same buffer is
// later reused as the scalar broadcast input for `_outlined_vf_0` inside the
// loop, eliminating an extra read/write round-trip.
//
// Five UB allocations (one promoted buffer per GM memref + the shared return-
// value/broadcast out-param) are hoisted to the start of the parent function.

// CHECK-LABEL: func.func @Fused_Mul_IsFinite_split_15301331195131479419
// CHECK: %[[VF1_INPUT_UB:[a-zA-Z0-9_]+]] = memref.alloc() : memref<128xbf16>
// CHECK-NEXT: %[[VF0_I8_OUT_UB:[a-zA-Z0-9_]+]] = memref.alloc() : memref<256xi8>
// CHECK-NEXT: %[[VF0_BF16_OUT_UB:[a-zA-Z0-9_]+]] = memref.alloc() : memref<128xbf16>
// CHECK-NEXT: %[[VF0_BF16_IN_UB:[a-zA-Z0-9_]+]] = memref.alloc() : memref<128xbf16>
// CHECK-NEXT: %[[SCALAR_RET_UB:[a-zA-Z0-9_]+]] = memref.alloc() : memref<64xf32>
// CHECK-NOT: memref.alloc

// Stage scalar into UB and call vf_1 with the UB input and the UB return-value
// out-param. No npuvector result is produced.
// CHECK: %[[VF1_SCALAR_RD:[a-zA-Z0-9_]+]] = npuvector.transfer_read %collapse_shape
// CHECK-SAME: memref<bf16>, !npuvector<1xbf16>
// CHECK: %[[SUB_VF1_IN:[a-zA-Z0-9_]+]] = memref.subview %[[VF1_INPUT_UB]]
// CHECK: npuvector.transfer_write %[[VF1_SCALAR_RD]], %[[SUB_VF1_IN]]
// CHECK-SAME: !npuvector<1xbf16>, memref<1xbf16, strided<[1], offset: ?>>
// CHECK: call @Fused_Mul_IsFinite_split_15301331195131479419_outlined_vf_1(%[[VF1_INPUT_UB]], %[[SCALAR_RET_UB]]) {hivm.vector_function, no_inline} : (memref<128xbf16>, memref<64xf32>) -> ()

// Inside the scf.for, vf_0 reads the scalar directly from %[[SCALAR_RET_UB]],
// which already holds the result from vf_1.
// CHECK: scf.for
// CHECK: %[[RD_ARG0:[a-zA-Z0-9_]+]] = npuvector.transfer_read %arg0
// CHECK-SAME: memref<3072xbf16>, !npuvector<?xbf16>
// CHECK: %[[SUB_VF0_IN:[a-zA-Z0-9_]+]] = memref.subview %[[VF0_BF16_IN_UB]]
// CHECK: npuvector.transfer_write %[[RD_ARG0]], %[[SUB_VF0_IN]]
// CHECK-SAME: !npuvector<?xbf16>, memref<?xbf16, strided<[1], offset: ?>>
// CHECK: func.call @Fused_Mul_IsFinite_split_15301331195131479419_outlined_vf_0(%[[VF0_BF16_IN_UB]], %{{.*}}, %[[SCALAR_RET_UB]], %[[VF0_BF16_OUT_UB]], %[[VF0_I8_OUT_UB]]) {hivm.vector_function, no_inline} : (memref<128xbf16>, index, memref<64xf32>, memref<128xbf16>, memref<256xi8>) -> ()
// CHECK: %[[SUB_I8_OUT:[a-zA-Z0-9_]+]] = memref.subview %[[VF0_I8_OUT_UB]]
// CHECK: %[[RD_I8:[a-zA-Z0-9_]+]] = npuvector.transfer_read %[[SUB_I8_OUT]]
// CHECK-SAME: memref<?xi8, strided<[1], offset: ?>>, !npuvector<?xi8>
// CHECK: npuvector.transfer_write %[[RD_I8]], %arg3
// CHECK-SAME: !npuvector<?xi8>, memref<3072xi8>
// CHECK: %[[SUB_BF16_OUT:[a-zA-Z0-9_]+]] = memref.subview %[[VF0_BF16_OUT_UB]]
// CHECK: %[[RD_BF16:[a-zA-Z0-9_]+]] = npuvector.transfer_read %[[SUB_BF16_OUT]]
// CHECK-SAME: memref<?xbf16, strided<[1], offset: ?>>, !npuvector<?xbf16>
// CHECK: npuvector.transfer_write %[[RD_BF16]], %arg2
// CHECK-SAME: !npuvector<?xbf16>, memref<3072xbf16>

// vf_0 takes UB memrefs for all inputs/outputs.
// CHECK-LABEL: func.func private @Fused_Mul_IsFinite_split_15301331195131479419_outlined_vf_0
// CHECK-SAME: (%arg0: memref<128xbf16>, %arg1: index, %arg2: memref<64xf32>, %arg3: memref<128xbf16>, %arg4: memref<256xi8>)
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: %[[SCALAR_RD:.*]] = npuvector.transfer_read %arg2[%c0] [%c1] [%c64]
// CHECK-SAME: memref<64xf32>, !npuvector<1xf32>
// CHECK: %[[VEC_RD:.*]] = npuvector.transfer_read %arg0[%c0_0] [%arg1] [%c128]
// CHECK-SAME: memref<128xbf16>, !npuvector<?xbf16>
// CHECK: npuvector.extf %[[VEC_RD]]
// CHECK: npuvector.broadcast %[[SCALAR_RD]]
// CHECK: arith.mulf
// CHECK: npuvector.transfer_write %{{.*}}, %arg3[%c0_0] : !npuvector<?xbf16>, memref<128xbf16>
// CHECK: npuvector.transfer_write %{{.*}}, %arg4[%c0_0] : !npuvector<?xi8>, memref<256xi8>
// CHECK: return

// vf_1: return dropped, replaced by write to out-param memref<64xf32>.
// CHECK-LABEL: func.func private @Fused_Mul_IsFinite_split_15301331195131479419_outlined_vf_1
// CHECK-SAME: (%arg0: memref<128xbf16>, %arg1: memref<64xf32>)
// CHECK-SAME: attributes {hivm.vector_function, no_inline}
// CHECK: %[[VF1_RD:.*]] = npuvector.transfer_read %arg0
// CHECK-SAME: memref<128xbf16>, !npuvector<1xbf16>
// CHECK: %[[VF1_EXTF:.*]] = npuvector.extf %[[VF1_RD]]
// CHECK-SAME: !npuvector<1xbf16> to !npuvector<1xf32>
// CHECK: npuvector.transfer_write %[[VF1_EXTF]], %arg1
// CHECK-SAME: !npuvector<1xf32>, memref<64xf32>
// CHECK: return
// CHECK-NOT: return %

#map = affine_map<(d0) -> (3072, d0 * 48)>
#map1 = affine_map<(d0) -> (3072, d0 * 48 + 48)>
module {
  func.func @Fused_Mul_IsFinite_split_15301331195131479419(%arg0: memref<3072xbf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1xbf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<3072xbf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, %arg3: memref<3072xi8> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}) attributes {OperatorType = "Elementwise", arch = "Ascend950PR_9599", compute_capability = "", enable_auto_mark_buffer_size, hacc.block_dim = 64 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mindspore_kernel, process = "aicore"} {
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32767_i16 = arith.constant 32767 : i16
    %c255_i16 = arith.constant 255 : i16
    %c32512_i16 = arith.constant 32512 : i16
    %c0_i16 = arith.constant 0 : i16
    %true = arith.constant true
    %collapse_shape = memref.collapse_shape %arg1 [] : memref<1xbf16> into memref<bf16>
    %c1_0 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c1_1 = arith.constant 1 : index
    %0 = npuvector.transfer_read %collapse_shape[] [%c1_1] [%c1_0], %cst : memref<bf16>, !npuvector<1xbf16>
    %1 = npuvector.extf %0 : !npuvector<1xbf16> to !npuvector<1xf32>
    scf.for %arg4 = %c0 to %c64 step %c1 {
      %2 = affine.min #map(%arg4)
      %3 = affine.min #map1(%arg4)
      %c48 = arith.constant 48 : index
      %4 = arith.subi %3, %2 : index
      %5 = arith.minsi %4, %c48 : index
      %cst_2 = arith.constant 0.000000e+00 : bf16
      %6 = npuvector.transfer_read %arg0[%2] [%5] [%c48], %cst_2 : memref<3072xbf16>, !npuvector<?xbf16>
      %7 = npuvector.extf %6 : !npuvector<?xbf16> to !npuvector<?xf32>
      %8 = npuvector.broadcast %1[%5] [%c48] : !npuvector<1xf32> to !npuvector<?xf32>
      %9 = arith.mulf %7, %8 : !npuvector<?xf32>
      %10 = npuvector.truncf %9 : !npuvector<?xf32> to !npuvector<?xbf16>
      npuvector.transfer_write %10, %arg2[%2] : !npuvector<?xbf16>, memref<3072xbf16>
      %11 = npuvector.truncf %9 : !npuvector<?xf32> to !npuvector<?xbf16>
      %12 = npuvector.bitcast %11 : !npuvector<?xbf16> to !npuvector<?xi16>
      %13 = npuvector.broadcast %c32767_i16[%5] [%c48] : i16 to !npuvector<?xi16>
      %14 = arith.andi %12, %13 : !npuvector<?xi16>
      %15 = npuvector.broadcast %c255_i16[%5] [%c48] : i16 to !npuvector<?xi16>
      %16 = arith.andi %14, %15 : !npuvector<?xi16>
      %17 = npuvector.broadcast %c32512_i16[%5] [%c48] : i16 to !npuvector<?xi16>
      %18 = arith.andi %14, %17 : !npuvector<?xi16>
      %19 = npuvector.broadcast %c32512_i16[%5] [%c48] : i16 to !npuvector<?xi16>
      %20 = npuvector.cmpi eq, %18, %19 : !npuvector<?xi16> to !npuvector<?xi1>
      %21 = npuvector.broadcast %c0_i16[%5] [%c48] : i16 to !npuvector<?xi16>
      %22 = npuvector.cmpi eq, %16, %21 : !npuvector<?xi16> to !npuvector<?xi1>
      %23 = arith.andi %20, %22 : !npuvector<?xi1>
      %24 = npuvector.broadcast %c0_i16[%5] [%c48] : i16 to !npuvector<?xi16>
      %25 = npuvector.cmpi ne, %16, %24 : !npuvector<?xi16> to !npuvector<?xi1>
      %26 = arith.andi %20, %25 : !npuvector<?xi1>
      %27 = arith.ori %23, %26 : !npuvector<?xi1>
      %28 = npuvector.broadcast %true[%5] [%c48] : i1 to !npuvector<?xi1>
      %29 = arith.xori %27, %28 : !npuvector<?xi1>
      %30 = npuvector.uitofp %29 : !npuvector<?xi1> to !npuvector<?xf16>
      %31 = npuvector.fptoui %30 : !npuvector<?xf16> to !npuvector<?xi8>
      npuvector.transfer_write %31, %arg3[%2] : !npuvector<?xi8>, memref<3072xi8>
    } {map_for_to_forall}
    return
  }
}
