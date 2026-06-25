// RUN: akg-opt %s -convert-arith-to-hivm | FileCheck %s

func.func @test_addf(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_addf
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}>) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}>) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %0 = arith.addf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_vadd_npuvector_scalar(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[CST1:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %[[CST1]] : memref<128xf32>, f32) outs(%[[UB_OUT]] : memref<128xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<128xf32>) outs(%[[SV_OUT]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v0 = npuvector.transfer_read %arg0[%c0], %cst : memref<128xf32>, !npuvector<128xf32>
  %b = arith.constant 1.000000e+00 : f32
  %v1 = npuvector.broadcast %b : f32 to !npuvector<128xf32>
  %0 = arith.addf %v0, %v1 : !npuvector<128xf32>
  npuvector.transfer_write %0, %arg1[%c0] : !npuvector<128xf32>, memref<128xf32>
  return
}

// -----

func.func @test_vadd_npuvector_scalar_commute(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[CST1:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %[[CST1]] : memref<128xf32>, f32) outs(%[[UB_OUT]] : memref<128xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<128xf32>) outs(%[[SV_OUT]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v0 = npuvector.transfer_read %arg0[%c0], %cst : memref<128xf32>, !npuvector<128xf32>
  %b = arith.constant 1.000000e+00 : f32
  %v1 = npuvector.broadcast %b : f32 to !npuvector<128xf32>
  %0 = arith.addf %v1, %v0 : !npuvector<128xf32>
  npuvector.transfer_write %0, %arg1[%c0] : !npuvector<128xf32>, memref<128xf32>
  return
}

// -----

func.func @test_vadd_npuvector_scalar_dynamic(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_vadd_npuvector_scalar_dynamic
  // CHECK: %[[CST1:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = memref.dim %arg0, %[[C0]] : memref<?xf32>
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %arg0[0] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[UB0]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<?xf32, strided<[1]>>) outs(%[[UB0]] : memref<?xf32>) init_out_buffer = false
  // CHECK: %[[UB_OUT:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[UB_OUT]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %[[CST1]] : memref<?xf32>, f32) outs(%[[UB_OUT]] : memref<?xf32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %arg1[0] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<?xf32>) outs(%[[SUBVIEW_DST]] : memref<?xf32, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %pad = arith.constant 0.000000e+00 : f32
  %dim = memref.dim %arg0, %c0 : memref<?xf32>
  %v0 = npuvector.transfer_read %arg0[%c0][%dim][%c4096], %pad : memref<?xf32>, !npuvector<?xf32>
  %b = arith.constant 1.000000e+00 : f32
  %v1 = npuvector.broadcast %b[%dim] [%c4096] : f32 to !npuvector<?xf32>
  %0 = arith.addf %v0, %v1 : !npuvector<?xf32>
  npuvector.transfer_write %0, %arg1[%c0] : !npuvector<?xf32>, memref<?xf32>
  return
}

// -----

func.func @test_vdiv_npuvector_scalar_dynamic(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_vdiv_npuvector_scalar_dynamic
  // CHECK: %[[CST8:.*]] = arith.constant 8.000000e+00 : f32
  // CHECK: %[[UB0:.*]] = memref.alloc(%{{.*}}) : memref<?xf32>
  // CHECK: hivm.hir.load
  // CHECK: %[[UB_BRC:.*]] = memref.alloc(%{{.*}}) : memref<?xf32>
  // CHECK: hivm.hir.vbrc ins(%[[CST8]] : f32) outs(%[[UB_BRC]] : memref<?xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc(%{{.*}}) : memref<?xf32>
  // CHECK: hivm.hir.vdiv ins(%[[UB0]], %[[UB_BRC]] : memref<?xf32>, memref<?xf32>) outs(%[[UB_OUT]] : memref<?xf32>)
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %pad = arith.constant 0.000000e+00 : f32
  %dim = memref.dim %arg0, %c0 : memref<?xf32>
  %v0 = npuvector.transfer_read %arg0[%c0][%dim][%c4096], %pad : memref<?xf32>, !npuvector<?xf32>
  %b = arith.constant 8.000000e+00 : f32
  %v1 = npuvector.broadcast %b[%dim] [%c4096] : f32 to !npuvector<?xf32>
  %0 = arith.divf %v0, %v1 : !npuvector<?xf32>
  npuvector.transfer_write %0, %arg1[%c0] : !npuvector<?xf32>, memref<?xf32>
  return
}

// -----

func.func @test_vsub_npuvector_scalar_arg(%arg0 : memref<10xf32>, %arg1 : memref<10xf32>, %arg2 : f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_vsub_npuvector_scalar_arg
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<10xf32>
  // CHECK: hivm.hir.load
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<10xf32>
  // CHECK: hivm.hir.vsub ins(%[[UB0]], %arg2 : memref<10xf32>, f32) outs(%[[UB_OUT]] : memref<10xf32>)
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f32
  %v0 = npuvector.transfer_read %arg0[%c0], %pad : memref<10xf32>, !npuvector<10xf32>
  %v1 = npuvector.broadcast %arg2 : f32 to !npuvector<10xf32>
  %0 = arith.subf %v0, %v1 : !npuvector<10xf32>
  npuvector.transfer_write %0, %arg1[%c0] : !npuvector<10xf32>, memref<10xf32>
  return
}

// -----

func.func @test_vadd_npuvector_vector_broadcast_otf(%arg0 : memref<32x10x10xf32>, %arg1 : memref<10x10xf32>, %arg2 : memref<32x10x10xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_vadd_npuvector_vector_broadcast_otf
  // CHECK: %[[BIG:.*]] = memref.alloc() : memref<32x10x10xf32>
  // CHECK: hivm.hir.load
  // CHECK: %[[SMALL:.*]] = memref.alloc() : memref<10x10xf32>
  // CHECK: hivm.hir.load
  // CHECK: %[[EXPAND:.*]] = memref.expand_shape %[[SMALL]]
  // CHECK-SAME: into memref<1x10x10xf32>
  // CHECK-NOT: hivm.hir.vbrc ins(%[[EXPAND]]
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x10x10xf32>
  // CHECK: hivm.hir.vadd ins(%[[BIG]], %[[EXPAND]] : memref<32x10x10xf32>, memref<1x10x10xf32>) outs(%[[UB_OUT]] : memref<32x10x10xf32>) broadcast = [0]
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c32 = arith.constant 32 : index
  %pad = arith.constant 0.000000e+00 : f32
  %big = npuvector.transfer_read %arg0[%c0, %c0, %c0] [%c32, %c10, %c10] [%c32, %c10, %c10], %pad : memref<32x10x10xf32>, !npuvector<32x10x10xf32>
  %small = npuvector.transfer_read %arg1[%c0, %c0] [%c10, %c10] [%c10, %c10], %pad : memref<10x10xf32>, !npuvector<10x10xf32>
  %brc = npuvector.broadcast %small[%c32, %c10, %c10] [%c32, %c10, %c10] : !npuvector<10x10xf32> to !npuvector<32x10x10xf32> {dimension = array<i64: 1, 2>}
  %0 = arith.addf %big, %brc : !npuvector<32x10x10xf32>
  npuvector.transfer_write %0, %arg2[%c0, %c0, %c0] : !npuvector<32x10x10xf32>, memref<32x10x10xf32>
  return
}

// -----

func.func @test_npuvector_broadcast_fold_lhs_alloc_from_rhs(%arg0 : memref<?x1024xf32>, %arg1 : memref<1024xf32>, %arg2 : memref<?x1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_broadcast_fold_lhs_alloc_from_rhs
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?x1024xf32>, %[[ARG1:.*]]: memref<1024xf32>, %[[ARG2:.*]]: memref<?x1024xf32>)
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?x1024xf32>
  // CHECK: %[[WIDE:.*]] = memref.alloc(%[[DIM]]) : memref<?x1024xf32>
  // CHECK: hivm.hir.load {{.*}} outs(%[[WIDE]] : memref<?x1024xf32>)
  // CHECK: %[[NARROW:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.load {{.*}} outs(%[[NARROW]] : memref<1024xf32>)
  // CHECK: %[[EXPAND:.*]] = memref.expand_shape %[[NARROW]]
  // CHECK-SAME: into memref<1x1024xf32>
  // CHECK: %[[OUT:.*]] = memref.alloc(%[[DIM]]) : memref<?x1024xf32>
  // CHECK: hivm.hir.vmul ins(%[[EXPAND]], %[[WIDE]] : memref<1x1024xf32>, memref<?x1024xf32>) outs(%[[OUT]] : memref<?x1024xf32>) broadcast = [0]
  %c0 = arith.constant 0 : index
  %c9 = arith.constant 9 : index
  %c1024 = arith.constant 1024 : index
  %dim = memref.dim %arg0, %c0 : memref<?x1024xf32>
  %pad = arith.constant 0.000000e+00 : f32
  %wide = npuvector.transfer_read %arg0[%c0, %c0] [%dim, %c1024] [%c9, %c1024], %pad : memref<?x1024xf32>, !npuvector<?x1024xf32>
  %narrow = npuvector.transfer_read %arg1[%c0] [%c1024] [%c1024], %pad : memref<1024xf32>, !npuvector<1024xf32>
  %brc = npuvector.broadcast %narrow[%dim, %c1024] [%c9, %c1024] : !npuvector<1024xf32> to !npuvector<?x1024xf32> {dimension = array<i64: 1>}
  %mul = arith.mulf %brc, %wide : !npuvector<?x1024xf32>
  npuvector.transfer_write %mul, %arg2[%c0, %c0] : !npuvector<?x1024xf32>, memref<?x1024xf32>
  return
}

// -----

func.func @test_npuvector_broadcast_rank_lift_preserves_subview_layout(%arg0 : memref<64xf32>, %out : memref<64x64xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_broadcast_rank_lift_preserves_subview_layout
  // CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<256x1x64xf32>
  // CHECK: %[[SLICE:.*]] = memref.subview %[[SCRATCH]]
  // CHECK-SAME: to memref<64xf32, strided<{{.*}}>
  // CHECK: %[[EXPAND:.*]] = memref.expand_shape %[[SLICE]]
  // CHECK-SAME: : memref<64xf32, strided<{{.*}}> into memref<1x64xf32{{.*}}>
  // CHECK: hivm.hir.vmul
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.000000e+00 : f32
  %scratch = memref.alloc() : memref<256x1x64xf32>
  %lhs0 = npuvector.transfer_read %arg0[%c0] [%c64] [%c64], %pad : memref<64xf32>, !npuvector<64xf32>
  %rhs0 = npuvector.transfer_read %scratch[%c0, %c0, %c0] [%c64] [%c64], %pad : memref<256x1x64xf32>, !npuvector<64xf32>
  %lhs = npuvector.broadcast %lhs0[%c64, %c64] [%c64, %c64] : !npuvector<64xf32> to !npuvector<64x64xf32> {dimension = array<i64: 0>}
  %rhs = npuvector.broadcast %rhs0[%c64, %c64] [%c64, %c64] : !npuvector<64xf32> to !npuvector<64x64xf32> {dimension = array<i64: 1>}
  %mul = arith.mulf %lhs, %rhs : !npuvector<64x64xf32>
  npuvector.transfer_write %mul, %out[%c0, %c0] : !npuvector<64x64xf32>, memref<64x64xf32>
  return
}

// -----

// Dynamic ?x32: vector broadcast must not fold when paired with scalar broadcast,
// so elementwise output alloc can take dynamic dims from the VBrc buffer.
func.func @test_dynamic_dual_broadcast_reduction_addf(%arg0: memref<64x128xf16>, %arg1: memref<32xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_dynamic_dual_broadcast_reduction_addf
  // CHECK: %[[C8:.*]] = arith.constant 8 : index
  // CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK-NOT: hivm.hir.vbrc ins(%[[ZERO]] : f16)
  // CHECK: %[[VAL_BRC:.*]] = memref.alloc(%[[C8]]) : memref<?x32xf16>
  // CHECK: hivm.hir.vbrc
  // CHECK: %[[OUT:.*]] = memref.alloc(%[[C8]]) : memref<?x32xf16>
  // CHECK: hivm.hir.vadd ins(%[[VAL_BRC]], %[[ZERO]] : memref<?x32xf16>, f16) outs(%[[OUT]] : memref<?x32xf16>)
  // CHECK: hivm.hir.vreduce
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %cst_f32 = arith.constant 0.000000e+00 : f32
  %small = npuvector.transfer_read %arg0[%c0, %c0] [%c1] [%c1], %cst : memref<64x128xf16>, !npuvector<1xf16>
  %wide = npuvector.extf %small {round_mode = #hivm.round_mode<round>} : !npuvector<1xf16> to !npuvector<1xf32>
  %val_f16 = npuvector.truncf %wide {round_mode = #hivm.round_mode<round>} : !npuvector<1xf32> to !npuvector<1xf16>
  %val = npuvector.broadcast %val_f16[%c8, %c32] [%c8, %c32] : !npuvector<1xf16> to !npuvector<?x32xf16>
  %init = npuvector.broadcast %cst[%c8, %c32] [%c8, %c32] : f16 to !npuvector<?x32xf16>
  %sum = arith.addf %val, %init {reduction_axes = [1 : index], reduction_type = "x"} : !npuvector<?x32xf16>
  %red = npuvector.reduction <add>, %sum {reduction_dims = array<i64: 1>} : !npuvector<?x32xf16> into !npuvector<?xf16>
  npuvector.transfer_write %red, %arg1[%c0] : !npuvector<?xf16>, memref<32xf16>
  return
}

// CHECK-LABEL: func.func @test_transfer_write_alloc_root_offset_after_truncf
// CHECK-SAME: %{{.*}}: memref<?xf32>, %[[BASE:.*]]: index, %[[SIZE:.*]]: index
func.func @test_transfer_write_alloc_root_offset_after_truncf(%arg0: memref<?xf32>, %arg1: index, %arg2: index) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[DST:.*]] = memref.alloc(%[[SIZE]]) : memref<1x?xbf16>
  // CHECK: %[[ADD:.*]] = arith.addi %[[BASE]], %[[SIZE]] : index
  // CHECK-NEXT: %[[SV:.*]] = memref.subview %[[DST]][0, %[[ADD]]] [1, %[[SIZE]]] [1, 1]
  // CHECK-NEXT: hivm.hir.vcast {{.*}} outs(%[[SV]]
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %pad = arith.constant 0.000000e+00 : f32
  %dst = memref.alloc(%arg2) : memref<1x?xbf16>
  %v0 = npuvector.transfer_read %arg0[%c0][%arg2][%c4096], %pad : memref<?xf32>, !npuvector<?xf32>
  %trunc = npuvector.truncf %v0 : !npuvector<?xf32> to !npuvector<?xbf16>
  %offset = arith.addi %arg1, %arg2 : index
  npuvector.transfer_write %trunc, %dst[%c0, %offset] : !npuvector<?xbf16>, memref<1x?xbf16>
  return
}

// -----

// CHECK-LABEL: func.func @test_transfer_read_mixed_arg_and_alloc
func.func @test_transfer_read_mixed_arg_and_alloc(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[A:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: %[[SV_A:.*]] = memref.subview %[[A]][0] [128] [1]
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0] [128] [1]
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK-NOT: hivm.hir.load
  // CHECK: %{{.*}} = memref.alloc() : memref<128xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %[[SV_A]]
  // CHECK: %{{.*}} = memref.subview %arg1[0] [128] [1]
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<128xf32>) outs(%{{.*}} : memref<128xf32, {{.*}}>)
  // CHECK: return
  %a = memref.alloc() : memref<128xf32>
  %sv_a = memref.subview %a[0] [128] [1] : memref<128xf32> to memref<128xf32, strided<[1]>>
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v0 = npuvector.transfer_read %arg0[%c0], %cst : memref<128xf32>, !npuvector<128xf32>
  %v1 = npuvector.transfer_read %sv_a[%c0], %cst : memref<128xf32, strided<[1]>>, !npuvector<128xf32>
  %0 = arith.addf %v0, %v1 : !npuvector<128xf32>
  npuvector.transfer_write %0, %arg1[%c0] : !npuvector<128xf32>, memref<128xf32>
  return
}

// -----

// npuvector.transpose 3D static: [2,1,0] 8x16x32 -> 32x16x8 (only dim 0 and 2 swap, HIVM supports 2-axis transpose directly)
// CHECK-LABEL: func.func @test_npuvector_transpose_3d_static
func.func @test_npuvector_transpose_3d_static(%arg0 : memref<8x16x32xf32>, %arg1 : memref<32x16x8xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0, 0] [8, 16, 32] [1, 1, 1] : memref<8x16x32xf32> to memref<8x16x32xf32, {{.*}}>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<8x16x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<8x16x32xf32, {{.*}}>) outs(%[[UB0]] : memref<8x16x32xf32>)
  // CHECK: %[[UB_TRANS:.*]] = memref.alloc() : memref<32x16x8xf32>
  // CHECK: hivm.hir.vtranspose ins(%[[UB0]] : memref<8x16x32xf32>) outs(%[[UB_TRANS]] : memref<32x16x8xf32>) permutation = [2, 1, 0]
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0, 0] [32, 16, 8] [1, 1, 1] : memref<32x16x8xf32> to memref<32x16x8xf32, {{.*}}>
  // CHECK: hivm.hir.store ins(%[[UB_TRANS]] : memref<32x16x8xf32>) outs(%[[SV_OUT]] : memref<32x16x8xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v0 = npuvector.transfer_read %arg0[%c0, %c0, %c0], %cst : memref<8x16x32xf32>, !npuvector<8x16x32xf32>
  %trans = npuvector.transpose %v0, [2, 1, 0] : !npuvector<8x16x32xf32> to !npuvector<32x16x8xf32>
  npuvector.transfer_write %trans, %arg1[%c0, %c0, %c0] : !npuvector<32x16x8xf32>, memref<32x16x8xf32>
  return
}

// -----

// npuvector.transpose 3D dynamic: all three dims dynamic ?x?x?, perm [2,1,0] (only dim 0 and 2 swap)
// CHECK-LABEL: func.func @test_npuvector_transpose_3d_dynamic
func.func @test_npuvector_transpose_3d_dynamic(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[DIM0:.*]] = memref.dim %arg0, %[[C0]] : memref<?x?x?xf32>
  // CHECK: %[[DIM1:.*]] = memref.dim %arg0, %[[C1]] : memref<?x?x?xf32>
  // CHECK: %[[DIM2:.*]] = memref.dim %arg0, %[[C2]] : memref<?x?x?xf32>
  // CHECK: hivm.hir.vtranspose {{.*}} permutation = [2, 1, 0]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4096 = arith.constant 4096 : index
  %cst = arith.constant 0.000000e+00 : f32
  %dim0 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
  %dim1 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
  %dim2 = memref.dim %arg0, %c2 : memref<?x?x?xf32>
  %v0 = npuvector.transfer_read %arg0[%c0, %c0, %c0][%dim0, %dim1, %dim2][%c4096, %c4096, %c4096], %cst : memref<?x?x?xf32>, !npuvector<?x?x?xf32>
  %trans = npuvector.transpose %v0, [2, 1, 0] : !npuvector<?x?x?xf32> to !npuvector<?x?x?xf32>
  npuvector.transfer_write %trans, %arg1[%c0, %c0, %c0] : !npuvector<?x?x?xf32>, memref<?x?x?xf32>
  return
}

// -----

// npuvector.transpose 3D [1,2,0]: all 3 axes participate, must decompose to 2 adjacent 2-axis vtranspose
// CHECK-LABEL: func.func @test_npuvector_transpose_3d_121
func.func @test_npuvector_transpose_3d_121(%arg0 : memref<8x16x32xf32>, %arg1 : memref<8x32x16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: hivm.hir.vtranspose {{.*}} permutation = [1, 0, 2]
  // CHECK: hivm.hir.vtranspose {{.*}} permutation = [0, 2, 1]
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v0 = npuvector.transfer_read %arg0[%c0, %c0, %c0], %cst : memref<8x16x32xf32>, !npuvector<8x16x32xf32>
  %trans = npuvector.transpose %v0, [1, 2, 0] : !npuvector<8x16x32xf32> to !npuvector<8x32x16xf32>
  npuvector.transfer_write %trans, %arg1[%c0, %c0, %c0] : !npuvector<8x32x16xf32>, memref<8x32x16xf32>
  return
}

// -----

func.func @test_float_binary_arith_ops(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_float_binary_arith_ops
  // CHECK: hivm.hir.vmul ins({{.*}} : memref<32x32xf32>, memref<32x32xf32>) outs({{.*}} : memref<32x32xf32>)
  // CHECK: hivm.hir.vsub ins({{.*}} : memref<32x32xf32>, memref<32x32xf32>) outs({{.*}} : memref<32x32xf32>)
  // CHECK: hivm.hir.vdiv ins({{.*}} : memref<32x32xf32>, memref<32x32xf32>) outs({{.*}} : memref<32x32xf32>)
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %mul = arith.mulf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %mul, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %sub = arith.subf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %sub, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %div = arith.divf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %div, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_int_binary_arith_ops(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>, %arg3 : memref<32x32xi64>, %arg4 : memref<32x32xi64>, %arg5 : memref<32x32xi64>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_int_binary_arith_ops
  // CHECK: hivm.hir.vadd ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vmul ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vsub ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vmax ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vmax ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vmin ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vmin ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vdiv ins({{.*}} : memref<32x32xi64>, memref<32x32xi64>) outs({{.*}} : memref<32x32xi64>)
  // CHECK: hivm.hir.vdiv ins({{.*}} : memref<32x32xi64>, memref<32x32xi64>) outs({{.*}} : memref<32x32xi64>)
  %c0 = arith.constant 0 : index
  %cst_i32 = arith.constant 0 : i32
  %i0 = vector.transfer_read %arg0[%c0, %c0], %cst_i32 : memref<32x32xi32>, vector<32x32xi32>
  %i1 = vector.transfer_read %arg1[%c0, %c0], %cst_i32 : memref<32x32xi32>, vector<32x32xi32>
  %add = arith.addi %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %add, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %mul = arith.muli %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %mul, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %sub = arith.subi %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %sub, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %maxs = arith.maxsi %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %maxs, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %maxu = arith.maxui %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %maxu, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %mins = arith.minsi %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %mins, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %minu = arith.minui %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %minu, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %cst_i64 = arith.constant 0 : i64
  %l0 = vector.transfer_read %arg3[%c0, %c0], %cst_i64 : memref<32x32xi64>, vector<32x32xi64>
  %l1 = vector.transfer_read %arg4[%c0, %c0], %cst_i64 : memref<32x32xi64>, vector<32x32xi64>
  %divs = arith.divsi %l0, %l1 : vector<32x32xi64>
  vector.transfer_write %divs, %arg5[%c0, %c0] : vector<32x32xi64>, memref<32x32xi64>
  %divu = arith.divui %l0, %l1 : vector<32x32xi64>
  vector.transfer_write %divu, %arg5[%c0, %c0] : vector<32x32xi64>, memref<32x32xi64>
  return
}

// -----

func.func @test_float_cast_ops(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf16>, %arg2 : memref<32x32xbf16>, %arg3 : memref<32x32xf16>, %arg4 : memref<32x32xbf16>, %arg5 : memref<32x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_float_cast_ops
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xf32>) outs({{.*}} : memref<32x32xf16>) round_mode = <ceil>
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xf32>) outs({{.*}} : memref<32x32xbf16>)
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xf16>) outs({{.*}} : memref<32x32xf32>) round_mode = <floor>
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xbf16>) outs({{.*}} : memref<32x32xf32>)
  %c0 = arith.constant 0 : index
  %cst_f32 = arith.constant 0.0 : f32
  %vf32 = vector.transfer_read %arg0[%c0, %c0], %cst_f32 : memref<32x32xf32>, vector<32x32xf32>
  %trunc_f16 = arith.truncf %vf32 {round_mode = #hivm.round_mode<ceil>} : vector<32x32xf32> to vector<32x32xf16>
  vector.transfer_write %trunc_f16, %arg3[%c0, %c0] : vector<32x32xf16>, memref<32x32xf16>
  %trunc_bf16 = arith.truncf %vf32 : vector<32x32xf32> to vector<32x32xbf16>
  vector.transfer_write %trunc_bf16, %arg4[%c0, %c0] : vector<32x32xbf16>, memref<32x32xbf16>
  %cst_f16 = arith.constant 0.0 : f16
  %vf16 = vector.transfer_read %arg1[%c0, %c0], %cst_f16 : memref<32x32xf16>, vector<32x32xf16>
  %ext_f16 = arith.extf %vf16 {round_mode = #hivm.round_mode<floor>} : vector<32x32xf16> to vector<32x32xf32>
  vector.transfer_write %ext_f16, %arg5[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %cst_bf16 = arith.constant 0.0 : bf16
  %vbf16 = vector.transfer_read %arg2[%c0, %c0], %cst_bf16 : memref<32x32xbf16>, vector<32x32xbf16>
  %ext_bf16 = arith.extf %vbf16 : vector<32x32xbf16> to vector<32x32xf32>
  vector.transfer_write %ext_bf16, %arg5[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_int_cast_ops(%arg0 : memref<32x32xi8>, %arg1 : memref<32x32xi16>, %arg2 : memref<32x32xi64>, %arg3 : memref<32x32xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_int_cast_ops
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xi8>) outs({{.*}} : memref<32x32xi16>)
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xi64>) outs({{.*}} : memref<32x32xi32>) round_mode = <truncwithoverflow>
  %c0 = arith.constant 0 : index
  %cst_i8 = arith.constant 0 : i8
  %vi8 = vector.transfer_read %arg0[%c0, %c0], %cst_i8 : memref<32x32xi8>, vector<32x32xi8>
  %extui = arith.extui %vi8 : vector<32x32xi8> to vector<32x32xi16>
  vector.transfer_write %extui, %arg1[%c0, %c0] : vector<32x32xi16>, memref<32x32xi16>
  %cst_i64 = arith.constant 0 : i64
  %vi64 = vector.transfer_read %arg2[%c0, %c0], %cst_i64 : memref<32x32xi64>, vector<32x32xi64>
  %trunci = arith.trunci %vi64 : vector<32x32xi64> to vector<32x32xi32>
  vector.transfer_write %trunci, %arg3[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_float_int_cast_ops(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf16>, %arg2 : memref<32x32xi32>, %arg3 : memref<32x32xi8>, %arg4 : memref<32x32xi32>, %arg5 : memref<32x32xi8>, %arg6 : memref<32x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_float_int_cast_ops
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xf32>) outs({{.*}} : memref<32x32xi32>) round_mode = <trunc>
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xf32>) outs({{.*}} : memref<32x32xi32>) round_mode = <trunc>
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xf16>) outs({{.*}} : memref<32x32xi8>) round_mode = <trunc>
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xi32>) outs({{.*}} : memref<32x32xf32>)
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xi32>) outs({{.*}} : memref<32x32xf32>)
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xi8>) outs({{.*}} : memref<32x32xf16>)
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<32x32xf16>) outs({{.*}} : memref<32x32xf32>)
  %c0 = arith.constant 0 : index
  %cst_f32 = arith.constant 0.0 : f32
  %vf32 = vector.transfer_read %arg0[%c0, %c0], %cst_f32 : memref<32x32xf32>, vector<32x32xf32>
  %fptosi = arith.fptosi %vf32 : vector<32x32xf32> to vector<32x32xi32>
  vector.transfer_write %fptosi, %arg4[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %fptoui = arith.fptoui %vf32 : vector<32x32xf32> to vector<32x32xi32>
  vector.transfer_write %fptoui, %arg4[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %cst_f16 = arith.constant 0.0 : f16
  %vf16 = vector.transfer_read %arg1[%c0, %c0], %cst_f16 : memref<32x32xf16>, vector<32x32xf16>
  %fptoui_i8 = arith.fptoui %vf16 : vector<32x32xf16> to vector<32x32xi8>
  vector.transfer_write %fptoui_i8, %arg5[%c0, %c0] : vector<32x32xi8>, memref<32x32xi8>
  %cst_i32 = arith.constant 0 : i32
  %vi32 = vector.transfer_read %arg2[%c0, %c0], %cst_i32 : memref<32x32xi32>, vector<32x32xi32>
  %sitofp = arith.sitofp %vi32 : vector<32x32xi32> to vector<32x32xf32>
  vector.transfer_write %sitofp, %arg6[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %uitofp = arith.uitofp %vi32 : vector<32x32xi32> to vector<32x32xf32>
  vector.transfer_write %uitofp, %arg6[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %cst_i8 = arith.constant 0 : i8
  %vi8 = vector.transfer_read %arg3[%c0, %c0], %cst_i8 : memref<32x32xi8>, vector<32x32xi8>
  %uitofp_i8 = arith.uitofp %vi8 : vector<32x32xi8> to vector<32x32xf32>
  vector.transfer_write %uitofp_i8, %arg6[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_bitcast_ops(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>, %arg3 : memref<32x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_bitcast_ops
  // CHECK: hivm.hir.bitcast {{.*}} : memref<32x32xf32> -> memref<32x32xi32>
  // CHECK: hivm.hir.bitcast {{.*}} : memref<32x32xi32> -> memref<32x32xf32>
  %c0 = arith.constant 0 : index
  %cst_f32 = arith.constant 0.0 : f32
  %vf32 = vector.transfer_read %arg0[%c0, %c0], %cst_f32 : memref<32x32xf32>, vector<32x32xf32>
  %to_i32 = arith.bitcast %vf32 : vector<32x32xf32> to vector<32x32xi32>
  vector.transfer_write %to_i32, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %cst_i32 = arith.constant 0 : i32
  %vi32 = vector.transfer_read %arg1[%c0, %c0], %cst_i32 : memref<32x32xi32>, vector<32x32xi32>
  %to_f32 = arith.bitcast %vi32 : vector<32x32xi32> to vector<32x32xf32>
  vector.transfer_write %to_f32, %arg3[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_predicates(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_cmpf_predicates
  // CHECK: hivm.hir.vcmp
  // CHECK: hivm.hir.vsel
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <ne>
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <le>
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <lt>
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <ge>
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <gt>
  // CHECK: hivm.hir.vcmp
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <ne>
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <le>
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <lt>
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <ge>
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <gt>
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %oeq = arith.cmpf oeq, %v0, %v1 : vector<32x32xf32>
  %sel_oeq = arith.select %oeq, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_oeq, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %one = arith.cmpf one, %v0, %v1 : vector<32x32xf32>
  %sel_one = arith.select %one, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_one, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %ole = arith.cmpf ole, %v0, %v1 : vector<32x32xf32>
  %sel_ole = arith.select %ole, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_ole, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %olt = arith.cmpf olt, %v0, %v1 : vector<32x32xf32>
  %sel_olt = arith.select %olt, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_olt, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %oge = arith.cmpf oge, %v0, %v1 : vector<32x32xf32>
  %sel_oge = arith.select %oge, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_oge, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %ogt = arith.cmpf ogt, %v0, %v1 : vector<32x32xf32>
  %sel_ogt = arith.select %ogt, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_ogt, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %ueq = arith.cmpf ueq, %v0, %v1 : vector<32x32xf32>
  %sel_ueq = arith.select %ueq, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_ueq, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %une = arith.cmpf une, %v0, %v1 : vector<32x32xf32>
  %sel_une = arith.select %une, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_une, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %ule = arith.cmpf ule, %v0, %v1 : vector<32x32xf32>
  %sel_ule = arith.select %ule, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_ule, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %ult = arith.cmpf ult, %v0, %v1 : vector<32x32xf32>
  %sel_ult = arith.select %ult, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_ult, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %uge = arith.cmpf uge, %v0, %v1 : vector<32x32xf32>
  %sel_uge = arith.select %uge, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_uge, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %ugt = arith.cmpf ugt, %v0, %v1 : vector<32x32xf32>
  %sel_ugt = arith.select %ugt, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %sel_ugt, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpi_predicates(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_cmpi_predicates
  // CHECK: hivm.hir.vcmp
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <ne>
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <lt>
  // CHECK: hivm.hir.vsel
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <gt>
  // CHECK: hivm.hir.vsel
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <le>
  // CHECK: hivm.hir.vsel
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <ge>
  // CHECK: hivm.hir.vsel
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <lt>
  // CHECK: hivm.hir.vsel
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <gt>
  // CHECK: hivm.hir.vsel
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <le>
  // CHECK: hivm.hir.vsel
  // CHECK: hivm.hir.vcmp{{.*}}compare_mode = <ge>
  // CHECK: hivm.hir.vsel
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %eq = arith.cmpi eq, %v0, %v1 : vector<32x32xi32>
  %sel_eq = arith.select %eq, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %sel_eq, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %ne = arith.cmpi ne, %v0, %v1 : vector<32x32xi32>
  %sel_ne = arith.select %ne, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %sel_ne, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %slt = arith.cmpi slt, %v0, %v1 : vector<32x32xi32>
  %sel_slt = arith.select %slt, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %sel_slt, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %sgt = arith.cmpi sgt, %v0, %v1 : vector<32x32xi32>
  %sel_sgt = arith.select %sgt, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %sel_sgt, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %sle = arith.cmpi sle, %v0, %v1 : vector<32x32xi32>
  %sel_sle = arith.select %sle, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %sel_sle, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %sge = arith.cmpi sge, %v0, %v1 : vector<32x32xi32>
  %sel_sge = arith.select %sge, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %sel_sge, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %ult = arith.cmpi ult, %v0, %v1 : vector<32x32xi32>
  %sel_ult = arith.select %ult, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %sel_ult, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %ugt = arith.cmpi ugt, %v0, %v1 : vector<32x32xi32>
  %sel_ugt = arith.select %ugt, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %sel_ugt, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %ule = arith.cmpi ule, %v0, %v1 : vector<32x32xi32>
  %sel_ule = arith.select %ule, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %sel_ule, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %uge = arith.cmpi uge, %v0, %v1 : vector<32x32xi32>
  %sel_uge = arith.select %uge, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %sel_uge, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----
func.func @test_mulsi_extended(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>, %arg3 : memref<32x32xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}>) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}>) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT_0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: %[[UB_OUT_1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vmulext ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT_0]], %[[UB_OUT_1]] : memref<32x32xi32>, memref<32x32xi32>)
  // CHECK: %[[SV_OUT_0:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT_0]] : memref<32x32xi32>) outs(%[[SV_OUT_0]] : memref<32x32xi32, {{.*}}>)
  // CHECK: %[[SV_OUT_1:.*]] = memref.subview %arg3[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT_1]] : memref<32x32xi32>) outs(%[[SV_OUT_1]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0, %res1 = arith.mulsi_extended %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  vector.transfer_write %res1, %arg3[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_mului_extended(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>, %arg3 : memref<32x32xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}>) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}>) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT_0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: %[[UB_OUT_1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vmulext ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT_0]], %[[UB_OUT_1]] : memref<32x32xi32>, memref<32x32xi32>)
  // CHECK: %[[SV_OUT_0:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT_0]] : memref<32x32xi32>) outs(%[[SV_OUT_0]] : memref<32x32xi32, {{.*}}>)
  // CHECK: %[[SV_OUT_1:.*]] = memref.subview %arg3[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT_1]] : memref<32x32xi32>) outs(%[[SV_OUT_1]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0, %res1 = arith.mului_extended %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  vector.transfer_write %res1, %arg3[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_elementwise_binary_ops(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>, %arg3 : memref<32x32xf32>, %arg4 : memref<32x32xf32>, %arg5 : memref<32x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_elementwise_binary_ops
  // CHECK: hivm.hir.vand ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vor ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vxor ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vmod ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vmod ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vshl ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
  // CHECK: hivm.hir.vshr ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>) round : true
  // CHECK: hivm.hir.vshr ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>) round : true
  // CHECK: hivm.hir.vmod ins({{.*}} : memref<32x32xf32>, memref<32x32xf32>) outs({{.*}} : memref<32x32xf32>)
  // CHECK: hivm.hir.vmin ins({{.*}} : memref<32x32xf32>, memref<32x32xf32>) outs({{.*}} : memref<32x32xf32>)
  // CHECK: hivm.hir.vmin ins({{.*}} : memref<32x32xf32>, memref<32x32xf32>) outs({{.*}} : memref<32x32xf32>)
  // CHECK: hivm.hir.vmax ins({{.*}} : memref<32x32xf32>, memref<32x32xf32>) outs({{.*}} : memref<32x32xf32>)
  // CHECK: hivm.hir.vmax ins({{.*}} : memref<32x32xf32>, memref<32x32xf32>) outs({{.*}} : memref<32x32xf32>)
  %c0 = arith.constant 0 : index
  %cst_i32 = arith.constant 0 : i32
  %i0 = vector.transfer_read %arg0[%c0, %c0], %cst_i32 : memref<32x32xi32>, vector<32x32xi32>
  %i1 = vector.transfer_read %arg1[%c0, %c0], %cst_i32 : memref<32x32xi32>, vector<32x32xi32>
  %andv = arith.andi %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %andv, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %orv = arith.ori %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %orv, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %xorv = arith.xori %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %xorv, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %remsi = arith.remsi %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %remsi, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %remui = arith.remui %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %remui, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %shl = arith.shli %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %shl, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %shrs = arith.shrsi %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %shrs, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %shru = arith.shrui %i0, %i1 : vector<32x32xi32>
  vector.transfer_write %shru, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  %cst_f32 = arith.constant 0.0 : f32
  %f0 = vector.transfer_read %arg3[%c0, %c0], %cst_f32 : memref<32x32xf32>, vector<32x32xf32>
  %f1 = vector.transfer_read %arg4[%c0, %c0], %cst_f32 : memref<32x32xf32>, vector<32x32xf32>
  %remf = arith.remf %f0, %f1 : vector<32x32xf32>
  vector.transfer_write %remf, %arg5[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %minnum = arith.minnumf %f0, %f1 : vector<32x32xf32>
  vector.transfer_write %minnum, %arg5[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %minimum = arith.minimumf %f0, %f1 : vector<32x32xf32>
  vector.transfer_write %minimum, %arg5[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %maxnum = arith.maxnumf %f0, %f1 : vector<32x32xf32>
  vector.transfer_write %maxnum, %arg5[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  %maximum = arith.maximumf %f0, %f1 : vector<32x32xf32>
  vector.transfer_write %maximum, %arg5[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----
func.func @test_constant_broadcast(%arg0 : memref<32x32xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[UB:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[UB]] : memref<32x32xf32>)
  // CHECK: %[[SV:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB]] : memref<32x32xf32>) outs(%[[SV]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %0 = arith.constant dense<1.0> : vector<32x32xf32>
  vector.transfer_write %0, %arg0[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}
// -----

// CHECK: func.func @test_vector_reduction_xor_standard(%[[ARG0:.*]]: memref<16xi32>, %[[ARG1:.*]]: memref<1xi32>)
func.func @test_vector_reduction_xor_standard(%arg0: memref<16xi32>, %arg1: memref<1xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  // CHECK: %[[SUBVIEW_READ:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xi32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_READ]] : memref<16xi32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xi32>) init_out_buffer = false
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xi32>
  // CHECK: hivm.hir.vreduce <xori> ins(%[[ALLOC_READ]] : memref<16xi32>) outs(%[[ALLOC_RED]] : memref<1xi32>) reduce_dims = [0]
  // CHECK: %[[SUBVIEW_WRITE:.*]] = memref.subview %[[ARG1]][0] [1] [1] : memref<1xi32> to memref<1xi32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC_RED]] : memref<1xi32>) outs(%[[SUBVIEW_WRITE]] : memref<1xi32, strided<[1]>>)
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xi32>, vector<16xi32>
  %1 = vector.reduction <xor>, %0 : vector<16xi32> into i32
  %2 = vector.broadcast %1 : i32 to vector<1xi32>
  vector.transfer_write %2, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  return
}

// -----

func.func @test_vector_reduction_integer_ops(%arg0: memref<16xi32>, %arg1: memref<1xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_vector_reduction_integer_ops
  // CHECK: hivm.hir.vreduce <sum> ins({{.*}} : memref<16xi32>) outs({{.*}} : memref<1xi32>) reduce_dims = [0]
  // CHECK: hivm.hir.vreduce <prod> ins({{.*}} : memref<16xi32>) outs({{.*}} : memref<1xi32>) reduce_dims = [0]
  // CHECK: hivm.hir.vreduce <andi> ins({{.*}} : memref<16xi32>) outs({{.*}} : memref<1xi32>) reduce_dims = [0]
  // CHECK: hivm.hir.vreduce <ori> ins({{.*}} : memref<16xi32>) outs({{.*}} : memref<1xi32>) reduce_dims = [0]
  // CHECK: hivm.hir.vreduce <max> ins({{.*}} : memref<16xi32>) outs({{.*}} : memref<1xi32>) reduce_dims = [0]
  // CHECK: hivm.hir.vreduce <min> ins({{.*}} : memref<16xi32>) outs({{.*}} : memref<1xi32>) reduce_dims = [0]
  %c0 = arith.constant 0 : index
  %cst0 = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0], %cst0 : memref<16xi32>, vector<16xi32>
  %add = vector.reduction <add>, %v0 : vector<16xi32> into i32
  %add_vec = vector.broadcast %add : i32 to vector<1xi32>
  vector.transfer_write %add_vec, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  %cst1 = arith.constant 1 : i32
  %v1 = vector.transfer_read %arg0[%c0], %cst1 : memref<16xi32>, vector<16xi32>
  %mul = vector.reduction <mul>, %v1 : vector<16xi32> into i32
  %mul_vec = vector.broadcast %mul : i32 to vector<1xi32>
  vector.transfer_write %mul_vec, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  %cst_all = arith.constant -1 : i32
  %vall = vector.transfer_read %arg0[%c0], %cst_all : memref<16xi32>, vector<16xi32>
  %and_red = vector.reduction <and>, %vall : vector<16xi32> into i32
  %and_vec = vector.broadcast %and_red : i32 to vector<1xi32>
  vector.transfer_write %and_vec, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  %or_red = vector.reduction <or>, %v0 : vector<16xi32> into i32
  %or_vec = vector.broadcast %or_red : i32 to vector<1xi32>
  vector.transfer_write %or_vec, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  %max = vector.reduction <maxsi>, %v0 : vector<16xi32> into i32
  %max_vec = vector.broadcast %max : i32 to vector<1xi32>
  vector.transfer_write %max_vec, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  %min = vector.reduction <minui>, %v0 : vector<16xi32> into i32
  %min_vec = vector.broadcast %min : i32 to vector<1xi32>
  vector.transfer_write %min_vec, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  return
}

// -----

func.func @test_vector_reduction_float_ops(%arg0: memref<16xf32>, %arg1: memref<1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_vector_reduction_float_ops
  // CHECK: hivm.hir.vreduce <sum> ins({{.*}} : memref<16xf32>) outs({{.*}} : memref<1xf32>) reduce_dims = [0]
  // CHECK: hivm.hir.vreduce <prod> ins({{.*}} : memref<16xf32>) outs({{.*}} : memref<1xf32>) reduce_dims = [0]
  // CHECK: hivm.hir.vreduce <max> ins({{.*}} : memref<16xf32>) outs({{.*}} : memref<1xf32>) reduce_dims = [0]
  %c0 = arith.constant 0 : index
  %cst0 = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0], %cst0 : memref<16xf32>, vector<16xf32>
  %add = vector.reduction <add>, %v0 : vector<16xf32> into f32
  %add_vec = vector.broadcast %add : f32 to vector<1xf32>
  vector.transfer_write %add_vec, %arg1[%c0] : vector<1xf32>, memref<1xf32>
  %cst1 = arith.constant 1.0 : f32
  %v1 = vector.transfer_read %arg0[%c0], %cst1 : memref<16xf32>, vector<16xf32>
  %mul = vector.reduction <mul>, %v1 : vector<16xf32> into f32
  %mul_vec = vector.broadcast %mul : f32 to vector<1xf32>
  vector.transfer_write %mul_vec, %arg1[%c0] : vector<1xf32>, memref<1xf32>
  %max = vector.reduction <maxnumf>, %v0 : vector<16xf32> into f32
  %max_vec = vector.broadcast %max : f32 to vector<1xf32>
  vector.transfer_write %max_vec, %arg1[%c0] : vector<1xf32>, memref<1xf32>
  return
}

// -----
// CHECK: func.func @test_vector_transfer(%[[ARG0:.*]]: memref<16xf32>, %[[ARG1:.*]]: memref<16xf32>)
func.func @test_vector_transfer(%arg0: memref<16xf32>, %arg1: memref<16xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32

  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xf32> to memref<16xf32, strided<[1]>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<16xf32, strided<[1]>>) outs(%[[ALLOC]] : memref<16xf32>)
  %0 = vector.transfer_read %arg0[%c0], %pad : memref<16xf32>, vector<16xf32>

  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[ARG1]][0] [16] [1] : memref<16xf32> to memref<16xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC]] : memref<16xf32>) outs(%[[SUBVIEW_DST]] : memref<16xf32, strided<[1]>>)
  vector.transfer_write %0, %arg1[%c0] : vector<16xf32>, memref<16xf32>
  return
}

// -----

func.func @test_minimal_scf_loop_npuvector() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_minimal_scf_loop_npuvector
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[UB:.*]] = memref.alloc() : memref<3072xf32>
  // CHECK: hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[UB]] : memref<3072xf32>)
  // CHECK: return

  %c0 = arith.constant 0 : index
  %c3072 = arith.constant 3072 : index
  %cst = arith.constant dense<1.0> : !npuvector<3072xf32>
  %0 = scf.for %i = %c0 to %c3072 step %c3072 iter_args(%acc = %cst) -> (!npuvector<3072xf32>) {
    scf.yield %acc : !npuvector<3072xf32>
  }
  return
}

// -----

func.func @test_minimal_scf_loop() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_minimal_scf_loop
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[UB:.*]] = memref.alloc() : memref<3072xf32>
  // CHECK: hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[UB]] : memref<3072xf32>)
  // CHECK: return

  %c0 = arith.constant 0 : index
  %c3072 = arith.constant 3072 : index
  %cst = arith.constant dense<1.0> : vector<3072xf32>
  %0 = scf.for %i = %c0 to %c3072 step %c3072 iter_args(%acc = %cst) -> (vector<3072xf32>) {
    scf.yield %acc : vector<3072xf32>
  }
  return
}

// -----

func.func @static_npuvector_transfer_read_write(%in: memref<1024xf32>, %out: memref<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @static_npuvector_transfer_read_write
  // CHECK-SAME: (%[[IN:.*]]: memref<1024xf32>, %[[OUT:.*]]: memref<1024xf32>)
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[IN]][0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<1024xf32, strided<[1]>>) outs(%[[ALLOC]] : memref<1024xf32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[OUT]][0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC]] : memref<1024xf32>) outs(%[[SUBVIEW_DST]] : memref<1024xf32, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %vec = npuvector.transfer_read %in[%c0], %padding : memref<1024xf32>, !npuvector<1024xf32>
  npuvector.transfer_write %vec, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  return
}

// -----

func.func @test_negf_npuvector(%in: memref<1024xf32>, %out: memref<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_negf_npuvector
  // CHECK-SAME: (%[[IN:.*]]: memref<1024xf32>, %[[OUT:.*]]: memref<1024xf32>)
  // CHECK: %[[CST0:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[IN]][0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>
  // CHECK: %[[IN_ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<1024xf32, strided<[1]>>) outs(%[[IN_ALLOC]] : memref<1024xf32>)
  // CHECK: %[[ZERO_ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.vbrc ins(%[[CST0]] : f32) outs(%[[ZERO_ALLOC]] : memref<1024xf32>)
  // CHECK: %[[RES_ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.vsub ins(%[[ZERO_ALLOC]], %[[IN_ALLOC]] : memref<1024xf32>, memref<1024xf32>) outs(%[[RES_ALLOC]] : memref<1024xf32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[OUT]][0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[RES_ALLOC]] : memref<1024xf32>) outs(%[[SUBVIEW_DST]] : memref<1024xf32, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %vec = npuvector.transfer_read %in[%c0], %padding : memref<1024xf32>, !npuvector<1024xf32>
  %neg = arith.negf %vec : !npuvector<1024xf32>
  npuvector.transfer_write %neg, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  return
}

// -----

func.func @test_math_unary_npuvector_ops(%in: memref<1024xf32>, %out: memref<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_math_unary_npuvector_ops
  // CHECK: hivm.hir.vexp ins({{.*}} : memref<1024xf32>) outs({{.*}} : memref<1024xf32>)
  // CHECK: hivm.hir.vrsqrt ins({{.*}} : memref<1024xf32>) outs({{.*}} : memref<1024xf32>)
  // CHECK: hivm.hir.vln ins({{.*}} : memref<1024xf32>) outs({{.*}} : memref<1024xf32>)
  // CHECK: hivm.hir.vabs ins({{.*}} : memref<1024xf32>) outs({{.*}} : memref<1024xf32>)
  // CHECK: hivm.hir.vsqrt ins({{.*}} : memref<1024xf32>) outs({{.*}} : memref<1024xf32>)
  // CHECK: hivm.hir.vtanh ins({{.*}} : memref<1024xf32>) outs({{.*}} : memref<1024xf32>)
  // CHECK: hivm.hir.vsin ins({{.*}} : memref<1024xf32>) outs({{.*}} : memref<1024xf32>)
  // CHECK: hivm.hir.vcos ins({{.*}} : memref<1024xf32>) outs({{.*}} : memref<1024xf32>)
  // CHECK: hivm.hir.verf ins({{.*}} : memref<1024xf32>) outs({{.*}} : memref<1024xf32>)
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %vec = npuvector.transfer_read %in[%c0], %padding : memref<1024xf32>, !npuvector<1024xf32>
  %exp = math.exp %vec : !npuvector<1024xf32>
  npuvector.transfer_write %exp, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %rsqrt = math.rsqrt %vec : !npuvector<1024xf32>
  npuvector.transfer_write %rsqrt, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %log = math.log %vec : !npuvector<1024xf32>
  npuvector.transfer_write %log, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %absf = math.absf %vec : !npuvector<1024xf32>
  npuvector.transfer_write %absf, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %sqrt = math.sqrt %vec : !npuvector<1024xf32>
  npuvector.transfer_write %sqrt, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %tanh = math.tanh %vec : !npuvector<1024xf32>
  npuvector.transfer_write %tanh, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %sin = math.sin %vec : !npuvector<1024xf32>
  npuvector.transfer_write %sin, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %cos = math.cos %vec : !npuvector<1024xf32>
  npuvector.transfer_write %cos, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %erf = math.erf %vec : !npuvector<1024xf32>
  npuvector.transfer_write %erf, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  return
}

// -----
func.func @test_math_unary_round_npuvector(%in: memref<1024xf32>, %out_ceil: memref<1024xf32>, %out_floor: memref<1024xf32>, %out_round: memref<1024xf32>, %out_roundeven: memref<1024xf32>, %out_trunc: memref<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_math_unary_round_npuvector
  // CHECK: hivm.hir.vcast{{.*}}round_mode = <ceil>
  // CHECK: hivm.hir.vcast{{.*}}round_mode = <floor>
  // CHECK: hivm.hir.vcast{{.*}}round_mode = <round>
  // roundeven maps to default RINT; assembly omits round_mode when it equals the default
  // CHECK: hivm.hir.vcast ins({{.*}} : memref<1024xf32>) outs({{.*}} : memref<1024xf32>)
  // CHECK: hivm.hir.vcast{{.*}}round_mode = <trunc>
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %vec = npuvector.transfer_read %in[%c0], %padding : memref<1024xf32>, !npuvector<1024xf32>
  %ceil = math.ceil %vec : !npuvector<1024xf32>
  npuvector.transfer_write %ceil, %out_ceil[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %floor = math.floor %vec : !npuvector<1024xf32>
  npuvector.transfer_write %floor, %out_floor[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %round = math.round %vec : !npuvector<1024xf32>
  npuvector.transfer_write %round, %out_round[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %roundeven = math.roundeven %vec : !npuvector<1024xf32>
  npuvector.transfer_write %roundeven, %out_roundeven[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  %trunc = math.trunc %vec : !npuvector<1024xf32>
  npuvector.transfer_write %trunc, %out_trunc[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  return
}

// -----

func.func @test_absi_npuvector(%in: memref<1024xi32>, %out: memref<1024xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_absi_npuvector
  // CHECK-SAME: (%[[IN:.*]]: memref<1024xi32>, %[[OUT:.*]]: memref<1024xi32>)
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[IN]][0] [1024] [1] : memref<1024xi32> to memref<1024xi32, strided<[1]>>
  // CHECK: %[[IN_ALLOC:.*]] = memref.alloc() : memref<1024xi32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<1024xi32, strided<[1]>>) outs(%[[IN_ALLOC]] : memref<1024xi32>)
  // CHECK: %[[RES_ALLOC:.*]] = memref.alloc() : memref<1024xi32>
  // CHECK: hivm.hir.vabs ins(%[[IN_ALLOC]] : memref<1024xi32>) outs(%[[RES_ALLOC]] : memref<1024xi32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[OUT]][0] [1024] [1] : memref<1024xi32> to memref<1024xi32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[RES_ALLOC]] : memref<1024xi32>) outs(%[[SUBVIEW_DST]] : memref<1024xi32, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0 : i32
  %vec = npuvector.transfer_read %in[%c0], %padding : memref<1024xi32>, !npuvector<1024xi32>
  %res = math.absi %vec : !npuvector<1024xi32>
  npuvector.transfer_write %res, %out[%c0] : !npuvector<1024xi32>, memref<1024xi32>
  return
}

// -----

func.func @dynamic_npuvector_transfer_read_write(%in: memref<?xf32>, %out: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @dynamic_npuvector_transfer_read_write
  // CHECK-SAME: (%[[IN:.*]]: memref<?xf32>, %[[OUT:.*]]: memref<?xf32>)
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = memref.dim %[[IN]], %[[C0]] : memref<?xf32>
  // CHECK: %[[SV_IN:.*]] = memref.subview %[[IN]][0] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.load ins(%[[SV_IN]] : memref<?xf32, strided<[1]>>) outs(%[[ALLOC]] : memref<?xf32>) init_out_buffer = false
  // CHECK: %[[SV_OUT:.*]] = memref.subview %[[OUT]][0] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC]] : memref<?xf32>) outs(%[[SV_OUT]] : memref<?xf32, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %dim = memref.dim %in, %c0 : memref<?xf32>
  %max_vl = arith.constant 4096 : index
  %vec = npuvector.transfer_read %in[%c0][%dim][%max_vl], %padding : memref<?xf32>, !npuvector<?xf32>
  npuvector.transfer_write %vec, %out[%c0] : !npuvector<?xf32>, memref<?xf32>
  return
}

// -----

func.func @dynamic_npuvector_transfer_read_static_dim_uses_rank_max(%in: memref<1024x1024x12xf32>, %out: memref<1024x1024x12xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @dynamic_npuvector_transfer_read_static_dim_uses_rank_max
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}, %{{.*}}) : memref<?x?x12xf32>
  // CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 28672 : index} : memref<?x?x12xf32>
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %c56 = arith.constant 56 : index
  %padding = arith.constant 0.0 : f32
  %vec = npuvector.transfer_read %in[%c0, %c0, %c0][%c8, %c56, %c12][%c8, %c56, %c16], %padding : memref<1024x1024x12xf32>, !npuvector<?x?x12xf32>
  npuvector.transfer_write %vec, %out[%c0, %c0, %c0] : !npuvector<?x?x12xf32>, memref<1024x1024x12xf32>
  return
}

// -----

func.func @static_npuvector_transfer_read_keeps_static_size(%in: memref<1xf32>, %out: memref<1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @static_npuvector_transfer_read_keeps_static_size
  // CHECK-SAME: (%[[IN:.*]]: memref<1xf32>, %[[OUT:.*]]: memref<1xf32>)
  // CHECK: %[[SV_IN:.*]] = memref.subview %[[IN]][0] [1] [1] : memref<1xf32> to memref<1xf32, strided<[1]>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: hivm.hir.load ins(%[[SV_IN]] : memref<1xf32, strided<[1]>>) outs(%[[ALLOC]] : memref<1xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %padding = arith.constant 0.0 : f32
  %vec = npuvector.transfer_read %in[%c0][%c1][%c1], %padding : memref<1xf32>, !npuvector<1xf32>
  npuvector.transfer_write %vec, %out[%c0] : !npuvector<1xf32>, memref<1xf32>
  return
}

// -----

func.func @simple_broadcast_static(%arg0: f32, %out: memref<1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @simple_broadcast_static
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.vbrc ins(%arg0 : f32) outs(%[[ALLOC]] : memref<1024xf32>)
  // CHECK: %{{.*}} = memref.subview %arg1[0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC]] : memref<1024xf32>) outs(%{{.*}} : memref<1024xf32, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %0 = npuvector.broadcast %arg0 : f32 to !npuvector<1024xf32>
  npuvector.transfer_write %0, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  return
}

// -----

func.func @simple_broadcast_dynamic(%mem: memref<?xf32>, %out: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @simple_broadcast_dynamic
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xf32>
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 16384 : index}
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[ALLOC]] : memref<?xf32>)
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %dim = memref.dim %mem, %c0 : memref<?xf32>
  %0 = npuvector.broadcast %cst_0[%dim] [%c4096] : f32 to !npuvector<?xf32>
  npuvector.transfer_write %0, %out[%c0] : !npuvector<?xf32>, memref<?xf32>
  return
}

// -----

func.func @dynamic_npuvector_broadcast_static_dim_uses_rank_max(%out: memref<12x1024x1024xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @dynamic_npuvector_broadcast_static_dim_uses_rank_max
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%{{.*}}, %{{.*}}) : memref<12x?x?xf32>
  // CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 21504 : index} : memref<12x?x?xf32>
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %c56 = arith.constant 56 : index
  %cst = arith.constant 1.0 : f32
  %vec = npuvector.broadcast %cst[%c12, %c8, %c56] [%c16, %c8, %c56] : f32 to !npuvector<12x?x?xf32>
  npuvector.transfer_write %vec, %out[%c0, %c0, %c0] : !npuvector<12x?x?xf32>, memref<12x1024x1024xf32>
  return
}

// -----

func.func @test_elementwise_static(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>, %arg3: f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_elementwise_static
  // CHECK: %{{.*}} = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 1024 : index
  // CHECK: scf.for %[[ARG4:.*]] = %c0 to %c1024 step %c1024 {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %arg0[%[[ARG4]]] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_1:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK:   hivm.hir.load ins(%[[SUBVIEW]] : memref<1024xf32, strided<[1], offset: ?>>) outs(%[[ALLOC_1]] : memref<1024xf32>) init_out_buffer = false
  // CHECK:   %[[SUBVIEW_3:.*]] = memref.subview %arg1[%[[ARG4]]] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_4:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK:   hivm.hir.load ins(%[[SUBVIEW_3]] : memref<1024xf32, strided<[1], offset: ?>>) outs(%[[ALLOC_4]] : memref<1024xf32>) init_out_buffer = false
  // CHECK:   %[[ALLOC_5:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK:   hivm.hir.vmul ins(%[[ALLOC_1]], %arg3 : memref<1024xf32>, f32) outs(%[[ALLOC_5]] : memref<1024xf32>)
  // CHECK:   %[[ALLOC_6:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK:   hivm.hir.vadd ins(%[[ALLOC_5]], %[[ALLOC_4]] : memref<1024xf32>, memref<1024xf32>) outs(%[[ALLOC_6]] : memref<1024xf32>)
  // CHECK:   %[[SUBVIEW_7:.*]] = memref.subview %arg2[%[[ARG4]]] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1], offset: ?>>
  // CHECK:   hivm.hir.store ins(%[[ALLOC_6]] : memref<1024xf32>) outs(%[[SUBVIEW_7]] : memref<1024xf32, strided<[1], offset: ?>>)
  // CHECK: }
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c1024_0 = arith.constant 1024 : index
  %0 = npuvector.broadcast %arg3 : f32 to !npuvector<1024xf32>
  scf.for %arg4 = %c0 to %c1024 step %c1024_0 {
    %cst = arith.constant 0.000000e+00 : f32
    %1 = npuvector.transfer_read %arg0[%arg4], %cst : memref<1024xf32>, !npuvector<1024xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %2 = npuvector.transfer_read %arg1[%arg4], %cst_1 : memref<1024xf32>, !npuvector<1024xf32>
    %3 = arith.mulf %0, %1 : !npuvector<1024xf32>
    %4 = arith.addf %3, %2 : !npuvector<1024xf32>
    npuvector.transfer_write %4, %arg2[%arg4] : !npuvector<1024xf32>, memref<1024xf32>
  }
  return
}

// -----

func.func @test_elementwise_dynamic(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: f32, %arg3: f32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_elementwise_dynamic
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32)
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xf32>
  // CHECK: scf.for %[[ARG4:.*]] = %[[C0]] to %[[DIM]] step %[[DIM]] {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[ARG0]][%[[ARG4]]] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_1:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK:   annotation.mark %[[ALLOC_1]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK:   hivm.hir.load ins(%[[SUBVIEW]] : memref<?xf32, strided<[1], offset: ?>>) outs(%[[ALLOC_1]] : memref<?xf32>) init_out_buffer = false
  // CHECK:   %[[ALLOC_4:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK:   annotation.mark %[[ALLOC_4]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK:   hivm.hir.vmul ins(%[[ALLOC_1]], %[[ARG2]] : memref<?xf32>, f32) outs(%[[ALLOC_4]] : memref<?xf32>)
  // CHECK:   %[[ALLOC_7:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK:   annotation.mark %[[ALLOC_7]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK:   hivm.hir.vadd ins(%[[ALLOC_4]], %[[ARG3]] : memref<?xf32>, f32) outs(%[[ALLOC_7]] : memref<?xf32>)
  // CHECK:   %[[SUBVIEW_10:.*]] = memref.subview %[[ARG1]][%[[ARG4]]] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK:   hivm.hir.store ins(%[[ALLOC_7]] : memref<?xf32>) outs(%[[SUBVIEW_10]] : memref<?xf32, strided<[1], offset: ?>>)
  // CHECK: }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?xf32>
  %c4096 = arith.constant 4096 : index
  %0 = npuvector.broadcast %arg2[%dim] [%c4096] : f32 to !npuvector<?xf32>
  %1 = npuvector.broadcast %arg3[%dim] [%c4096] : f32 to !npuvector<?xf32>
  scf.for %arg4 = %c0 to %dim step %dim {
    %cst = arith.constant 0.000000e+00 : f32
    %2 = npuvector.transfer_read %arg0[%arg4][%dim][%c4096], %cst : memref<?xf32>, !npuvector<?xf32>
    %3 = arith.mulf %2, %0 : !npuvector<?xf32>
    %4 = arith.addf %3, %1 : !npuvector<?xf32>
    npuvector.transfer_write %4, %arg1[%arg4] : !npuvector<?xf32>, memref<?xf32>
  }
  return
}

// -----

func.func @test_npuvector_extf_static(%arg0: memref<128xbf16>, %arg1: memref<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xbf16> to memref<128xbf16, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xbf16>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xbf16, {{.*}}>) outs(%[[UB0]] : memref<128xbf16>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xbf16>) outs(%[[UB1]] : memref<128xf32>) round_mode = <round>
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xf32>) outs(%[[SV1]] : memref<128xf32, {{.*}}>)
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = npuvector.transfer_read %arg0[%arg2], %cst : memref<128xbf16>, !npuvector<128xbf16>
    %1 = npuvector.extf %0 {round_mode = #hivm.round_mode<round>} : !npuvector<128xbf16> to !npuvector<128xf32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xf32>, memref<128xf32>
  }
  return
}

// -----

func.func @test_npuvector_truncf_static(%arg0: memref<128xf32>, %arg1: memref<128xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_truncf_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xbf16>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xf32>) outs(%[[UB1]] : memref<128xbf16>) round_mode = <round>
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xbf16> to memref<128xbf16, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xbf16>) outs(%[[SV1]] : memref<128xbf16, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = npuvector.transfer_read %arg0[%arg2], %cst : memref<128xf32>, !npuvector<128xf32>
    %1 = npuvector.truncf %0 {round_mode = #hivm.round_mode<round>} : !npuvector<128xf32> to !npuvector<128xbf16>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xbf16>, memref<128xbf16>
  }
  return
}

// -----

func.func @test_npuvector_extsi_static(%arg0: memref<128xi8>, %arg1: memref<128xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_extsi_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi8> to memref<128xi8, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi8>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi8, {{.*}}>) outs(%[[UB0]] : memref<128xi8>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi8>) outs(%[[UB1]] : memref<128xi32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi32>) outs(%[[SV1]] : memref<128xi32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i8 = arith.constant 0 : i8
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i8 : memref<128xi8>, !npuvector<128xi8>
    %1 = npuvector.extsi %0 : !npuvector<128xi8> to !npuvector<128xi32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi32>, memref<128xi32>
  }
  return
}

// -----

func.func @test_npuvector_extui_static(%arg0: memref<128xi8>, %arg1: memref<128xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_extui_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi8> to memref<128xi8, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi8>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi8, {{.*}}>) outs(%[[UB0]] : memref<128xi8>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi8>) outs(%[[UB1]] : memref<128xi32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi32>) outs(%[[SV1]] : memref<128xi32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i8 = arith.constant 0 : i8
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i8 : memref<128xi8>, !npuvector<128xi8>
    %1 = npuvector.extui %0 : !npuvector<128xi8> to !npuvector<128xi32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi32>, memref<128xi32>
  }
  return
}

// -----

func.func @test_npuvector_trunci_static(%arg0: memref<128xi32>, %arg1: memref<128xi8>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_trunci_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi32, {{.*}}>) outs(%[[UB0]] : memref<128xi32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi8>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi32>) outs(%[[UB1]] : memref<128xi8>) round_mode = <truncwithoverflow>
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi8> to memref<128xi8, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi8>) outs(%[[SV1]] : memref<128xi8, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i32 = arith.constant 0 : i32
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i32 : memref<128xi32>, !npuvector<128xi32>
    %1 = npuvector.trunci %0 : !npuvector<128xi32> to !npuvector<128xi8>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi8>, memref<128xi8>
  }
  return
}

// -----

func.func @test_npuvector_sitofp_static(%arg0: memref<128xi32>, %arg1: memref<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_sitofp_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi32, {{.*}}>) outs(%[[UB0]] : memref<128xi32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi32>) outs(%[[UB1]] : memref<128xf32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xf32>) outs(%[[SV1]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i32 = arith.constant 0 : i32
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i32 : memref<128xi32>, !npuvector<128xi32>
    %1 = npuvector.sitofp %0 : !npuvector<128xi32> to !npuvector<128xf32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xf32>, memref<128xf32>
  }
  return
}

// -----

func.func @test_npuvector_sitofp_i8_to_f32_static(%arg0: memref<128xi8>, %arg1: memref<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_sitofp_i8_to_f32_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi8> to memref<128xi8, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi8>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi8, {{.*}}>) outs(%[[UB0]] : memref<128xi8>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf16>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi8>) outs(%[[UB1]] : memref<128xf16>)
  // CHECK:   %[[UB2:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB1]] : memref<128xf16>) outs(%[[UB2]] : memref<128xf32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB2]] : memref<128xf32>) outs(%[[SV1]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i8 = arith.constant 0 : i8
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i8 : memref<128xi8>, !npuvector<128xi8>
    %1 = npuvector.sitofp %0 : !npuvector<128xi8> to !npuvector<128xf32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xf32>, memref<128xf32>
  }
  return
}

// -----

func.func @test_npuvector_uitofp_static(%arg0: memref<128xi32>, %arg1: memref<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_uitofp_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi32, {{.*}}>) outs(%[[UB0]] : memref<128xi32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi32>) outs(%[[UB1]] : memref<128xf32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xf32>) outs(%[[SV1]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i32 = arith.constant 0 : i32
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i32 : memref<128xi32>, !npuvector<128xi32>
    %1 = npuvector.uitofp %0 : !npuvector<128xi32> to !npuvector<128xf32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xf32>, memref<128xf32>
  }
  return
}

// -----

func.func @test_npuvector_uitofp_i8_to_f32_static(%arg0: memref<128xi8>, %arg1: memref<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_uitofp_i8_to_f32_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi8> to memref<128xi8, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi8>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi8, {{.*}}>) outs(%[[UB0]] : memref<128xi8>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf16>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi8>) outs(%[[UB1]] : memref<128xf16>)
  // CHECK:   %[[UB2:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB1]] : memref<128xf16>) outs(%[[UB2]] : memref<128xf32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB2]] : memref<128xf32>) outs(%[[SV1]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i8 = arith.constant 0 : i8
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i8 : memref<128xi8>, !npuvector<128xi8>
    %1 = npuvector.uitofp %0 : !npuvector<128xi8> to !npuvector<128xf32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xf32>, memref<128xf32>
  }
  return
}

// -----

func.func @test_npuvector_fptosi_static(%arg0: memref<128xf32>, %arg1: memref<128xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_fptosi_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xf32>) outs(%[[UB1]] : memref<128xi32>) round_mode = <trunc>
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi32>) outs(%[[SV1]] : memref<128xi32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = npuvector.transfer_read %arg0[%arg2], %cst : memref<128xf32>, !npuvector<128xf32>
    %1 = npuvector.fptosi %0 : !npuvector<128xf32> to !npuvector<128xi32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi32>, memref<128xi32>
  }
  return
}

// -----

func.func @test_npuvector_fptoui_static(%arg0: memref<128xf32>, %arg1: memref<128xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_fptoui_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xf32>) outs(%[[UB1]] : memref<128xi32>) round_mode = <trunc>
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi32>) outs(%[[SV1]] : memref<128xi32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = npuvector.transfer_read %arg0[%arg2], %cst : memref<128xf32>, !npuvector<128xf32>
    %1 = npuvector.fptoui %0 : !npuvector<128xf32> to !npuvector<128xi32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi32>, memref<128xi32>
  }
  return
}

// -----

func.func @test_npuvector_bitcast_static(%arg0: memref<128xbf16>, %arg1: memref<128xi16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_bitcast_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xbf16> to memref<128xbf16, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xbf16>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xbf16, {{.*}}>) outs(%[[UB0]] : memref<128xbf16>)
  // CHECK:   %[[UB1:.*]] = hivm.hir.bitcast %[[UB0]] : memref<128xbf16> -> memref<128xi16>
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi16> to memref<128xi16, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi16>) outs(%[[SV1]] : memref<128xi16, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = npuvector.transfer_read %arg0[%arg2], %cst : memref<128xbf16>, !npuvector<128xbf16>
    %1 = npuvector.bitcast %0 : !npuvector<128xbf16> to !npuvector<128xi16>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi16>, memref<128xi16>
  }
  return
}

// -----

func.func @test_npuvector_cmpi_static(%arg0: memref<128xi32>, %arg1: memref<128xi32>, %arg2: memref<128xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_cmpi_static
  // CHECK: scf.for %[[ARG3:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG3]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi32, {{.*}}>) outs(%[[UB0]] : memref<128xi32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG3]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.load ins(%[[SV1]] : memref<128xi32, {{.*}}>) outs(%[[UB1]] : memref<128xi32>)
  // CHECK:   %[[UB_OUT:.*]] = memref.alloc() : memref<128xi1>
  // CHECK:   hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<128xi32>, memref<128xi32>) outs(%[[UB_OUT]] : memref<128xi1>) compare_mode = <gt>
  // CHECK:   %[[SEL_RES:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<128xi1>, memref<128xi32>, memref<128xi32>) outs(%[[SEL_RES]] : memref<128xi32>)
  // CHECK:   %[[SV2:.*]] = memref.subview %arg2[%[[ARG3]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%{{.*}} : memref<128xi32>) outs(%[[SV2]] : memref<128xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg3 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0 : i32
    %0 = npuvector.transfer_read %arg0[%arg3], %cst : memref<128xi32>, !npuvector<128xi32>
    %1 = npuvector.transfer_read %arg1[%arg3], %cst : memref<128xi32>, !npuvector<128xi32>
    %2 = npuvector.cmpi sgt, %0, %1 : !npuvector<128xi32> to !npuvector<128xi1>
    %3 = npuvector.select %2, %0, %1 : !npuvector<128xi1>, !npuvector<128xi32>
    npuvector.transfer_write %3, %arg2[%arg3] : !npuvector<128xi32>, memref<128xi32>
  }
  return
}

// -----

func.func @test_npuvector_cmpf_static(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_cmpf_static
  // CHECK: scf.for %[[ARG3:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG3]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG3]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.load ins(%[[SV1]] : memref<128xf32, {{.*}}>) outs(%[[UB1]] : memref<128xf32>)
  // CHECK:   %[[UB_OUT:.*]] = memref.alloc() : memref<128xi1>
  // CHECK:   hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<128xf32>, memref<128xf32>) outs(%[[UB_OUT]] : memref<128xi1>) compare_mode = <gt>
  // CHECK:   %[[SEL_RES:.*]] = memref.alloc() : memref<128x{{.*}}>
  // CHECK:   hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<128xi1>, memref<128x{{.*}}>, memref<128x{{.*}}>) outs(%[[SEL_RES]] : memref<128x{{.*}}>)
  // CHECK:   %[[SV2:.*]] = memref.subview %arg2[%[[ARG3]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%{{.*}} : memref<128xf32>) outs(%[[SV2]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg3 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = npuvector.transfer_read %arg0[%arg3], %cst : memref<128xf32>, !npuvector<128xf32>
    %1 = npuvector.transfer_read %arg1[%arg3], %cst : memref<128xf32>, !npuvector<128xf32>
    %2 = npuvector.cmpf ogt, %0, %1 : !npuvector<128xf32> to !npuvector<128xi1>
    %3 = npuvector.select %2, %0, %1 : !npuvector<128xi1>, !npuvector<128xf32>
    npuvector.transfer_write %3, %arg2[%arg3] : !npuvector<128xf32>, memref<128xf32>
  }
  return
}

// -----

func.func @test_npuvector_mixed_type_conversion(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C128:.*]] = arith.constant 128 : index
  // CHECK: %[[CST1:.*]] = arith.constant 2.000000e+00 : f32
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128 {
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xbf16> to memref<128xbf16, strided<[1], offset: ?>>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xbf16>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xbf16, strided<[1], offset: ?>>) outs(%[[UB0]] : memref<128xbf16>) init_out_buffer = false
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xbf16>) outs(%[[UB1]] : memref<128xf32>)
  // CHECK:   %[[UB2:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vmul ins(%[[UB1]], %[[CST1]] : memref<128xf32>, f32) outs(%[[UB2]] : memref<128xf32>)
  // CHECK:   %[[UB3:.*]] = memref.alloc() : memref<128xbf16>
  // CHECK:   hivm.hir.vcast ins(%[[UB2]] : memref<128xf32>) outs(%[[UB3]] : memref<128xbf16>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xbf16> to memref<128xbf16, strided<[1], offset: ?>>
  // CHECK:   hivm.hir.store ins(%[[UB3]] : memref<128xbf16>) outs(%[[SV1]] : memref<128xbf16, strided<[1], offset: ?>>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.000000e+00 : f32
  %c128_0 = arith.constant 128 : index
  %0 = npuvector.broadcast %cst : f32 to !npuvector<128xf32>
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %1 = npuvector.transfer_read %arg0[%arg2], %cst_1 : memref<128xbf16>, !npuvector<128xbf16>
    %2 = npuvector.extf %1 : !npuvector<128xbf16> to !npuvector<128xf32>
    %3 = arith.mulf %2, %0 : !npuvector<128xf32>
    %4 = npuvector.truncf %3 : !npuvector<128xf32> to !npuvector<128xbf16>
    npuvector.transfer_write %4, %arg1[%arg2] : !npuvector<128xbf16>, memref<128xbf16>
  }
  return
}

// -----

func.func @test_npuvector_extf_dynamic(%arg0: memref<?xbf16>, %arg1: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_extf_dynamic
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = memref.dim %arg0, %[[C0]] : memref<?xbf16>
  // CHECK: scf.for %[[ARG2:.*]] = %[[C0]] to %[[DIM]] step %[[DIM]] {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %arg0[%[[ARG2]]] [%[[DIM]]] [1] : memref<?xbf16> to memref<?xbf16, strided<[1], offset: ?>>
  // CHECK:   %[[UB0:.*]] = memref.alloc(%[[DIM]]) : memref<?xbf16>
  // CHECK:   annotation.mark %[[UB0]] {buffer_size_in_byte = 8192 : index} : memref<?xbf16>
  // CHECK:   hivm.hir.load ins(%[[SUBVIEW]] : memref<?xbf16, strided<[1], offset: ?>>) outs(%[[UB0]] : memref<?xbf16>)
  // CHECK:   %[[UB1:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK:   annotation.mark %[[UB1]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<?xbf16>) outs(%[[UB1]] : memref<?xf32>)
  // CHECK:   %[[SUBVIEW_OUT:.*]] = memref.subview %arg1[%[[ARG2]]] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<?xf32>) outs(%[[SUBVIEW_OUT]] : memref<?xf32, strided<[1], offset: ?>>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?xbf16>
  %c4096 = arith.constant 4096 : index
  scf.for %arg2 = %c0 to %dim step %dim {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = npuvector.transfer_read %arg0[%arg2] [%dim] [%c4096], %cst : memref<?xbf16>, !npuvector<?xbf16>
    %1 = npuvector.extf %0 : !npuvector<?xbf16> to !npuvector<?xf32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<?xf32>, memref<?xf32>
  }
  return
}

// -----

func.func @test_rank0_transfer_read_write_dynamic(%arg0: memref<f32>, %arg1: memref<1x?xf32>, %arg2: memref<?xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM:.*]] = memref.dim %arg1, %[[C1]] : memref<1x?xf32>
  // CHECK: %[[SCALAR:.*]] = memref.load %arg0[] : memref<f32>
  // CHECK: %[[UB0:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[UB0]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.vbrc ins(%[[SCALAR]] : f32) outs(%[[UB0]] : memref<?xf32>)
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %arg2[0] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, {{.*}}>
  // CHECK: %[[UB1:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[UB1]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<?xf32, {{.*}}>) outs(%[[UB1]] : memref<?xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[UB_OUT]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %[[UB1]] : memref<?xf32>, memref<?xf32>) outs(%[[UB_OUT]] : memref<?xf32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %arg1[0, 0] [1, %[[DIM]]] [1, 1] : memref<1x?xf32> to memref<?xf32, {{.*}}>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<?xf32>) outs(%[[SUBVIEW_DST]] : memref<?xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg1, %c1 : memref<1x?xf32>
  %c4096 = arith.constant 4096 : index
  %pad = arith.constant 0.0 : f32
  %vec_scalar = npuvector.transfer_read %arg0[] [%dim] [%c4096], %pad : memref<f32>, !npuvector<?xf32>
  %vec_arg2 = npuvector.transfer_read %arg2[%c0] [%dim] [%c4096], %pad : memref<?xf32>, !npuvector<?xf32>
  %vec_add = arith.addf %vec_scalar, %vec_arg2 : !npuvector<?xf32>
  npuvector.transfer_write %vec_add, %arg1[%c0, %c0] : !npuvector<?xf32>, memref<1x?xf32>
  return
}

// -----

func.func @test_rank0_transfer_read_write_static(%arg0: memref<f32>, %arg1: memref<1x170xf32>,%arg2: memref<170xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[SCALAR:.*]] = memref.load %arg0[] : memref<f32>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<170xf32>
  // CHECK: hivm.hir.vbrc ins(%[[SCALAR]] : f32) outs(%[[UB0]] : memref<170xf32>)
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %arg2[0] [170] [1] : memref<170xf32> to memref<170xf32, {{.*}}>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<170xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<170xf32, {{.*}}>) outs(%[[UB1]] : memref<170xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<170xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %[[UB1]] : memref<170xf32>, memref<170xf32>) outs(%[[UB_OUT]] : memref<170xf32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %arg1[0, 0] [1, 170] [1, 1] : memref<1x170xf32> to memref<170xf32, {{.*}}>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<170xf32>) outs(%[[SUBVIEW_DST]] : memref<170xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %vec_scalar = npuvector.transfer_read %arg0[], %pad : memref<f32>, !npuvector<170xf32>
  %vec_arg2 = npuvector.transfer_read %arg2[%c0], %pad : memref<170xf32>, !npuvector<170xf32>
  %vec_add = arith.addf %vec_scalar, %vec_arg2 : !npuvector<170xf32>
  npuvector.transfer_write %vec_add, %arg1[%c0, %c0] : !npuvector<170xf32>, memref<1x170xf32>
  return
}

// -----

func.func @test_npuvector_rank0_transfer_read_write(%arg0: memref<f32>, %arg1: memref<f32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_rank0_transfer_read_write
  // CHECK: %[[SRC:.*]] = memref.subview %arg0[] [] [] : memref<f32> to memref<f32, strided<[]>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<f32>
  // CHECK: hivm.hir.load ins(%[[SRC]] : memref<f32, strided<[]>>) outs(%[[ALLOC]] : memref<f32>)
  // CHECK: %[[DEST:.*]] = memref.subview %arg1[] [] [] : memref<f32> to memref<f32, strided<[]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC]] : memref<f32>) outs(%[[DEST]] : memref<f32, strided<[]>>)
  %c1 = arith.constant 1 : index
  %pad = arith.constant 0.0 : f32
  %vec = npuvector.transfer_read %arg0[], %pad : memref<f32>, !npuvector.f32
  // The extra index is tile/VF metadata and is not a coordinate of the rank-0 memref.
  npuvector.transfer_write %vec, %arg1[%c1] : !npuvector.f32, memref<f32>
  return
}

// -----

func.func @test_npuvector_rank0_transfer_write_alloc_root(%arg0: memref<f16>, %arg1: memref<f16>, %out: memref<f16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_rank0_transfer_write_alloc_root
  // CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<f16>
  // CHECK: %[[INIT_VIEW:.*]] = memref.subview %[[SCRATCH]][] [] [] : memref<f16> to memref<f16, strided<[]>>
  // CHECK: hivm.hir.vbrc
  // CHECK-SAME: outs(%[[INIT_VIEW]] : memref<f16, strided<[]>>)
  // CHECK-NOT: hivm.hir.store {{.*}} outs(%[[SCRATCH]] : memref<f16>)
  // CHECK: hivm.hir.load {{.*}} outs(%{{.*}} : memref<f16>)
  // CHECK: hivm.hir.load {{.*}} outs(%{{.*}} : memref<f16>)
  // CHECK: %[[ADD_VIEW:.*]] = memref.subview %[[SCRATCH]][] [] [] : memref<f16> to memref<f16, strided<[]>>
  // CHECK: hivm.hir.vadd
  // CHECK-SAME: outs(%[[ADD_VIEW]] : memref<f16, strided<[]>>)
  // CHECK-NOT: hivm.hir.store {{.*}} outs(%[[SCRATCH]] : memref<f16>)
  // CHECK: %[[RESULT_VIEW:.*]] = memref.subview %[[SCRATCH]][] [] [] : memref<f16> to memref<f16, strided<[]>>
  // CHECK: %[[OUT_VIEW:.*]] = memref.subview %arg2[] [] [] : memref<f16> to memref<f16, strided<[]>>
  // CHECK: hivm.hir.store ins(%[[RESULT_VIEW]] : memref<f16, strided<[]>>) outs(%[[OUT_VIEW]] : memref<f16, strided<[]>>)
  %scratch = memref.alloc() : memref<f16>
  %pad = arith.constant 0.0 : f16
  %init = npuvector.broadcast %pad : f16 to !npuvector.f16
  npuvector.transfer_write %init, %scratch[] : !npuvector.f16, memref<f16>
  %lhs = npuvector.transfer_read %arg0[], %pad : memref<f16>, !npuvector.f16
  %rhs = npuvector.transfer_read %arg1[], %pad : memref<f16>, !npuvector.f16
  %sum = arith.addf %lhs, %rhs : !npuvector.f16
  npuvector.transfer_write %sum, %scratch[] : !npuvector.f16, memref<f16>
  %result = npuvector.transfer_read %scratch[], %pad : memref<f16>, !npuvector.f16
  npuvector.transfer_write %result, %out[] : !npuvector.f16, memref<f16>
  return
}

// -----

func.func @test_npuvector_rank0_reduction(%arg0: memref<16xf32>, %arg1: memref<f32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_rank0_reduction
  // CHECK: %[[SRC:.*]] = memref.subview %arg0[0] [16] [1] : memref<16xf32> to memref<16xf32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xf32>
  // CHECK: hivm.hir.load ins(%[[SRC]] : memref<16xf32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xf32>)
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[ALLOC_READ]] : memref<16xf32>) outs(%[[ALLOC_RED]] : memref<1xf32>) reduce_dims = [0]
  // CHECK: %[[COLLAPSED:.*]] = memref.collapse_shape %[[ALLOC_RED]] [] : memref<1xf32> into memref<f32>
  // CHECK: %[[DEST:.*]] = memref.subview %arg1[] [] [] : memref<f32> to memref<f32, strided<[]>>
  // CHECK: hivm.hir.store ins(%[[COLLAPSED]] : memref<f32>) outs(%[[DEST]] : memref<f32, strided<[]>>)
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %vec = npuvector.transfer_read %arg0[%c0], %pad : memref<16xf32>, !npuvector<16xf32>
  %sum = npuvector.reduction <add>, %vec {reduction_dims = array<i64: 0>} : !npuvector<16xf32> into !npuvector.f32
  npuvector.transfer_write %sum, %arg1[] : !npuvector.f32, memref<f32>
  return
}

// -----

func.func @test_npuvector_bitcast_dynamic_shape(%arg0: memref<128xbf16>, %arg1: memref<128xi16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_bitcast_dynamic_shape
  // CHECK: %[[C1_I16:.*]] = arith.constant 1 : i16
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C128:.*]] = arith.constant 128 : index
  // CHECK: scf.for %[[ARG2:.*]] = %[[C0]] to %[[C128]] step %[[C128]] {
  // CHECK:   %[[SUBVIEW0:.*]] = memref.subview %arg0[0] [%[[C128]]] [1] : memref<128xbf16> to memref<?xbf16, strided<[1]>>
  // CHECK:   %[[ALLOC0:.*]] = memref.alloc(%[[C128]]) : memref<?xbf16>
  // CHECK:   annotation.mark %[[ALLOC0]] {buffer_size_in_byte = 8192 : index}
  // CHECK:   hivm.hir.load ins(%[[SUBVIEW0]] : memref<?xbf16, strided<[1]>>) outs(%[[ALLOC0]] : memref<?xbf16>) init_out_buffer = false
  // CHECK:   %[[BC:.*]] = hivm.hir.bitcast %[[ALLOC0]] : memref<?xbf16> -> memref<?xi16>
  // CHECK:   %[[ALLOC1:.*]] = memref.alloc(%[[C128]]) : memref<?xi16>
  // CHECK:   annotation.mark %[[ALLOC1]] {buffer_size_in_byte = 8192 : index}
  // CHECK:   hivm.hir.vbrc ins(%[[C1_I16]] : i16) outs(%[[ALLOC1]] : memref<?xi16>)
  // CHECK:   %[[ALLOC2:.*]] = memref.alloc(%[[C128]]) : memref<?xi16>
  // CHECK:   hivm.hir.vand ins(%[[BC]], %[[ALLOC1]] : memref<?xi16>, memref<?xi16>) outs(%[[ALLOC2]] : memref<?xi16>)
  // CHECK:   %[[SUBVIEW1:.*]] = memref.subview %arg1[0] [%[[C128]]] [1] : memref<128xi16> to memref<?xi16, strided<[1]>>
  // CHECK:   hivm.hir.store ins(%[[ALLOC2]] : memref<?xi16>) outs(%[[SUBVIEW1]] : memref<?xi16, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  %c4096 = arith.constant 4096 : index
  %dim = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = npuvector.transfer_read %arg0[%c0][%dim][%c4096], %cst : memref<128xbf16>, !npuvector<?xbf16>
    %1 = npuvector.bitcast %0 : !npuvector<?xbf16> to !npuvector<?xi16>
    %2 = arith.constant 1 : i16
    %3 = npuvector.broadcast %2[%dim] [%c4096] : i16 to !npuvector<?xi16>
    %res = arith.andi %1, %3 : !npuvector<?xi16>
    npuvector.transfer_write %res, %arg1[%c0] : !npuvector<?xi16>, memref<128xi16>
  }
  return
}

// -----

func.func @test_npuvector_broadcast_index_cast_index(%arg0 : memref<?xi64>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_npuvector_broadcast_index_cast_index
  // CHECK: %[[C16:.*]] = arith.constant 16 : index
  // CHECK: %[[C0_I64:.*]] = arith.constant 0 : i64
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[C16]]) : memref<?xi64>
  // CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 32768 : index} : memref<?xi64>
  // CHECK: hivm.hir.vbrc ins(%[[C0_I64]] : i64) outs(%[[ALLOC]] : memref<?xi64>)
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %arg0[0] [%[[C16]]] [1] : memref<?xi64> to memref<?xi64, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC]] : memref<?xi64>) outs(%[[SUBVIEW]] : memref<?xi64, {{.*}}>)
  %1 = arith.constant 0 : index
  %5 = arith.constant 0 : index
  %2 = arith.constant 16 : index
  %c4096 = arith.constant 4096 : index
  %3 = npuvector.broadcast %1[%2] [%c4096] : index to !npuvector<?xindex>
  %4 = npuvector.index_cast %3 : !npuvector<?xindex> to !npuvector<?xi64>
  npuvector.transfer_write %4, %arg0[%5] : !npuvector<?xi64>, memref<?xi64>
  return
}

// -----

func.func @test_npuvector_transfer_write_zero_output(%arg0: memref<1xi32>, %arg1: memref<1xi32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c1 = arith.constant 1 : index
  %c1_0 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1_1 = arith.constant 1 : index
  %c1_2 = arith.constant 1 : index
  %c0_3 = arith.constant 0 : index
  %c1_4 = arith.constant 1 : index
  scf.for %arg2 = %c0_3 to %c1_4 step %c1_4 {
    scf.for %arg3 = %c0_3 to %c1_4 step %c1_4 {
      %c40 = arith.constant 40 : index
      %0 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%arg2, %arg3)
      %1 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%0)[%c1_0]
      %c1_5 = arith.constant 1 : index
      %2 = arith.addi %0, %c1_5 : index
      %3 = affine.apply affine_map<(d0)[s0] -> (d0 * s0)>(%2)[%c1_0]
      %c1_6 = arith.constant 1 : index
      %4 = affine.min affine_map<(d0, d1) -> (d0, d1)>(%3, %c1_6)
      %c4096 = arith.constant 4096 : index
      %5 = arith.subi %4, %1 : index
      %6 = arith.minsi %5, %c4096 : index
      %c1000_i32 = arith.constant 1000 : i32
      %collapse_shape = memref.collapse_shape %arg0 [] : memref<1xi32> into memref<i32>
      %collapse_shape_7 = memref.collapse_shape %arg1 [] : memref<1xi32> into memref<i32>
      %7 = memref.load %collapse_shape[] : memref<i32>
      %8 = npuvector.broadcast %7[%6] [%c4096] : i32 to !npuvector<?xi32>
      %9 = npuvector.broadcast %c1000_i32[%6] [%c4096] : i32 to !npuvector<?xi32>
      %10 = arith.addi %8, %9 : !npuvector<?xi32>
      npuvector.transfer_write %10, %collapse_shape_7[] : !npuvector<?xi32>, memref<i32>
    }
  }
  return
}

// -----

func.func @test_transfer_write_alloc_root_rewrites_later_broadcast_use(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %out: memref<1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK-LABEL: func.func @test_transfer_write_alloc_root_rewrites_later_broadcast_use
  // CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: %[[SCRATCH_VIEW:.*]] = memref.subview %[[SCRATCH]][0] [1] [1] : memref<1xf32> to memref<1xf32, {{.*}}>
  // CHECK: hivm.hir.vmul
  // CHECK-SAME: outs(%[[SCRATCH_VIEW]] : memref<1xf32, {{.*}}>)
  // CHECK-NOT: hivm.hir.vbrc
  // CHECK: hivm.hir.vmul ins(%[[SCRATCH_VIEW]], {{.*}} : memref<1xf32, {{.*}}>, memref<1xf32>)
  // CHECK: hivm.hir.store
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %pad = arith.constant 0.000000e+00 : f32
  %scratch = memref.alloc() : memref<1xf32>
  %lhs = npuvector.transfer_read %arg0[%c0] [%c1] [%c1], %pad : memref<1xf32>, !npuvector<1xf32>
  %rhs = npuvector.transfer_read %arg1[%c0] [%c1] [%c1], %pad : memref<1xf32>, !npuvector<1xf32>
  %mul = arith.mulf %lhs, %rhs : !npuvector<1xf32>
  npuvector.transfer_write %mul, %scratch[%c0] : !npuvector<1xf32>, memref<1xf32>
  %bcast = npuvector.broadcast %mul[%c1] [%c1] : !npuvector<1xf32> to !npuvector<1xf32>
  %outv = arith.mulf %bcast, %lhs : !npuvector<1xf32>
  npuvector.transfer_write %outv, %out[%c0] : !npuvector<1xf32>, memref<1xf32>
  return
}

// -----

// Helper outlined vector function that consumes a scratch allocation so the
// forwarded buffer remains live. It must be defined before its call site.
func.func private @vf_npu_touch_scratch(%scratch: memref<64xf32>, %touch: memref<1xf32>) attributes {hivm.vector_function, no_inline} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.000000e+00 : f32
  %v = npuvector.transfer_read %scratch[%c0][%c1][%c64], %pad : memref<64xf32>, !npuvector<1xf32>
  npuvector.transfer_write %v, %touch[%c0] : !npuvector<1xf32>, memref<1xf32>
  return
}

// -----

// Forward a partial alloc-to-alloc copy with matching shapes.
// CHECK-LABEL: func.func @test_forward_alloc_partial_same_shape
func.func @test_forward_alloc_partial_same_shape(%arg0: memref<64xf32>, %out: memref<1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[SRC:.*]] = memref.alloc() : memref<64xf32>
  // CHECK-NOT: memref.alloc() : memref<64xf32>
  // CHECK: memref.subview %arg0[0] [64] [1]
  // CHECK: hivm.hir.load ins(%{{.*}} : memref<64xf32{{.*}}>) outs(%{{.*}} : memref<64xf32{{.*}}>)
  // CHECK-NOT: hivm.hir.store ins({{.*}} : memref<64xf32>) outs({{.*}} : memref<64xf32>)
  // CHECK: hivm.hir.store ins({{.*}} : memref<1xf32{{.*}}>) outs({{.*}} : memref<1xf32{{.*}}>)
  %alloc = memref.alloc() : memref<64xf32>
  %alloc_4 = memref.alloc() : memref<64xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.000000e+00 : f32
  %v_in = npuvector.transfer_read %arg0[%c0][%c64][%c1], %pad : memref<64xf32>, !npuvector<64xf32>
  npuvector.transfer_write %v_in, %alloc_4[%c0] : !npuvector<64xf32>, memref<64xf32>
  %1 = npuvector.transfer_read %alloc_4[%c0][%c1][%c64], %pad : memref<64xf32>, !npuvector<1xf32>
  npuvector.transfer_write %1, %alloc[%c0] : !npuvector<1xf32>, memref<64xf32>
  %2 = npuvector.transfer_read %alloc[%c0][%c1][%c64], %pad : memref<64xf32>, !npuvector<1xf32>
  npuvector.transfer_write %2, %out[%c0] : !npuvector<1xf32>, memref<1xf32>
  return
}

// -----

// Forward a partial alloc-to-alloc copy from outside a loop to a loop-local write.
// CHECK-LABEL: func.func @test_forward_alloc_partial_same_shape_in_loop
func.func @test_forward_alloc_partial_same_shape_in_loop(%arg0: memref<64xf32>, %out: memref<1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[SRC:.*]] = memref.alloc() : memref<64xf32>
  // CHECK-NOT: memref.alloc() : memref<64xf32>
  // CHECK: memref.subview %arg0[0] [64] [1]
  // CHECK: hivm.hir.load ins(%{{.*}} : memref<64xf32{{.*}}>) outs(%{{.*}} : memref<64xf32{{.*}}>)
  // CHECK: scf.for
  // CHECK-NOT: hivm.hir.store ins({{.*}} : memref<64xf32>) outs({{.*}} : memref<64xf32>)
  // CHECK: hivm.hir.store ins({{.*}} : memref<1xf32{{.*}}>) outs({{.*}} : memref<1xf32{{.*}}>)
  %alloc = memref.alloc() : memref<64xf32>
  %alloc_4 = memref.alloc() : memref<64xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.000000e+00 : f32
  %v_in = npuvector.transfer_read %arg0[%c0][%c64][%c1], %pad : memref<64xf32>, !npuvector<64xf32>
  npuvector.transfer_write %v_in, %alloc_4[%c0] : !npuvector<64xf32>, memref<64xf32>
  %1 = npuvector.transfer_read %alloc_4[%c0][%c1][%c64], %pad : memref<64xf32>, !npuvector<1xf32>
  scf.for %iv = %c0 to %c1 step %c1 {
    %c0_in = arith.constant 0 : index
    npuvector.transfer_write %1, %alloc[%c0_in] : !npuvector<1xf32>, memref<64xf32>
    %2 = npuvector.transfer_read %alloc[%c0_in][%c1][%c64], %pad : memref<64xf32>, !npuvector<1xf32>
    npuvector.transfer_write %2, %out[%c0_in] : !npuvector<1xf32>, memref<1xf32>
  }
  return
}

// -----

// Forward between allocations with different ranks but the same element count.
// CHECK-LABEL: func.func @test_forward_alloc_diff_rank
func.func @test_forward_alloc_diff_rank(%arg0: memref<8xf32>, %out: memref<2x4xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: memref.alloc() : memref<8xf32>
  // CHECK-NOT: memref.alloc() : memref<2x4xf32>
  // CHECK: memref.subview %arg0[0] [8] [1]
  // CHECK: hivm.hir.load ins(%{{.*}} : memref<8xf32{{.*}}>) outs(%{{.*}} : memref<8xf32{{.*}}>)
  // CHECK: memref.expand_shape %{{.*}} output_shape [2, 4] : memref<8xf32> into memref<2x4xf32>
  // CHECK-NOT: hivm.hir.store ins({{.*}} : memref<2x4xf32>) outs({{.*}} : memref<2x4xf32>)
  // CHECK: hivm.hir.store ins({{.*}} : memref<2x4xf32{{.*}}>) outs({{.*}} : memref<2x4xf32{{.*}}>)
  %src = memref.alloc() : memref<8xf32>
  %dst = memref.alloc() : memref<2x4xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 0.000000e+00 : f32
  %v_in = npuvector.transfer_read %arg0[%c0][%c8][%c1], %pad : memref<8xf32>, !npuvector<8xf32>
  npuvector.transfer_write %v_in, %src[%c0] : !npuvector<8xf32>, memref<8xf32>
  %v = npuvector.transfer_read %src[%c0][%c8][%c1], %pad : memref<8xf32>, !npuvector<8xf32>
  npuvector.transfer_write %v, %dst[%c0, %c0] : !npuvector<8xf32>, memref<2x4xf32>
  %v2 = npuvector.transfer_read %dst[%c0, %c0][%c2, %c4][%c4, %c1], %pad : memref<2x4xf32>, !npuvector<2x4xf32>
  npuvector.transfer_write %v2, %out[%c0, %c0] : !npuvector<2x4xf32>, memref<2x4xf32>
  return
}

// -----

// Forward the contiguous prefix when the source allocation is larger than the destination.
// CHECK-LABEL: func.func @test_forward_alloc_src_larger
func.func @test_forward_alloc_src_larger(%arg0: memref<128xf32>, %out: memref<1xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: memref.alloc() : memref<128xf32>
  // CHECK: memref.subview %arg0[0] [128] [1]
  // CHECK-NOT: memref.alloc() : memref<64xf32>
  // CHECK: hivm.hir.load ins(%{{.*}} : memref<128xf32{{.*}}>) outs(%{{.*}} : memref<128xf32{{.*}}>)
  // CHECK-NOT: hivm.hir.store ins({{.*}} : memref<64xf32>) outs({{.*}} : memref<64xf32>)
  // CHECK: call @vf_npu_touch_scratch(%{{.*}}, %{{.*}}) {hivm.vector_function, no_inline}
  %src = memref.alloc() : memref<128xf32>
  %dst = memref.alloc() : memref<64xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %pad = arith.constant 0.000000e+00 : f32
  %v_in = npuvector.transfer_read %arg0[%c0][%c128][%c1], %pad : memref<128xf32>, !npuvector<128xf32>
  npuvector.transfer_write %v_in, %src[%c0] : !npuvector<128xf32>, memref<128xf32>
  %v = npuvector.transfer_read %src[%c0][%c64][%c1], %pad : memref<128xf32>, !npuvector<64xf32>
  npuvector.transfer_write %v, %dst[%c0] : !npuvector<64xf32>, memref<64xf32>
  func.call @vf_npu_touch_scratch(%dst, %out) {hivm.vector_function, no_inline} : (memref<64xf32>, memref<1xf32>) -> ()
  return
}

// -----

// Fuse a same-block elementwise producer into a scratch allocation.
// CHECK-LABEL: func.func @test_fuse_producer_same_block_gm_to_alloc
func.func @test_fuse_producer_same_block_gm_to_alloc(%arg0: memref<8xf32>, %out: memref<8xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<8xf32>
  // CHECK: memref.subview %arg0[0] [8] [1]
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<8xf32>
  // CHECK: hivm.hir.load ins(%{{.*}} : {{.*}}) outs(%[[UB0]] : {{.*}})
  // CHECK: hivm.hir.vadd ins(%[[UB0]], {{.*}} : memref<8xf32>, f32) outs(%{{.*}} : memref<8xf32{{.*}}>)
  // CHECK-NOT: hivm.hir.store ins(%{{.*}} : memref<8xf32>) outs(%{{.*}} : memref<8xf32>)
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<8xf32{{.*}}>) outs(%{{.*}} : memref<8xf32{{.*}}>)
  %scratch = memref.alloc() : memref<8xf32>
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %pad = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %v0 = npuvector.transfer_read %arg0[%c0][%c8][%c1], %pad : memref<8xf32>, !npuvector<8xf32>
  %vb = npuvector.broadcast %one : f32 to !npuvector<8xf32>
  %v1 = arith.addf %v0, %vb : !npuvector<8xf32>
  npuvector.transfer_write %v1, %scratch[%c0] : !npuvector<8xf32>, memref<8xf32>
  %v2 = npuvector.transfer_read %scratch[%c0][%c8][%c1], %pad : memref<8xf32>, !npuvector<8xf32>
  npuvector.transfer_write %v2, %out[%c0] : !npuvector<8xf32>, memref<8xf32>
  return
}

// -----

// Sink a cross-block elementwise producer into a loop-local scratch allocation.
// CHECK-LABEL: func.func @test_fuse_producer_sink_cross_block
func.func @test_fuse_producer_sink_cross_block(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %out: memref<8xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: memref.subview %arg0[0] [8] [1]
  // CHECK: hivm.hir.load ins(%{{.*}} : {{.*}}) outs(%{{.*}} : memref<8xf32>)
  // CHECK: memref.subview %arg1[0] [8] [1]
  // CHECK: hivm.hir.load ins(%{{.*}} : {{.*}}) outs(%{{.*}} : memref<8xf32>)
  // CHECK: scf.for
  // CHECK: %[[SCRATCH:.*]] = memref.alloc() : memref<8xf32>
  // CHECK: hivm.hir.vmul ins(%{{.*}}, %{{.*}} : memref<8xf32>, memref<8xf32>) outs(%{{.*}} : memref<8xf32{{.*}}>)
  // CHECK-NOT: hivm.hir.store ins(%{{.*}} : memref<8xf32>) outs(%{{.*}} : memref<8xf32>)
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<8xf32{{.*}}>) outs(%{{.*}} : memref<8xf32{{.*}}>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 0.000000e+00 : f32
  %lhs = npuvector.transfer_read %arg0[%c0][%c8][%c1], %pad : memref<8xf32>, !npuvector<8xf32>
  %rhs = npuvector.transfer_read %arg1[%c0][%c8][%c1], %pad : memref<8xf32>, !npuvector<8xf32>
  %mul = arith.mulf %lhs, %rhs : !npuvector<8xf32>
  scf.for %i = %c0 to %c1 step %c1 {
    %scratch = memref.alloc() : memref<8xf32>
    npuvector.transfer_write %mul, %scratch[%c0] : !npuvector<8xf32>, memref<8xf32>
    %v = npuvector.transfer_read %scratch[%c0][%c8][%c1], %pad : memref<8xf32>, !npuvector<8xf32>
    npuvector.transfer_write %v, %out[%c0] : !npuvector<8xf32>, memref<8xf32>
  }
  return
}

// -----

// Sink a dense splat constant into a loop-local allocation.
// CHECK-LABEL: func.func @test_fuse_producer_sink_dense_constant
func.func @test_fuse_producer_sink_dense_constant(%out: memref<512xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: scf.for
  // CHECK: memref.alloc() : memref<512xf32>
  // CHECK: hivm.hir.vbrc ins({{.*}} : f32) outs(%{{.*}} : memref<512xf32{{.*}}>)
  // CHECK-NOT: hivm.hir.store ins({{.*}} : memref<512xf32>) outs({{.*}} : memref<512xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c512 = arith.constant 512 : index
  %pad = arith.constant 0.000000e+00 : f32
  %cst = arith.constant dense<1.0> : !npuvector<512xf32>
  scf.for %i = %c0 to %c1 step %c1 {
    %alloc_2 = memref.alloc() : memref<512xf32>
    npuvector.transfer_write %cst, %alloc_2[%c0] : !npuvector<512xf32>, memref<512xf32>
    %c1_in = arith.constant 1 : index
    %v = npuvector.transfer_read %alloc_2[%c0][%c512][%c1_in], %pad : memref<512xf32>, !npuvector<512xf32>
    npuvector.transfer_write %v, %out[%c0] : !npuvector<512xf32>, memref<512xf32>
  }
  return
}

// -----

// Lower shared dense constants directly into each loop-local allocation.
// CHECK-LABEL: func.func @test_dense_constant_transfer_write_shared
func.func @test_dense_constant_transfer_write_shared() attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: scf.for
  // CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<16xf32>
  // CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<16xf32>
  // CHECK: %[[VIEW0:.*]] = memref.subview %[[ALLOC0]][0] [16] [1]
  // CHECK: hivm.hir.vbrc ins({{.*}} : f32) outs(%[[VIEW0]] : memref<16xf32{{.*}}>)
  // CHECK: %[[VIEW1:.*]] = memref.subview %[[ALLOC1]][0] [16] [1]
  // CHECK: hivm.hir.vbrc ins({{.*}} : f32) outs(%[[VIEW1]] : memref<16xf32{{.*}}>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<0.0> : !npuvector<16xf32>
  scf.for %i = %c0 to %c1 step %c1 {
    %alloc0 = memref.alloc() : memref<16xf32>
    %alloc1 = memref.alloc() : memref<16xf32>
    npuvector.transfer_write %cst, %alloc0[%c0] : !npuvector<16xf32>, memref<16xf32>
    npuvector.transfer_write %cst, %alloc1[%c0] : !npuvector<16xf32>, memref<16xf32>
  }
  return
}

// -----

// npuvector.extract_slice lowers to a rank-reduced memref.subview on the source buffer.
// CHECK-LABEL: func.func @test_npuvector_extract_slice
func.func @test_npuvector_extract_slice(%arg0 : memref<8x4x2xf32>, %arg1 : memref<8x4xf32>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[SRC:.*]] = memref.alloc() : memref<8x4x2xf32>
  // CHECK: hivm.hir.load
  // CHECK: %[[SLICE:.*]] = memref.subview %[[SRC]][0, 0, 1] [8, 4, 1] [1, 1, 1] : memref<8x4x2xf32> to memref<8x4xf32, {{.*}}>
  // CHECK: hivm.hir.store ins(%[[SLICE]]
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %pad = arith.constant 0.000000e+00 : f32
  %tile = npuvector.transfer_read %arg0[%c0, %c0, %c0] [%c8, %c4, %c2] [%c8, %c4, %c2], %pad : memref<8x4x2xf32>, !npuvector<8x4x2xf32>
  %slice = npuvector.extract_slice %tile [%c0, %c0, %c1] [%c8, %c4, %c1] [%c1, %c1, %c1] {keep_dims = array<i64: 0, 1>} : !npuvector<8x4x2xf32> to !npuvector<8x4xf32>
  npuvector.transfer_write %slice, %arg1[%c0, %c0] : !npuvector<8x4xf32>, memref<8x4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @test_npuvector_extract_slice_preserves_dynamic_kept_dims
func.func @test_npuvector_extract_slice_preserves_dynamic_kept_dims(%arg0 : memref<32x16x1x2xbf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  // CHECK: %[[TILE:.*]] = memref.alloc({{.*}}) : memref<32x?x?x2xbf16>
  // CHECK: %[[SLICE:.*]] = memref.subview %[[TILE]][0, 0, 0, 0] [32, %{{.*}}, %{{.*}}, 1] [1, 1, 1, 1] : memref<32x?x?x2xbf16> to memref<32x?x?xbf16, {{.*}}>
  // CHECK: %[[CAST:.*]] = memref.alloc({{.*}}) : memref<32x?x?xf32>
  // CHECK: hivm.hir.vcast ins(%[[SLICE]] : memref<32x?x?xbf16, {{.*}}>) outs(%[[CAST]] : memref<32x?x?xf32>)
  // CHECK: %[[MUL:.*]] = memref.alloc({{.*}}) : memref<32x?x?xf32>
  // CHECK: hivm.hir.vmul ins(%[[CAST]], %[[CAST]] : memref<32x?x?xf32>, memref<32x?x?xf32>) outs(%[[MUL]] : memref<32x?x?xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %pad = arith.constant 0.000000e+00 : bf16
  %tile = npuvector.transfer_read %arg0[%c0, %c0, %c0, %c0] [%c32, %c16, %c1, %c2] [%c32, %c16, %c1, %c2], %pad : memref<32x16x1x2xbf16>, !npuvector<32x?x?x2xbf16>
  %slice = npuvector.extract_slice %tile [%c0, %c0, %c0, %c0] [%c32, %c16, %c1, %c1] [%c1, %c1, %c1, %c1] {keep_dims = array<i64: 0, 1, 2>} : !npuvector<32x?x?x2xbf16> to !npuvector<32x?x?xbf16>
  %cast = npuvector.extf %slice : !npuvector<32x?x?xbf16> to !npuvector<32x?x?xf32>
  %mul = arith.mulf %cast, %cast : !npuvector<32x?x?xf32>
  return
}
