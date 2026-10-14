// RUN: akg-opt %s -convert-vector-to-pypto | FileCheck %s

func.func @test_elementwise(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK-LABEL: func.func @test_elementwise(
  // CHECK-SAME: %arg0: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: %arg2: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 2>, emitpy.pypto_kernel}
  // CHECK: %[[V0:.*]] = emitpy.call_opaque "pl.load"(%arg0)
  // CHECK: %[[V1:.*]] = emitpy.call_opaque "pl.load"
  // CHECK: %[[SUM:.*]] = emitpy.call_opaque "pl.add"(%[[V0]], %[[V1]])
  // CHECK: %[[EXP:.*]] = emitpy.call_opaque "pl.exp"(%[[SUM]])
  // CHECK: emitpy.call_opaque "pl.store"(%[[EXP]], %arg2)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %sum = arith.addf %v0, %v1 : vector<32x32xf32>
  %exp = math.exp %sum : vector<32x32xf32>
  vector.transfer_write %exp, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_npuvector_ops(%arg0 : !npuvector<8x16xf32>, %arg1 : f32) -> f32 {
  // CHECK-LABEL: func.func @test_npuvector_ops(
  // CHECK-SAME: %arg0: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: %arg1: !emitpy.opaque<"pl.Scalar{{[^"]*}}pl.FP32
  // CHECK-SAME: %arg2: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 2>, emitpy.pypto_kernel}
  // Tile parameters cross the kernel boundary as tensors and are loaded
  // back into tiles at the entry.
  // CHECK: %[[V0:.*]] = emitpy.call_opaque "pl.load"(%arg0)
  // CHECK: %[[T:.*]] = emitpy.call_opaque "pl.transpose"(%[[V0]])
  // A broadcast scalar is carried by a zero tile; the scalar form of pl.add
  // broadcasts it onto the carrier.
  // CHECK: %[[CARRIER:.*]] = emitpy.call_opaque "pl.tile.full"()
  // CHECK: %[[B:.*]] = emitpy.call_opaque "pl.add"(%[[CARRIER]], %arg1)
  // CHECK: %[[M:.*]] = emitpy.call_opaque "pl.mul"(%[[T]], %[[B]])
  // A full reduction row-reduces on device, then reads out and combines the
  // surviving rows with scalar-form adds.
  // CHECK: %[[ROW_TMP:.*]] = emitpy.call_opaque "pl.create_tile"()
  // CHECK: %[[ROW:.*]] = emitpy.call_opaque "pl.row_sum"(%[[M]], %[[ROW_TMP]])
  // CHECK: %[[R0:.*]] = emitpy.call_opaque "pl.read"(%[[ROW]])
  // CHECK: %[[R1:.*]] = emitpy.call_opaque "pl.read"(%[[ROW]])
  // CHECK: emitpy.call_opaque "pl.add"(%[[R0]], %[[R1]])
  // The scalar result is written straight into the [1] output tensor.
  // CHECK: emitpy.call_opaque "pl.write"(%arg2, {{.*}})

  %t = npuvector.transpose %arg0, [1, 0] : !npuvector<8x16xf32> to !npuvector<16x8xf32>
  %b = npuvector.broadcast %arg1 : f32 to !npuvector<16x8xf32>
  %m = arith.mulf %t, %b : !npuvector<16x8xf32>
  %r = npuvector.reduction <add>, %m : !npuvector<16x8xf32> into f32
  return %r : f32
}

// -----

func.func @test_partial_reduction_and_casts(%arg0 : !npuvector<4x8xi32>) -> !npuvector<4xi32> {
  // CHECK-LABEL: func.func @test_partial_reduction_and_casts(
  // CHECK-SAME: %arg0: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.INT32
  // CHECK-SAME: %arg1: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.INT32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel}
  // CHECK: %[[V0:.*]] = emitpy.call_opaque "pl.load"(%arg0)
  // Partial reductions consume a scratch tile.
  // CHECK: %[[TMP:.*]] = emitpy.call_opaque "pl.create_tile"()
  // CHECK: %[[R:.*]] = emitpy.call_opaque "pl.row_max"(%[[V0]], %[[TMP]])
  // CHECK: %[[E:.*]] = emitpy.call_opaque "pl.cast"(%[[R]]) {{.*}}"pl.INT64"
  // CHECK: %[[W:.*]] = emitpy.call_opaque "pl.cast"(%[[E]]) {{.*}}"pl.INT32"
  // The [4] output row is shorter than a 32-byte unit, so it is written
  // element by element.
  // CHECK: %[[E0:.*]] = emitpy.call_opaque "pl.read"(%[[W]])
  // CHECK: emitpy.call_opaque "pl.write"(%arg1, %[[E0]])

  %r = npuvector.reduction <maxsi>, %arg0 {reduction_dims = array<i64: 1>} : !npuvector<4x8xi32> into !npuvector<4xi32>
  %e = npuvector.extsi %r : !npuvector<4xi32> to !npuvector<4xi64>
  %w = npuvector.trunci %e : !npuvector<4xi64> to !npuvector<4xi32>
  return %w : !npuvector<4xi32>
}

// -----

func.func @test_rank1_full_reduction(%arg0 : !npuvector<3072xf32>) -> f32 {
  // CHECK-LABEL: func.func @test_rank1_full_reduction(
  // CHECK-SAME: %arg0: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: %arg1: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel}
  // CHECK: %[[V0:.*]] = emitpy.call_opaque "pl.load"(%arg0)
  // A rank-1 tile is viewed as [8, N/8] so the row reduction leaves an
  // allocatable [8, 1] tile -- a [1, N] view would reduce to [1, 1],
  // which the device cannot allocate.
  // CHECK: %[[R:.*]] = emitpy.call_opaque "pl.reshape"(%[[V0]])
  // CHECK: %[[ROW_TMP:.*]] = emitpy.call_opaque "pl.create_tile"()
  // CHECK: %[[ROW:.*]] = emitpy.call_opaque "pl.row_sum"(%[[R]], %[[ROW_TMP]])
  // The surviving row sums are read out and combined with scalar adds.
  // CHECK: %[[S0:.*]] = emitpy.call_opaque "pl.read"(%[[ROW]])
  // CHECK: %[[S1:.*]] = emitpy.call_opaque "pl.read"(%[[ROW]])
  // CHECK: %[[A:.*]] = emitpy.call_opaque "pl.add"(%[[S0]], %[[S1]])
  // CHECK: %[[S2:.*]] = emitpy.call_opaque "pl.read"(%[[ROW]])
  // CHECK: emitpy.call_opaque "pl.add"(%[[A]], %[[S2]])
  // The scalar result is written straight into the [1] output tensor.
  // CHECK: emitpy.call_opaque "pl.write"(%arg1, {{.*}})

  %r = npuvector.reduction <add>, %arg0 : !npuvector<3072xf32> into f32
  return %r : f32
}

// -----

func.func @test_select_alloc(%arg0 : vector<8xf32>, %arg1 : vector<8xf32>) -> memref<8xf32> {
  // CHECK-LABEL: func.func @test_select_alloc(
  // CHECK-SAME: %arg0: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: %arg1: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: %arg2: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 2>, emitpy.pypto_kernel}
  // CHECK: %[[V0:.*]] = emitpy.call_opaque "pl.load"(%arg0)
  // CHECK: %[[V1:.*]] = emitpy.call_opaque "pl.load"(%arg1)
  // Splat tile constants map onto pl.tile.full.
  // CHECK: %[[C:.*]] = emitpy.call_opaque "pl.tile.full"()
  // CHECK: %[[CMP:.*]] = emitpy.call_opaque "pl.cmp"(%[[V0]], %[[V1]])
  // pl.sel takes tiles and an inline scratch tile.
  // CHECK: %[[TMP:.*]] = emitpy.call_opaque "pl.create_tile"()
  // CHECK: %[[SEL:.*]] = emitpy.call_opaque "pl.sel"(%[[CMP]], %[[V0]], %[[C]], %[[TMP]])
  // The allocation became an appended output parameter, presented as a [1, 8]
  // tensor because tile stores need a 2D destination; the tile is reshaped
  // through the same view.
  // CHECK: %[[R2:.*]] = emitpy.call_opaque "pl.reshape"(%[[SEL]])
  // CHECK: emitpy.call_opaque "pl.store"(%[[R2]], %arg2)

  %c = arith.constant dense<1.0> : vector<8xf32>
  %cmp = arith.cmpf olt, %arg0, %arg1 : vector<8xf32>
  %sel = arith.select %cmp, %arg0, %c : vector<8xi1>, vector<8xf32>
  %buf = memref.alloc() : memref<8xf32>
  %c0 = arith.constant 0 : index
  vector.transfer_write %sel, %buf[%c0] : vector<8xf32>, memref<8xf32>
  return %buf : memref<8xf32>
}

// -----

func.func @test_spmd_tile_loop(%arg0 : memref<1024x1024xf32>, %arg1 : memref<1024x1024xf32>) {
  // CHECK-LABEL: func.func @test_spmd_tile_loop(
  // CHECK-SAME: %arg1: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel}
  // scf.parallel becomes an SPMD launch; the block argument is the core
  // index, computed against the entry-block literals.
  // CHECK: emitpy.spmd_scope core_num 64 {
  // CHECK: %[[ROW0:.*]] = emitpy.call_opaque "pl.mul"(%{{.+}}, %{{.+}}) : (index, index) -> index
  // scf.for becomes a pl.range tile loop.
  // CHECK: emitpy.for %{{.+}} to %{{.+}} step %{{.+}} : index index index {
  // CHECK: %[[ROW:.*]] = emitpy.call_opaque "pl.add"(%[[ROW0]], %{{.+}}) : (index, index) -> index
  // Loop-dependent offsets are referenced by operand index inside the list.
  // CHECK: emitpy.call_opaque "pl.load"(%arg0, %[[ROW]], %{{.+}}) {args = [0 : index, {{\[[0-9]+ : index, [0-9]+ : index\]}}, #emitpy.opaque<"{{\[8, 1024\]}}">
  // CHECK: emitpy.call_opaque "pl.store"(%{{.+}}, %arg1, %[[ROW]], %{{.+}}) {args = [0 : index, {{\[[0-9]+ : index, [0-9]+ : index\]}}, 1 : index]

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %cf0 = arith.constant 0.0 : f32
  scf.parallel (%blk) = (%c0) to (%c64) step (%c1) {
    %row0 = arith.muli %blk, %c16 : index
    scf.for %t = %c0 to %c2 step %c1 {
      %t8 = arith.muli %t, %c8 : index
      %row = arith.addi %row0, %t8 : index
      %v = vector.transfer_read %arg0[%row, %c0], %cf0 : memref<1024x1024xf32>, vector<8x1024xf32>
      vector.transfer_write %v, %arg1[%row, %c0] : vector<8x1024xf32>, memref<1024x1024xf32>
    }
  }
  return
}
