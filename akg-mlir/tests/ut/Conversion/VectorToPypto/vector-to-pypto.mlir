// RUN: akg-opt %s -convert-vector-to-pypto -convert-pypto-to-emitpy | FileCheck %s
// RUN: akg-opt %s -convert-vector-to-pypto | FileCheck %s --check-prefix=PYPTO
// RUN: akg-opt %s -convert-vector-to-pypto | akg-opt -convert-pypto-to-emitpy | FileCheck %s

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

  // After the first stage (`convert-vector-to-pypto` alone) the elementwise
  // dataflow stays in the pypto layer with the source result types; the
  // signature keeps its original memref form and the finalize/sweep steps
  // are left to `convert-pypto-to-emitpy`.
  // PYPTO-LABEL: func.func @test_elementwise(
  // PYPTO-SAME: %arg0: memref<32x32xf32>
  // PYPTO-SAME: %arg2: memref<32x32xf32>
  // PYPTO: %[[V0:.*]] = pypto.load %arg0[{{.*}}] : memref<32x32xf32> -> vector<32x32xf32>
  // PYPTO: %[[V1:.*]] = pypto.load %arg1[{{.*}}] : memref<32x32xf32> -> vector<32x32xf32>
  // PYPTO: %[[SUM:.*]] = pypto.add %[[V0]], %[[V1]] : vector<32x32xf32>, vector<32x32xf32> to vector<32x32xf32>
  // PYPTO: %[[EXP:.*]] = pypto.exp %[[SUM]] : vector<32x32xf32> to vector<32x32xf32>
  // PYPTO: pypto.store %[[EXP]], %arg2[{{.*}}] : vector<32x32xf32>, memref<32x32xf32>

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
  // The surviving rows combine as scalar arith, mapped onto the infix
  // addf pseudo-callee.
  // CHECK: emitpy.call_opaque "addf"(%[[R0]], %[[R1]])
  // The scalar result is written straight into the [1] output tensor.
  // CHECK: emitpy.call_opaque "pl.write"(%arg2, {{.*}})

  // The transpose view, the broadcast carrier, the elementwise ops, the
  // scratch tile and the reduction chain all sit in the pypto layer after
  // the first stage.
  // PYPTO-LABEL: func.func @test_npuvector_ops(
  // PYPTO: %[[T:.*]] = pypto.transpose %arg0, 0, 1 : !npuvector<8x16xf32> to !npuvector<16x8xf32>
  // PYPTO: %[[CARRIER:.*]] = pypto.tile_full "0.0" : !npuvector<16x8xf32>
  // PYPTO: %[[B:.*]] = pypto.add %[[CARRIER]], %arg1 : !npuvector<16x8xf32>, f32 to !npuvector<16x8xf32>
  // PYPTO: %[[M:.*]] = pypto.mul %[[T]], %[[B]] : !npuvector<16x8xf32>, !npuvector<16x8xf32> to !npuvector<16x8xf32>
  // The full row reduction works in the scratch tile and yields a scalar.
  // PYPTO: %[[ROW_TMP:.*]] = pypto.create_tile : <f32, [16, 8]>
  // PYPTO: %[[ROW:.*]] = pypto.reduction sum, row, %[[M]], %[[ROW_TMP]] : !npuvector<16x8xf32>, {{.*}} -> f32
  // The surviving rows are read out and combined with scalar arith.
  // PYPTO: %[[R0:.*]] = pypto.read %[[ROW]] [0, 0] : f32 -> f32
  // PYPTO: %[[R1:.*]] = pypto.read %[[ROW]] [1, 0] : f32 -> f32
  // PYPTO: arith.addf %[[R0]], %[[R1]] : f32
  // The scalar result goes straight into the [1] output tensor.
  // PYPTO: pypto.write %arg2[{{.*}}], {{.*}} : memref<1xf32>, f32

  %t = npuvector.transpose %arg0, [1, 0] : !npuvector<8x16xf32> to !npuvector<16x8xf32>
  %b = npuvector.broadcast %arg1 : f32 to !npuvector<16x8xf32>
  %m = arith.mulf %t, %b : !npuvector<16x8xf32>
  %r = npuvector.reduction <add>, %m : !npuvector<16x8xf32> into f32
  return %r : f32
}

// -----

func.func @test_scalar_broadcast_fusion(%arg0 : memref<1x8xf32>, %arg1 : memref<1x8xf32>, %arg2 : memref<1x8xf32>) {
  // CHECK-LABEL: func.func @test_scalar_broadcast_fusion(
  // CHECK-SAME: %arg0: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 2>, emitpy.pypto_kernel
  // A literal scalar broadcast feeding a comparison's rhs and a select's
  // true branch -- the relu-style where(a <= 0, 0, b) shape -- fuses into
  // the scalar operand form; no carrier tile is materialized.
  // CHECK-NOT: "pl.tile.full"
  // CHECK: %[[V0:.*]] = emitpy.call_opaque "pl.load"(%arg0)
  // CHECK: %[[V1:.*]] = emitpy.call_opaque "pl.load"(%arg1)
  // CHECK: %[[C:.*]] = emitpy.call_opaque "pl.tile.cmps"(%[[V0]]) {{.*}}pl.const(0.0, pl.FP32)
  // CHECK: emitpy.call_opaque "pl.tile.select"(%[[C]], %[[V1]]) {{.*}}pl.const(0.0, pl.FP32)
  // CHECK: emitpy.call_opaque "pl.store"

  // PYPTO-LABEL: func.func @test_scalar_broadcast_fusion(
  // PYPTO: %[[CST:.*]] = pypto.constant "0.0" : f32
  // PYPTO: %[[V0:.*]] = pypto.load %arg0[{{.*}}] : memref<1x8xf32> -> !npuvector<1x8xf32>
  // PYPTO: %[[V1:.*]] = pypto.load %arg1[{{.*}}] : memref<1x8xf32> -> !npuvector<1x8xf32>
  // PYPTO-NOT: pypto.tile_full
  // PYPTO: %[[CMP:.*]] = pypto.cmp {{ *}}le, %[[V0]], %[[CST]] : !npuvector<1x8xf32>, f32 -> !npuvector<1x8xi1>
  // PYPTO: pypto.select %[[CMP]], %[[CST]], %[[V1]] : !npuvector<1x8xi1>, f32, !npuvector<1x8xf32> to !npuvector<1x8xf32>
  // PYPTO: pypto.store {{.*}}, %arg2[{{.*}}]

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = npuvector.transfer_read %arg0[%c0, %c0], %cst : memref<1x8xf32>, !npuvector<1x8xf32>
  %v1 = npuvector.transfer_read %arg1[%c0, %c0], %cst : memref<1x8xf32>, !npuvector<1x8xf32>
  %b = npuvector.broadcast %cst : f32 to !npuvector<1x8xf32>
  %cmp = npuvector.cmpf ole, %v0, %b : !npuvector<1x8xf32> to !npuvector<1x8xi1>
  %sel = npuvector.select %cmp, %b, %v1 : !npuvector<1x8xi1>, !npuvector<1x8xf32>
  npuvector.transfer_write %sel, %arg2[%c0, %c0] : !npuvector<1x8xf32>, memref<1x8xf32>
  return
}

// -----

func.func @test_scalar_broadcast_no_fusion(%arg0 : memref<1x8xf32>, %arg1 : memref<1x8xf32>, %arg2 : memref<1x8xf32>) {
  // CHECK-LABEL: func.func @test_scalar_broadcast_no_fusion(
  // CHECK-SAME: %arg0: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 2>, emitpy.pypto_kernel
  // CHECK: %[[V0:.*]] = emitpy.call_opaque "pl.load"(%arg0)
  // CHECK: %[[V1:.*]] = emitpy.call_opaque "pl.load"(%arg1)
  // The broadcast sits on the comparison's lhs, so the fusion pattern does
  // not apply: the literal scalar is materialized on a full carrier tile and
  // the comparison stays in tile-vs-tile form.
  // CHECK: %[[CARRIER:.*]] = emitpy.call_opaque "pl.tile.full"()
  // CHECK: %[[C:.*]] = emitpy.call_opaque "pl.cmp"(%[[CARRIER]], %[[V0]])
  // CHECK: emitpy.call_opaque "pl.tile.select"(%[[C]], %[[CARRIER]], %[[V1]])
  // CHECK: emitpy.call_opaque "pl.store"

  // PYPTO-LABEL: func.func @test_scalar_broadcast_no_fusion(
  // PYPTO: %[[V0:.*]] = pypto.load %arg0[{{.*}}] : memref<1x8xf32> -> !npuvector<1x8xf32>
  // PYPTO: %[[V1:.*]] = pypto.load %arg1[{{.*}}] : memref<1x8xf32> -> !npuvector<1x8xf32>
  // PYPTO: %[[CARRIER:.*]] = pypto.tile_full "0.0" : !npuvector<1x8xf32>
  // PYPTO: %[[CMP:.*]] = pypto.cmp {{ *}}le, %[[CARRIER]], %[[V0]] : !npuvector<1x8xf32>, !npuvector<1x8xf32> -> !npuvector<1x8xi1>
  // PYPTO: pypto.select %[[CMP]], %[[CARRIER]], %[[V1]] : !npuvector<1x8xi1>, !npuvector<1x8xf32>, !npuvector<1x8xf32> to !npuvector<1x8xf32>
  // PYPTO: pypto.store {{.*}}, %arg2[{{.*}}]

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = npuvector.transfer_read %arg0[%c0, %c0], %cst : memref<1x8xf32>, !npuvector<1x8xf32>
  %v1 = npuvector.transfer_read %arg1[%c0, %c0], %cst : memref<1x8xf32>, !npuvector<1x8xf32>
  %b = npuvector.broadcast %cst : f32 to !npuvector<1x8xf32>
  %cmp = npuvector.cmpf ole, %b, %v0 : !npuvector<1x8xf32> to !npuvector<1x8xi1>
  %sel = npuvector.select %cmp, %b, %v1 : !npuvector<1x8xi1>, !npuvector<1x8xf32>
  npuvector.transfer_write %sel, %arg2[%c0, %c0] : !npuvector<1x8xf32>, memref<1x8xf32>
  return
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

  // The partial reduction, the dtype casts, the scalar reads, the scratch
  // tile and the element-wise [4] store all ride in the pypto layer after
  // the first stage.
  // PYPTO-LABEL: func.func @test_partial_reduction_and_casts(
  // PYPTO: %[[TMP:.*]] = pypto.create_tile : <i32, [4, 8]>
  // PYPTO: %[[R:.*]] = pypto.reduction max, row, %arg0, %[[TMP]] : !npuvector<4x8xi32>, {{.*}} -> !npuvector<4xi32>
  // PYPTO: %[[E:.*]] = pypto.cast %[[R]] : !npuvector<4xi32> to !npuvector<4xi64>
  // PYPTO: %[[W:.*]] = pypto.cast %[[E]] : !npuvector<4xi64> to !npuvector<4xi32>
  // The short output row is written element by element.
  // PYPTO: %[[E0:.*]] = pypto.read %[[W]] [0] : !npuvector<4xi32> -> i32
  // PYPTO: pypto.write %arg1[{{.*}}], %[[E0]] : memref<4xi32>, i32

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
  // The surviving row sums are read out and combined with scalar arith,
  // mapped onto the infix addf pseudo-callee.
  // CHECK: %[[S0:.*]] = emitpy.call_opaque "pl.read"(%[[ROW]])
  // CHECK: %[[S1:.*]] = emitpy.call_opaque "pl.read"(%[[ROW]])
  // CHECK: %[[A:.*]] = emitpy.call_opaque "addf"(%[[S0]], %[[S1]])
  // CHECK: %[[S2:.*]] = emitpy.call_opaque "pl.read"(%[[ROW]])
  // CHECK: emitpy.call_opaque "addf"(%[[A]], %[[S2]])
  // The scalar result is written straight into the [1] output tensor.
  // CHECK: emitpy.call_opaque "pl.write"(%arg1, {{.*}})

  // The [8, N/8] reshape view, the scratch tile, the reduction and the
  // scalar reads ride in the pypto layer after the first stage; the
  // combine chain keeps scalar arith.
  // PYPTO-LABEL: func.func @test_rank1_full_reduction(
  // PYPTO: pypto.reshape %arg0 : !npuvector<3072xf32> to !pypto.tile<f32, [8, 384]>
  // PYPTO: %[[ROW_TMP:.*]] = pypto.create_tile : <f32, [8, 384]>
  // PYPTO: %[[ROW:.*]] = pypto.reduction sum, row, {{.*}}, %[[ROW_TMP]] : {{.*}} -> f32
  // PYPTO: %[[S0:.*]] = pypto.read %[[ROW]] [0, 0] : f32 -> f32
  // PYPTO: %[[S1:.*]] = pypto.read %[[ROW]] [1, 0] : f32 -> f32
  // PYPTO: %[[A:.*]] = arith.addf %[[S0]], %[[S1]] : f32
  // PYPTO: pypto.write %arg1[{{.*}}], {{.*}} : memref<1xf32>, f32

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
  // The appended output parameter is rewritten into a [1, 8] memref view:
  // the tile store needs a memref destination.
  // CHECK: %[[OUT:.*]] = emitpy.call_opaque "pl.reshape"(%arg2) {args = [0 : index, #emitpy.opaque<"{{\[1, 8\]}}">
  // Splat tile constants map onto pl.tile.full.
  // CHECK: %[[C:.*]] = emitpy.call_opaque "pl.tile.full"()
  // CHECK: %[[CMP:.*]] = emitpy.call_opaque "pl.cmp"(%[[V0]], %[[V1]])
  // pl.tile.select is the composite form: the runtime synthesizes the
  // architecture-defined scratch tile.
  // CHECK: %[[SEL:.*]] = emitpy.call_opaque "pl.tile.select"(%[[CMP]], %[[V0]], %[[C]])
  // The allocation became an appended output parameter, presented as a [1, 8]
  // tensor because tile stores need a 2D destination; the tile is reshaped
  // through the matching view and stored into it.
  // CHECK: %[[R2:.*]] = emitpy.call_opaque "pl.reshape"(%[[SEL]])
  // CHECK: emitpy.call_opaque "pl.store"(%[[R2]], %[[OUT]])

  // The splat constant, the comparison, the select and the store side
  // with its [1, 8] reshape glue all ride in the pypto layer after the
  // first stage.
  // PYPTO-LABEL: func.func @test_select_alloc(
  // PYPTO: %[[C:.*]] = pypto.tile_full "1.0" : vector<8xf32>
  // PYPTO: %[[CMP:.*]] = pypto.cmp lt, %arg0, %arg1 : vector<8xf32>, vector<8xf32> -> vector<8xi1>
  // PYPTO: %[[SEL:.*]] = pypto.select %[[CMP]], %arg0, %[[C]] : vector<8xi1>, vector<8xf32>, vector<8xf32> to vector<8xf32>

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
  // CHECK: emitpy.spmd_scope (%{{.+}}) in (64) {
  // The pypto layer prints the structured ops scf-style: the block index
  // and the induction variable are bound inline, without ^bb0 headers.
  // PYPTO: pypto.spmd (%{{.+}}) in (64) {
  // PYPTO: pypto.range %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} {
  // Scalar index math stays arith (pypto is a tile dialect) and lowers
  // onto infix pseudo-callees named after the arith ops.
  // CHECK: %[[ROW0:.*]] = emitpy.call_opaque "muli"(%{{.+}}, %{{.+}}) : (index, index) -> index
  // scf.for becomes a pl.range tile loop.
  // CHECK: emitpy.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : index {
  // CHECK: %[[ROW:.*]] = emitpy.call_opaque "addi"(%[[ROW0]], %{{.+}}) : (index, index) -> index
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

// -----

func.func @test_spmd_forall_grid(%arg0 : memref<1024x1024xf32>, %arg1 : memref<1024x1024xf32>) {
  // CHECK-LABEL: func.func @test_spmd_forall_grid(
  // CHECK-SAME: %arg1: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel}
  // A single-induction scf.forall becomes the same SPMD launch as
  // scf.parallel; the block argument is the core index.
  // CHECK: emitpy.spmd_scope (%{{.+}}) in (64) {
  // CHECK: %[[ROW0:.*]] = emitpy.call_opaque "muli"(%{{.+}}, %{{.+}}) : (index, index) -> index
  // CHECK: emitpy.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : index {
  // CHECK: %[[ROW:.*]] = emitpy.call_opaque "addi"(%[[ROW0]], %{{.+}}) : (index, index) -> index
  // CHECK: emitpy.call_opaque "pl.load"(%arg0, %[[ROW]], %{{.+}}) {args = [0 : index, {{\[[0-9]+ : index, [0-9]+ : index\]}}, #emitpy.opaque<"{{\[8, 1024\]}}">
  // CHECK: emitpy.call_opaque "pl.store"(%{{.+}}, %arg1, %[[ROW]], %{{.+}}) {args = [0 : index, {{\[[0-9]+ : index, [0-9]+ : index\]}}, 1 : index]

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %cf0 = arith.constant 0.0 : f32
  scf.forall (%blk) = (0) to (64) step (1) {
    %row0 = arith.muli %blk, %c16 : index
    scf.for %t = %c0 to %c2 step %c1 {
      %t8 = arith.muli %t, %c8 : index
      %row = arith.addi %row0, %t8 : index
      %v = vector.transfer_read %arg0[%row, %c0], %cf0 : memref<1024x1024xf32>, vector<8x1024xf32>
      vector.transfer_write %v, %arg1[%row, %c0] : vector<8x1024xf32>, memref<1024x1024xf32>
    }
    scf.forall.in_parallel {
    }
  }
  return
}

// -----

func.func @test_spmd_forall_dynamic_bounds(%arg0 : memref<1024x1024xf32>, %arg1 : memref<1024x1024xf32>) {
  // CHECK-LABEL: func.func @test_spmd_forall_dynamic_bounds(
  // CHECK-SAME: %arg1: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel}
  // forall keeps its bounds as dynamic operands in this spelling; the grid
  // still folds them into the same constant SPMD launch.
  // CHECK: emitpy.spmd_scope (%{{.+}}) in (64) {
  // CHECK: %[[ROW:.*]] = emitpy.call_opaque "muli"(%{{.+}}, %{{.+}}) : (index, index) -> index
  // CHECK: emitpy.call_opaque "pl.load"(%arg0, %[[ROW]], %{{.+}}) {args = [0 : index, {{\[[0-9]+ : index, [0-9]+ : index\]}}, #emitpy.opaque<"{{\[8, 1024\]}}">
  // CHECK: emitpy.call_opaque "pl.store"(%{{.+}}, %arg1, %[[ROW]], %{{.+}}) {args = [0 : index, {{\[[0-9]+ : index, [0-9]+ : index\]}}, 1 : index]

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  %c16 = arith.constant 16 : index
  %cf0 = arith.constant 0.0 : f32
  scf.forall (%blk) = (%c0) to (%c64) step (%c1) {
    %row = arith.muli %blk, %c16 : index
    %v = vector.transfer_read %arg0[%row, %c0], %cf0 : memref<1024x1024xf32>, vector<8x1024xf32>
    vector.transfer_write %v, %arg1[%row, %c0] : vector<8x1024xf32>, memref<1024x1024xf32>
    scf.forall.in_parallel {
    }
  }
  return
}

// -----

// A loop-invariant splat defined before the forall lowers to a pypto
// tile_full outside the spmd region. The DSL lowers `with pl.spmd(N):`
// into a standalone InCore function that cannot capture tile values, so
// the conversion sinks the tile_full into the region (right before its
// first user).
func.func @test_spmd_sink_tile_full(%arg0 : memref<1024xf32>, %arg1 : memref<1024xf32>) {
  // CHECK-LABEL: func.func @test_spmd_sink_tile_full(
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel}
  // CHECK: emitpy.spmd_scope (%{{.+}}) in (1) {
  // CHECK: emitpy.call_opaque "pl.load"
  // CHECK: emitpy.call_opaque "pl.tile.full"
  // CHECK: emitpy.call_opaque "pl.add"
  // CHECK: emitpy.call_opaque "pl.store"
  // The tile_full must be created inside the region, never before it.
  // PYPTO-NOT: pypto.tile_full
  // PYPTO: pypto.spmd (%{{.+}}) in (1) {
  // PYPTO: pypto.load
  // PYPTO: pypto.tile_full
  // PYPTO: pypto.add
  // PYPTO: pypto.store

  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %zero = arith.constant dense<0.0> : vector<1024xf32>
  scf.forall (%blk) = (0) to (1) step (1) {
    %v = vector.transfer_read %arg0[%c0], %cf0 : memref<1024xf32>, vector<1024xf32>
    %sum = arith.addf %v, %zero : vector<1024xf32>
    vector.transfer_write %sum, %arg1[%c0] : vector<1024xf32>, memref<1024xf32>
    scf.forall.in_parallel {
    }
  }
  return
}

// -----

// The bufferization routes stores through reshaped views of parameters: a
// transfer_write into a view of an argument still makes that argument an
// output.
func.func @test_store_through_view(%arg0 : memref<64x64xi32>, %arg1 : memref<64x64xi32>) {
  // CHECK-LABEL: func.func @test_store_through_view(
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel}
  // CHECK: %[[RIN:.*]] = emitpy.call_opaque "pl.reshape"(%arg0)
  // CHECK: %[[ROUT:.*]] = emitpy.call_opaque "pl.reshape"(%arg1)
  // The tile store needs a 2D memref destination: the flat output view is
  // reshaped again into a [1, 4096] memref.
  // CHECK: %[[OUT:.*]] = emitpy.call_opaque "pl.reshape"(%[[ROUT]]) {args = [0 : index, #emitpy.opaque<"{{\[1, 4096\]}}">
  // CHECK: emitpy.call_opaque "pl.store"({{.*}}, %[[OUT]])

  // The collapse views of the parameters, the [1, N] store-destination
  // reshape and the tile store all ride in the pypto layer after the
  // first stage.
  // PYPTO-LABEL: func.func @test_store_through_view(
  // PYPTO: %[[RIN:.*]] = pypto.reshape %arg0 : memref<64x64xi32> to memref<4096xi32>
  // PYPTO: %[[ROUT:.*]] = pypto.reshape %arg1 : memref<64x64xi32> to memref<4096xi32>
  // PYPTO: %[[OUT:.*]] = pypto.reshape %[[ROUT]] : memref<4096xi32> to memref<1x4096xi32>
  // PYPTO: pypto.store {{.*}}, %[[OUT]][{{.*}}] : !pypto.tile<i32, [1, 88]>, memref<1x4096xi32>

  %c0 = arith.constant 0 : index
  %c86 = arith.constant 86 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %flat_in = memref.collapse_shape %arg0 [[0, 1]] : memref<64x64xi32> into memref<4096xi32>
  %flat_out = memref.collapse_shape %arg1 [[0, 1]] : memref<64x64xi32> into memref<4096xi32>
  %v = npuvector.transfer_read %flat_in[%c0] [%c86] [%c86], %c0_i32 : memref<4096xi32>, !npuvector<?xi32>
  %one = npuvector.broadcast %c1_i32[%c86] [%c86] : i32 to !npuvector<?xi32>
  %sum = arith.addi %v, %one : !npuvector<?xi32>
  npuvector.transfer_write %sum, %flat_out[%c0] : !npuvector<?xi32>, memref<4096xi32>
  return
}

// -----

// A view that is read in addition to being written keeps its root argument
// an InOut instead of an Out.
func.func @test_store_through_read_view(%arg0 : memref<64x64xi32>, %arg1 : memref<64x64xi32>) {
  // CHECK-LABEL: func.func @test_store_through_read_view(
  // CHECK-SAME: attributes {emitpy.inout_args = array<i64: 1>, emitpy.pypto_kernel}

  %c0 = arith.constant 0 : index
  %c86 = arith.constant 86 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %flat_out = memref.collapse_shape %arg1 [[0, 1]] : memref<64x64xi32> into memref<4096xi32>
  %v = npuvector.transfer_read %flat_out[%c0] [%c86] [%c86], %c0_i32 : memref<4096xi32>, !npuvector<?xi32>
  %one = npuvector.broadcast %c1_i32[%c86] [%c86] : i32 to !npuvector<?xi32>
  %sum = arith.addi %v, %one : !npuvector<?xi32>
  npuvector.transfer_write %sum, %flat_out[%c0] : !npuvector<?xi32>, memref<4096xi32>
  return
}

// -----

func.func @test_collapse_and_dynamic_npuvector(%arg0 : memref<64x64xi32>, %arg1 : memref<64x64xi32>) {
  // CHECK-LABEL: func.func @test_collapse_and_dynamic_npuvector(
  // CHECK-SAME: %arg0: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.INT32
  // CHECK-SAME: %arg1: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.INT32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel}
  // A statically shaped collapse view becomes pl.reshape over the source
  // tensor. The replacement keeps the memref result type: the type converter
  // is the identity, so a type-changing replacement would demand a
  // materialization at already-converted users (a live pl.load) and abort
  // the full conversion.
  // CHECK: %[[FLAT0:.*]] = emitpy.call_opaque "pl.reshape"(%arg0) {args = [0 : index, #emitpy.opaque<"{{\[4096\]}}">
  // CHECK-SAME: -> memref<4096xi32>
  // CHECK: %[[FLAT1:.*]] = emitpy.call_opaque "pl.reshape"(%arg1) {args = [0 : index, #emitpy.opaque<"{{\[4096\]}}">
  // CHECK-SAME: -> memref<4096xi32>
  // The tile store needs a 2D memref destination: the flat output view is
  // reshaped again into a [1, 4096] memref.
  // CHECK: %[[OUT:.*]] = emitpy.call_opaque "pl.reshape"(%[[FLAT1]]) {args = [0 : index, #emitpy.opaque<"{{\[1, 4096\]}}">
  // CHECK: emitpy.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} : index {
  // Index arithmetic lowers to scalar pl calls: floordivsi keeps Python
  // floor semantics through its own infix pseudo-callee, and minsi becomes
  // pl.min.
  // CHECK: emitpy.call_opaque "floordivsi"(%{{.+}}, %{{.+}})
  // CHECK: %[[BEG:.*]] = emitpy.call_opaque "pl.min"({{.+}}) : (index, index) -> index
  // CHECK: %{{.+}} = emitpy.call_opaque "pl.min"({{.+}}) : (index, index) -> index
  // CHECK: %[[LEN:.*]] = emitpy.call_opaque "pl.min"({{.+}}) : (index, index) -> index
  // A dynamic-extent read allocates the tile at its maxSizes bound padded to
  // whole 32-byte rows (86 i32 = 344B -> 88 elements = 352B) and carries the
  // runtime valid extent as an operand reference.
  // CHECK: %[[V:.*]] = emitpy.call_opaque "pl.load"(%[[FLAT0]], %[[BEG]], %[[LEN]]) {args = [0 : index, {{\[[0-9]+ : index\]}}, #emitpy.opaque<"{{\[88\]}}">, {{\[[0-9]+ : index\]}}
  // CHECK-SAME: -> !npuvector<?xi32>
  // A scalar broadcast onto a dynamic tile is a zero tile whose allocated
  // extent comes from the padded maxSizes bound; the folded constant fills it.
  // CHECK: %[[B:.*]] = emitpy.call_opaque "pl.tile.full"() {args = [#emitpy.opaque<"{{\[88\]}}">, #emitpy.opaque<"pl.INT32">, #emitpy.opaque<"1">
  // CHECK: %[[SUM:.*]] = emitpy.call_opaque "pl.add"(%[[V]], %[[B]])
  // The dynamic tile is reshaped into a matching [1, 88] tile and stored
  // through the 2D view, its base offset riding along as a dynamic operand.
  // CHECK: %[[RTILE:.*]] = emitpy.call_opaque "pl.reshape"(%[[SUM]]) {args = [0 : index, #emitpy.opaque<"{{\[1, 88\]}}">
  // CHECK: emitpy.call_opaque "pl.store"(%[[RTILE]], %[[OUT]], {{.+}}, %[[BEG]]) {args = [0 : index, {{\[[0-9]+ : index, [0-9]+ : index\]}}, 1 : index]

  // The collapse views ride in the pypto layer together with the scalar
  // index arithmetic; the floor division keeps its floordivsi pseudo-callee
  // for Python floor semantics.
  // PYPTO-LABEL: func.func @test_collapse_and_dynamic_npuvector(
  // PYPTO: %[[FLAT0:.*]] = pypto.reshape %arg0 : memref<64x64xi32> to memref<4096xi32>
  // PYPTO: %[[FLAT1:.*]] = pypto.reshape %arg1 : memref<64x64xi32> to memref<4096xi32>
  // PYPTO: %[[BEG:.*]] = pypto.min {{.*}} : index, index to index
  // PYPTO: %[[V:.*]] = pypto.load %[[FLAT0]][%[[BEG]]] padded[{{.*}}] valid[{{.*}}] : memref<4096xi32> -> !npuvector<?xi32>

  %c-1 = arith.constant -1 : index
  %c4096 = arith.constant 4096 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c48 = arith.constant 48 : index
  %c86 = arith.constant 86 : index
  %c-4128 = arith.constant -4128 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %flat0 = memref.collapse_shape %arg0 [[0, 1]] : memref<64x64xi32> into memref<4096xi32>
  %flat1 = memref.collapse_shape %arg1 [[0, 1]] : memref<64x64xi32> into memref<4096xi32>
  scf.for %i = %c0 to %c48 step %c1 {
    %row = arith.muli %i, %c86 : index
    %blk = arith.floordivsi %i, %c48 : index
    %pad = arith.muli %blk, %c-4128 : index
    %beg0 = arith.addi %row, %pad : index
    %beg = arith.minsi %beg0, %c4096 : index
    %end0 = arith.addi %beg, %c86 : index
    %end = arith.minsi %end0, %c4096 : index
    %neg = arith.muli %beg, %c-1 : index
    %len0 = arith.addi %end, %neg : index
    %len = arith.minsi %len0, %c86 : index
    %v = npuvector.transfer_read %flat0[%beg] [%len] [%c86], %c0_i32 : memref<4096xi32>, !npuvector<?xi32>
    %b = npuvector.broadcast %c1_i32[%len] [%c86] : i32 to !npuvector<?xi32>
    %sum = arith.addi %v, %b : !npuvector<?xi32>
    npuvector.transfer_write %sum, %flat1[%beg] : !npuvector<?xi32>, memref<4096xi32>
  }
  return
}

// -----

func.func @test_expand_shape(%arg0 : memref<4096xf32>, %arg1 : memref<64x64xf32>) {
  // CHECK-LABEL: func.func @test_expand_shape(
  // CHECK-SAME: %arg0: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: %arg1: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // A statically shaped expand view becomes pl.reshape over the source
  // tensor, mirroring the collapse handling; the replacement keeps the
  // memref result type so no materialization is needed at users.
  // CHECK: %[[VIEW:.*]] = emitpy.call_opaque "pl.reshape"(%arg0) {args = [0 : index, #emitpy.opaque<"{{\[64, 64\]}}">
  // CHECK-SAME: -> memref<64x64xf32>
  // CHECK: %[[V:.*]] = emitpy.call_opaque "pl.load"(%[[VIEW]])
  // CHECK: emitpy.call_opaque "pl.store"(%[[V]], %arg1)

  // The expand view rides in the pypto layer; its memref result type feeds
  // the pypto.load directly.
  // PYPTO-LABEL: func.func @test_expand_shape(
  // PYPTO: %[[VIEW:.*]] = pypto.reshape %arg0 : memref<4096xf32> to memref<64x64xf32>
  // PYPTO: %[[V:.*]] = pypto.load %[[VIEW]][{{.*}}] : memref<64x64xf32> -> vector<32x32xf32>

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %view = memref.expand_shape %arg0 [[0, 1]] output_shape [64, 64] : memref<4096xf32> into memref<64x64xf32>
  %v = vector.transfer_read %view[%c0, %c0], %cst : memref<64x64xf32>, vector<32x32xf32>
  vector.transfer_write %v, %arg1[%c0, %c0] : vector<32x32xf32>, memref<64x64xf32>
  return
}

// -----

func.func @test_memref_shape_views(%arg0 : memref<64x32xf32>, %arg1 : memref<32x64xf32>, %arg2 : memref<16x16xf32>, %shape : memref<2xindex>) {
  // CHECK-LABEL: func.func @test_memref_shape_views(
  // CHECK: %[[T:.*]] = emitpy.call_opaque "pl.transpose"(%arg0)
  // CHECK: %[[R:.*]] = emitpy.call_opaque "pl.reshape"(%arg1)
  // The subview folds into its load: the memory op addresses the reshape's
  // result directly, with the window offsets on its own offsets.
  // CHECK: %[[V:.*]] = emitpy.call_opaque "pl.load"(%[[R]])
  // CHECK: emitpy.call_opaque "pl.store"

  // The transpose and reshape ride in the pypto layer; the subview folds
  // into the load, which addresses the reshape result directly.
  // PYPTO-LABEL: func.func @test_memref_shape_views(
  // PYPTO: %[[T:.*]] = pypto.transpose %arg0, 0, 1 : memref<64x32xf32> to memref<32x64xf32, #map>
  // PYPTO: %[[R:.*]] = pypto.reshape %arg1 : memref<32x64xf32> to memref<16x128xf32>
  // PYPTO: %[[V:.*]] = pypto.load %[[R]][{{.*}}] : memref<16x128xf32> -> vector<16x16xf32>

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %t = memref.transpose %arg0 (d0, d1) -> (d1, d0) : memref<64x32xf32> to memref<32x64xf32, affine_map<(d0, d1) -> (d0 + d1 * 32)>>
  %vt = vector.transfer_read %t[%c0, %c0], %cst : memref<32x64xf32, affine_map<(d0, d1) -> (d0 + d1 * 32)>>, vector<16x16xf32>
  vector.transfer_write %vt, %arg1[%c0, %c0] : vector<16x16xf32>, memref<32x64xf32>
  %r = memref.reshape %arg1(%shape) : (memref<32x64xf32>, memref<2xindex>) -> memref<16x128xf32>
  %s = memref.subview %r[0, 0] [16, 16] [1, 1] : memref<16x128xf32> to memref<16x16xf32, strided<[128, 1]>>
  %v = vector.transfer_read %s[%c0, %c0], %cst : memref<16x16xf32, strided<[128, 1]>>, vector<16x16xf32>
  vector.transfer_write %v, %arg2[%c0, %c0] : vector<16x16xf32>, memref<16x16xf32>
  return
}

// -----

func.func @test_memref_reinterpret_cast(%arg0 : memref<64xf32>, %arg1 : memref<16xf32>) {
  // CHECK-LABEL: func.func @test_memref_reinterpret_cast(
  // The reinterpret_cast folds into its load: the window offset lands on
  // the memory op's offsets, addressing the base directly.
  // CHECK: %[[V:.*]] = emitpy.call_opaque "pl.load"(%arg0)
  // CHECK-SAME: #emitpy.opaque<"{{\[16\]}}">
  // CHECK: emitpy.call_opaque "pl.store"

  // A reinterpret_cast with a static offset folds into the load.
  // PYPTO-LABEL: func.func @test_memref_reinterpret_cast(
  // PYPTO: %[[V:.*]] = pypto.load %arg0[{{.*}}] : memref<64xf32> -> vector<16xf32>

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %r = memref.reinterpret_cast %arg0 to offset: [16], sizes: [16], strides: [1] : memref<64xf32> to memref<16xf32, strided<[1], offset: 16>>
  %v = vector.transfer_read %r[%c0], %cst : memref<16xf32, strided<[1], offset: 16>>, vector<16xf32>
  vector.transfer_write %v, %arg1[%c0] : vector<16xf32>, memref<16xf32>
  return
}

// -----

func.func @test_scalar_select(%arg0 : memref<1024x1024xf32>, %arg1 : memref<1024x1024xf32>) {
  // CHECK-LABEL: func.func @test_scalar_select(
  // CHECK-SAME: %arg1: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel}
  // CHECK: emitpy.spmd_scope (%{{.+}}) in (64) {
  // Scalar index math stays arith in the first stage (pypto is a tile
  // dialect); the second stage maps it onto infix pseudo-callees named
  // after the arith ops (printed by the translator as plain Python
  // expressions and the conditional-expression form of select).
  // CHECK: %[[MUL:.*]] = emitpy.call_opaque "muli"({{.+}}) : (index, index) -> index
  // CHECK: %[[CMP:.*]] = emitpy.call_opaque "lt"({{.+}}) : (index, index) -> i1
  // CHECK: %[[SEL:.*]] = emitpy.call_opaque "select"(%[[CMP]], %[[MUL]], %{{.+}}) : (i1, index, index) -> index
  // CHECK: emitpy.call_opaque "pl.load"(%arg0, %[[SEL]], %{{.+}}) {args = [0 : index, {{\[[0-9]+ : index, [0-9]+ : index\]}}, #emitpy.opaque<"{{\[8, 1024\]}}">
  // CHECK: emitpy.call_opaque "pl.store"(%{{.+}}, %arg1, %[[SEL]], %{{.+}}) {args = [0 : index, {{\[[0-9]+ : index, [0-9]+ : index\]}}, 1 : index]

  // Scalar arithmetic keeps the arith op through the first stage; only
  // the tile work rides in the pypto layer.
  // PYPTO-LABEL: func.func @test_scalar_select(
  // PYPTO: %[[MUL:.*]] = arith.muli %{{.+}}, %{{.+}} : index
  // PYPTO: %[[CMP:.*]] = arith.cmpi slt, %{{.+}}, %{{.+}} : index
  // PYPTO: %[[SEL:.*]] = arith.select %[[CMP]], %[[MUL]], %{{.+}} : index

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index
  %cf0 = arith.constant 0.0 : f32
  scf.parallel (%blk) = (%c0) to (%c64) step (%c1) {
    %row0 = arith.muli %blk, %c2 : index
    scf.for %t = %c0 to %c2 step %c1 {
      %cmp = arith.cmpi slt, %t, %c2 : index
      %row = arith.select %cmp, %row0, %c0 : index
      %v = vector.transfer_read %arg0[%row, %c0], %cf0 : memref<1024x1024xf32>, vector<8x1024xf32>
      vector.transfer_write %v, %arg1[%row, %c0] : vector<8x1024xf32>, memref<1024x1024xf32>
    }
  }
  return
}
