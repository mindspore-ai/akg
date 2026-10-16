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
  // The appended output parameter is rewritten into a [1, 8] memref view:
  // the tile store needs a memref destination.
  // CHECK: %[[OUT:.*]] = emitpy.call_opaque "pl.reshape"(%arg2) {args = [0 : index, #emitpy.opaque<"{{\[1, 8\]}}">
  // Splat tile constants map onto pl.tile.full.
  // CHECK: %[[C:.*]] = emitpy.call_opaque "pl.tile.full"()
  // CHECK: %[[CMP:.*]] = emitpy.call_opaque "pl.cmp"(%[[V0]], %[[V1]])
  // pl.sel takes tiles and an inline scratch tile.
  // CHECK: %[[TMP:.*]] = emitpy.call_opaque "pl.create_tile"()
  // CHECK: %[[SEL:.*]] = emitpy.call_opaque "pl.sel"(%[[CMP]], %[[V0]], %[[C]], %[[TMP]])
  // The allocation became an appended output parameter, presented as a [1, 8]
  // tensor because tile stores need a 2D destination; the tile is reshaped
  // through the matching view and stored into it.
  // CHECK: %[[R2:.*]] = emitpy.call_opaque "pl.reshape"(%[[SEL]])
  // CHECK: emitpy.call_opaque "pl.store"(%[[R2]], %[[OUT]])

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

// -----

func.func @test_spmd_forall_grid(%arg0 : memref<1024x1024xf32>, %arg1 : memref<1024x1024xf32>) {
  // CHECK-LABEL: func.func @test_spmd_forall_grid(
  // CHECK-SAME: %arg1: !emitpy.opaque<"pl.Tensor{{[^"]*}}pl.FP32
  // CHECK-SAME: attributes {emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel}
  // A single-induction scf.forall becomes the same SPMD launch as
  // scf.parallel; the block argument is the core index.
  // CHECK: emitpy.spmd_scope core_num 64 {
  // CHECK: %[[ROW0:.*]] = emitpy.call_opaque "pl.mul"(%{{.+}}, %{{.+}}) : (index, index) -> index
  // CHECK: emitpy.for %{{.+}} to %{{.+}} step %{{.+}} : index index index {
  // CHECK: %[[ROW:.*]] = emitpy.call_opaque "pl.add"(%[[ROW0]], %{{.+}}) : (index, index) -> index
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
  // CHECK: emitpy.spmd_scope core_num 64 {
  // CHECK: %[[ROW:.*]] = emitpy.call_opaque "pl.mul"(%{{.+}}, %{{.+}}) : (index, index) -> index
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
  // CHECK: emitpy.for %{{.+}} to %{{.+}} step %{{.+}} : index index index {
  // Index arithmetic lowers to scalar pl calls: floordivsi keeps Python
  // floor semantics through util_floordiv, and minsi becomes pl.min.
  // CHECK: emitpy.call_opaque "util_floordiv"(%{{.+}}, %{{.+}})
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
  // CHECK: %[[S:.*]] = emitpy.call_opaque "pl.slice"(%[[R]])
  // CHECK-SAME: -> memref<16x16xf32, strided<[128, 1]>>
  // CHECK: emitpy.call_opaque "pl.store"

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
  // CHECK: %[[S:.*]] = emitpy.call_opaque "pl.slice"(%arg0)
  // CHECK-SAME: -> memref<16xf32, strided<[1], offset: 16>>
  // CHECK: emitpy.call_opaque "pl.store"

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
  // CHECK: emitpy.spmd_scope core_num 64 {
  // The loop-invariant arith.muli is emitted ahead of the loop; scalar
  // comparisons keep pl.cmp and the scalar select becomes the util_select
  // pseudo-callee (printed by the translator as a Python conditional
  // expression).
  // CHECK: %[[MUL:.*]] = emitpy.call_opaque "pl.mul"({{.+}}) : (index, index) -> index
  // CHECK: %[[CMP:.*]] = emitpy.call_opaque "util_cmp_lt"({{.+}}) : (index, index) -> i1
  // CHECK: %[[SEL:.*]] = emitpy.call_opaque "util_select"(%[[CMP]], %[[MUL]], %{{.+}}) : (i1, index, index) -> index
  // CHECK: emitpy.call_opaque "pl.load"(%arg0, %[[SEL]], %{{.+}}) {args = [0 : index, {{\[[0-9]+ : index, [0-9]+ : index\]}}, #emitpy.opaque<"{{\[8, 1024\]}}">
  // CHECK: emitpy.call_opaque "pl.store"(%{{.+}}, %arg1, %[[SEL]], %{{.+}}) {args = [0 : index, {{\[[0-9]+ : index, [0-9]+ : index\]}}, 1 : index]

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
