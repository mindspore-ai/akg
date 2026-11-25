// RUN: akg-opt %s -convert-vector-to-pypto -convert-pypto-to-emitpy | akg-translate --mlir-to-pypto | FileCheck %s

func.func @test_elementwise(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: import pypto.language as pl
  // CHECK-LABEL: def test_elementwise(
  // CHECK: inputs: pl.Tensor{{.*}}pl.FP32], inputs_0: pl.Tensor{{.*}}pl.FP32], inputs_1: pl.Out{{.*}}pl.FP32
  // CHECK: with pl.at(level=pl.Level.CORE_GROUP):
  // CHECK: {{.*}} = pl.load(inputs,
  // CHECK: {{.*}} = pl.load(inputs_0,
  // CHECK: {{.*}} = pl.add(
  // CHECK: {{.*}} = pl.exp(
  // CHECK: pl.store(
  // CHECK: return inputs_1

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

func.func @test_tile_ops(%arg0 : !npuvector<8x16xf32>, %arg1 : f32, %out : memref<f32>) {
  // CHECK-LABEL: def test_tile_ops(
  // CHECK: inputs: pl.Tensor{{.*}}pl.FP32], inputs_0: pl.Scalar{{.*}}pl.FP32], inputs_1: pl.Out{{.*}}pl.FP32
  // CHECK: [[OUT_VIEW:var_[0-9]+]] = pl.reshape(inputs_1, [1])
  // CHECK-NEXT: [[OUT:var_[0-9]+]] = pl.reshape([[OUT_VIEW]], [1, 1])
  // CHECK: with pl.at(level=pl.Level.CORE_GROUP):
  // Tile parameters are loaded into tiles at the entry.
  // CHECK: {{.*}} = pl.load(inputs,
  // CHECK: {{.*}} = pl.transpose({{.*}}, 0, 1)
  // A broadcast scalar feeding the arith mul fuses onto its native
  // tile-first scalar operand: no carrier tile is materialized.
  // CHECK: {{.*}} = pl.mul({{.*}}, inputs_0)
  // Reduce both axes on device and store the resulting tile.
  // CHECK: [[ROWS:var_[0-9]+]] = pl.row_sum(
  // CHECK: [[ROW:var_[0-9]+]] = pl.reshape([[ROWS]], [1, 16])
  // CHECK: [[TMP:var_[0-9]+]] = pl.create_tile([1, 16],
  // CHECK: [[SUM:var_[0-9]+]] = pl.row_sum([[ROW]], [[TMP]])
  // CHECK-NOT: pl.read(
  // CHECK-NOT: pl.write(
  // CHECK-NEXT: pl.store([[SUM]], [0, 0], [[OUT]])
  // CHECK: return inputs_1

  %t = npuvector.transpose %arg0, [1, 0] : !npuvector<8x16xf32> to !npuvector<16x8xf32>
  %b = npuvector.broadcast %arg1 : f32 to !npuvector<16x8xf32>
  %m = arith.mulf %t, %b : !npuvector<16x8xf32>
  %r = npuvector.reduction <add>, %m : !npuvector<16x8xf32> into !npuvector.f32
  %view = memref.expand_shape %out [] output_shape [1] : memref<f32> into memref<1xf32>
  %c0 = arith.constant 0 : index
  npuvector.transfer_write %r, %view[%c0] : !npuvector.f32, memref<1xf32>
  return
}

// -----

func.func @test_rank1_full_reduction(%arg0 : !npuvector<3072xf32>, %out : memref<f32>) {
  // CHECK-LABEL: def test_rank1_full_reduction(
  // CHECK: inputs: pl.Tensor{{.*}}pl.FP32], inputs_0: pl.Out{{.*}}pl.FP32
  // CHECK: [[OUT_VIEW:var_[0-9]+]] = pl.reshape(inputs_0, [1])
  // CHECK-NEXT: [[OUT:var_[0-9]+]] = pl.reshape([[OUT_VIEW]], [1, 1])
  // CHECK: with pl.at(level=pl.Level.CORE_GROUP):
  // Reshape the rank-1 input to a single row and reduce it on device.
  // CHECK: [[IN:var_[0-9]+]] = pl.load(inputs,
  // CHECK: [[ROW:var_[0-9]+]] = pl.reshape([[IN]], [1, 3072])
  // CHECK: [[TMP:var_[0-9]+]] = pl.create_tile([1, 3072],
  // CHECK: [[SUM:var_[0-9]+]] = pl.row_sum([[ROW]], [[TMP]])
  // CHECK-NOT: pl.read(
  // CHECK-NOT: pl.write(
  // CHECK-NEXT: pl.store([[SUM]], [0, 0], [[OUT]])
  // CHECK: return inputs_0

  %r = npuvector.reduction <add>, %arg0 : !npuvector<3072xf32> into !npuvector.f32
  %view = memref.expand_shape %out [] output_shape [1] : memref<f32> into memref<1xf32>
  %c0 = arith.constant 0 : index
  npuvector.transfer_write %r, %view[%c0] : !npuvector.f32, memref<1xf32>
  return
}

// -----

func.func @test_tensor_view(%arg0 : memref<64x64xi32>, %arg1 : memref<64x64xi32>) {
  // CHECK-LABEL: def test_tensor_view(
  // A view of a kernel parameter is reshaped at host level, before the
  // CORE_GROUP scope: inside it pl.reshape would be materialized as a tile
  // op on the tensor parameter. The tile-level reshape of the loaded value
  // stays inside the scope.
  // CHECK: {{.*}} = pl.reshape(inputs, [4096])
  // CHECK: {{.*}} = pl.reshape(inputs_0, [4096])
  // CHECK: with pl.at(level=pl.Level.CORE_GROUP):
  // CHECK: {{.*}} = pl.load({{.*}}, [0], [4096])
  // CHECK: {{.*}} = pl.reshape({{.*}}, [1, 4096])
  // CHECK: pl.store(
  // CHECK: return inputs_0

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %in = memref.collapse_shape %arg0 [[0, 1]] : memref<64x64xi32> into memref<4096xi32>
  %out = memref.collapse_shape %arg1 [[0, 1]] : memref<64x64xi32> into memref<4096xi32>
  %v = vector.transfer_read %in[%c0], %cst : memref<4096xi32>, vector<4096xi32>
  vector.transfer_write %v, %out[%c0] : vector<4096xi32>, memref<4096xi32>
  return
}

// -----

func.func @test_spmd_tile_loop(%arg0 : memref<1024x1024xf32>, %arg1 : memref<1024x1024xf32>) {
  // CHECK-LABEL: def test_spmd_tile_loop(
  // Multi-core kernels launch an SPMD grid instead of a pl.at scope; the
  // block index binds to pl.tile.get_block_idx() and scf.for becomes a
  // pl.range loop whose loads and stores carry inlined dynamic offsets.
  // CHECK-NOT: with pl.at
  // CHECK: with pl.spmd(64):
  // CHECK: block_idx = pl.tile.get_block_idx()
  // Scalar index math rides on plain Python arithmetic: the parser maps
  // `*` onto pto.muli and `+` onto pto.addi.
  // CHECK: {{.*}} = (block_idx * 16)
  // CHECK: for loop_var in pl.range(0, 2, 1):
  // CHECK: {{.*}} = (loop_var * 8)
  // CHECK: {{.*}} = ({{.*}} + {{.*}})
  // CHECK: {{.*}} = pl.load(inputs, {{\[.*0\]}}, {{\[8, 1024\]}})
  // CHECK: pl.store({{.*}}, {{\[.*0\]}}, inputs_0)
  // CHECK: return inputs_0

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
