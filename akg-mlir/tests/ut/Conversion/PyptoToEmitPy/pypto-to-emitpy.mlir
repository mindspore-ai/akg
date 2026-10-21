// RUN: akg-opt %s -convert-pypto-to-emitpy | FileCheck %s
// RUN: akg-opt %s -convert-pypto-to-emitpy | akg-translate --allow-unregistered-dialect --mlir-to-pypto | FileCheck %s --check-prefix=PYTHON

// End-to-end check of the pypto -> emitpy lowering:
//   pypto.load / pypto.store  ->  emitpy.call_opaque "pl.load" / "pl.store"
//   !pypto.tile<f32, [4, 8]>  ->  !emitpy.opaque<"pl.Tile[[4, 8], pl.FP32]">
// The source/dest operands are PyPTO tensor-typed values, mirroring what the
// future vector/npuvector -> pypto layer will feed in.

func.func @tile_load_store_const_offsets(%arg0 : !emitpy.opaque<"pl.Tensor[[32, 32], pl.FP32]">) {
  // CHECK-LABEL: func.func @tile_load_store_const_offsets(
  // Constant offsets fold into the opaque list literal "[0, 1]", and the
  // tile shape is passed as the third argument.
  // CHECK: %[[TILE:.*]] = emitpy.call_opaque "pl.load"(%arg0) {args = [0 : index, #emitpy.opaque<"{{\[0, 1\]}}">, #emitpy.opaque<"{{\[4, 8\]}}">],
  // CHECK-SAME: keyword_args = ["", "", ""]}
  // CHECK-SAME: : (!emitpy.opaque<"pl.Tensor{{[^"]*}}">) -> !emitpy.opaque<"pl.Tile{{\[\[4, 8\], pl\.FP32\]}}">
  // The store targets the tensor positionally (argument 1).
  // CHECK: emitpy.call_opaque "pl.store"(%[[TILE]], %arg0) {args = [0 : index, #emitpy.opaque<"{{\[1, 0\]}}">, 1 : index],
  // CHECK-SAME: keyword_args = ["", "", ""]}

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile = pypto.load %arg0[%c0, %c1] : !emitpy.opaque<"pl.Tensor[[32, 32], pl.FP32]"> -> !pypto.tile<f32, [4, 8]>
  pypto.store %tile, %arg0[%c1, %c0] : !pypto.tile<f32, [4, 8]>, !emitpy.opaque<"pl.Tensor[[32, 32], pl.FP32]">
  return
}

// -----

func.func @tile_load_store_dynamic_offsets(%arg0 : !emitpy.opaque<"pl.Tensor[[32, 32], pl.FP32]">, %i : index, %j : index) {
  // CHECK-LABEL: func.func @tile_load_store_dynamic_offsets(
  // Dynamic offsets flow as call operands and are referenced positionally
  // from the nested args list.
  // CHECK: %[[TILE:.*]] = emitpy.call_opaque "pl.load"(%arg0, %arg1, %arg2) {args = [0 : index, {{\[1 : index, 2 : index\]}}, #emitpy.opaque<"{{\[4, 8\]}}">],
  // CHECK-SAME: keyword_args = ["", "", ""]}
  // CHECK: emitpy.call_opaque "pl.store"(%[[TILE]], %arg0, %arg1, %arg2) {args = [0 : index, {{\[2 : index, 3 : index\]}}, 1 : index],
  // CHECK-SAME: keyword_args = ["", "", ""]}

  %tile = pypto.load %arg0[%i, %j] : !emitpy.opaque<"pl.Tensor[[32, 32], pl.FP32]"> -> !pypto.tile<f32, [4, 8]>
  pypto.store %tile, %arg0[%i, %j] : !pypto.tile<f32, [4, 8]>, !emitpy.opaque<"pl.Tensor[[32, 32], pl.FP32]">
  return
}

// -----

// A `?` extent cannot be spelled as an annotation, so the dynamic form of
// pypto.load passes the per-dimension allocation bounds (padded) and actual
// extents (valid) and keeps the source tile type on the call.

func.func @tile_load_dynamic_extents(%arg0 : !emitpy.opaque<"pl.Tensor[[32, 32], pl.FP32]">, %i : index, %j : index, %p0 : index, %p1 : index, %v0 : index, %v1 : index) {
  // CHECK-LABEL: func.func @tile_load_dynamic_extents(
  // CHECK: %[[DYN:.*]] = emitpy.call_opaque "pl.load"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6) {args = [0 : index, {{\[1 : index, 2 : index\]}}, {{\[3 : index, 4 : index\]}}, {{\[5 : index, 6 : index\]}}],
  // CHECK-SAME: keyword_args = ["", "", "", ""]}
  // CHECK-SAME: : (!emitpy.opaque<"pl.Tensor{{[^"]*}}">, index, index, index, index, index, index) -> !pypto.tile<f32, [?, 8]>

  %dyn = pypto.load %arg0[%i, %j] padded [%p0, %p1] valid [%v0, %v1] : !emitpy.opaque<"pl.Tensor[[32, 32], pl.FP32]"> -> !pypto.tile<f32, [?, 8]>
  return
}

// -----

// Constants and allocation.

func.func @constants_and_allocs() {
  // CHECK-LABEL: func.func @constants_and_allocs(
  // The dtype pins the exact PyPTO scalar type.
  // CHECK: %[[C:.*]] = emitpy.call_opaque "pl.const"() {args = [#emitpy.opaque<"1.0">, #emitpy.opaque<"pl.FP32">], keyword_args = ["", ""]} : () -> f32
  // CHECK: %[[I:.*]] = emitpy.call_opaque "pl.const"() {args = [#emitpy.opaque<"42">, #emitpy.opaque<"pl.INT32">], keyword_args = ["", ""]} : () -> i32
  // Splat allocation takes shape and dtype from the result type.
  // CHECK: %[[Z:.*]] = emitpy.call_opaque "pl.tile.full"() {args = [#emitpy.opaque<"[4, 8]">, #emitpy.opaque<"pl.FP32">, #emitpy.opaque<"0.0">], keyword_args = ["", "", ""]}
  // Scratch allocation carries the dtype/target_memory keywords.
  // CHECK: %[[T:.*]] = emitpy.call_opaque "pl.create_tile"() {args = [#emitpy.opaque<"[4, 8]">, #emitpy.opaque<"pl.FP32">, #emitpy.opaque<"pl.MemorySpace.Vec">], keyword_args = ["", "dtype", "target_memory"]}
  // CHECK: emitpy.call_opaque "pl.write"(%[[T]], %[[C]]) {args = [0 : index, #emitpy.opaque<"{{\[0, 0\]}}">, 1 : index], keyword_args = ["", "", ""]}

  %c0 = arith.constant 0 : index
  %c = pypto.constant "1.0" : f32
  %i = pypto.constant "42" : i32
  %z = pypto.tile_full "0.0" : !pypto.tile<f32, [4, 8]>
  %t = pypto.create_tile : !pypto.tile<f32, [4, 8]>
  pypto.write %t[%c0, %c0], %c : !pypto.tile<f32, [4, 8]>, f32
  return
}

// -----

// Tile views.

func.func @tile_slice(%src : !pypto.tile<f32, [8, 16]>, %i : index, %j : index) {
  // CHECK-LABEL: func.func @tile_slice(
  // Constant offsets fold into the opaque list literal.
  // CHECK: emitpy.call_opaque "pl.slice"(%arg0) {args = [0 : index, #emitpy.opaque<"[4, 8]">, #emitpy.opaque<"{{\[0, 0\]}}">], keyword_args = ["", "", ""]}
  // Loop-dependent offsets flow as operands referenced from the args list.
  // CHECK: emitpy.call_opaque "pl.slice"(%arg0, %arg1, %arg2) {args = [0 : index, #emitpy.opaque<"[1, 4]">, {{\[1 : index, 2 : index\]}}, #emitpy.opaque<"[0]">], keyword_args = ["", "", "", "drop_dims"]}

  %c0 = arith.constant 0 : index
  %v = pypto.slice %src[%c0, %c0], [4, 8] : !pypto.tile<f32, [8, 16]> to !pypto.tile<f32, [4, 8]>
  %u = pypto.slice %src[%i, %j], [1, 4] drop_dims = [0] : !pypto.tile<f32, [8, 16]> to !pypto.tile<f32, [4]>
  return
}

// -----

func.func @tile_views(%a : !pypto.tile<f32, [4, 8]>) {
  // CHECK-LABEL: func.func @tile_views(
  // CHECK: emitpy.call_opaque "pl.reshape"(%arg0) {args = [0 : index, #emitpy.opaque<"[2, 16]">], keyword_args = ["", ""]}
  // CHECK: emitpy.call_opaque "pl.transpose"(%arg0) {args = [0 : index, #emitpy.opaque<"0">, #emitpy.opaque<"1">], keyword_args = ["", "", ""]}
  // CHECK: emitpy.call_opaque "pl.reinterpret_view"(%arg0) {args = [0 : index, #emitpy.opaque<"pl.INT32">], keyword_args = ["", ""]}
  // CHECK: emitpy.call_opaque "pl.cast"(%arg0) {args = [0 : index, #emitpy.opaque<"pl.FP16">], keyword_args = ["", ""]}

  %r = pypto.reshape %a : !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [2, 16]>
  %t = pypto.transpose %a, 0, 1 : !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [8, 4]>
  %v = pypto.reinterpret_view %a : !pypto.tile<f32, [4, 8]> to !pypto.tile<i32, [4, 8]>
  %c = pypto.cast %a : !pypto.tile<f32, [4, 8]> to !pypto.tile<f16, [4, 8]>
  return
}

// -----

// Comparison and selection.

func.func @tile_cmp_select(%a : !pypto.tile<f32, [4, 8]>, %b : !pypto.tile<f32, [4, 8]>) {
  // CHECK-LABEL: func.func @tile_cmp_select(
  // The predicate enum value doubles as the cmp_type code (lt == 2).
  // CHECK: %[[P:.*]] = emitpy.call_opaque "pl.cmp"(%arg0, %arg1) {args = [0 : index, 1 : index, #emitpy.opaque<"2">], keyword_args = ["", "", "cmp_type"]}
  // pl.tile.select is the composite form: the runtime synthesizes the
  // architecture-defined scratch tile.
  // CHECK: %[[S:.*]] = emitpy.call_opaque "pl.tile.select"(%[[P]], %arg0, %arg1)

  %p = pypto.cmp lt, %a, %b : !pypto.tile<f32, [4, 8]>, !pypto.tile<f32, [4, 8]> -> !pypto.tile<i1, [4, 8]>
  %s = pypto.select %p, %a, %b : !pypto.tile<i1, [4, 8]>, !pypto.tile<f32, [4, 8]>, !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  return
}

// -----

// Elementwise binaries; and/or/xor/shl/shr need an integer dtype.

func.func @tile_binary_ops(%a : !pypto.tile<f32, [4, 8]>, %b : !pypto.tile<f32, [4, 8]>, %ia : !pypto.tile<i32, [4, 8]>, %ib : !pypto.tile<i32, [4, 8]>) {
  // CHECK-LABEL: func.func @tile_binary_ops(
  // CHECK: emitpy.call_opaque "pl.add"(%arg0, %arg1)
  // CHECK: emitpy.call_opaque "pl.sub"(%arg0, %arg1)
  // CHECK: emitpy.call_opaque "pl.mul"(%arg0, %arg1)
  // CHECK: emitpy.call_opaque "pl.div"(%arg0, %arg1)
  // CHECK: emitpy.call_opaque "pl.fmod"(%arg0, %arg1)
  // CHECK: emitpy.call_opaque "pl.maximum"(%arg0, %arg1)
  // CHECK: emitpy.call_opaque "pl.minimum"(%arg0, %arg1)
  // CHECK: emitpy.call_opaque "pl.and_"(%arg2, %arg3)
  // CHECK: emitpy.call_opaque "pl.or_"(%arg2, %arg3)
  // CHECK: emitpy.call_opaque "pl.xor"(%arg2, %arg3)
  // CHECK: emitpy.call_opaque "pl.shl"(%arg2, %arg3)
  // CHECK: emitpy.call_opaque "pl.shr"(%arg2, %arg3)

  %0 = pypto.add %a, %b : !pypto.tile<f32, [4, 8]>, !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %1 = pypto.sub %a, %b : !pypto.tile<f32, [4, 8]>, !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %2 = pypto.mul %a, %b : !pypto.tile<f32, [4, 8]>, !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %3 = pypto.div %a, %b : !pypto.tile<f32, [4, 8]>, !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %4 = pypto.fmod %a, %b : !pypto.tile<f32, [4, 8]>, !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %5 = pypto.maximum %a, %b : !pypto.tile<f32, [4, 8]>, !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %6 = pypto.minimum %a, %b : !pypto.tile<f32, [4, 8]>, !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %7 = pypto.and %ia, %ib : !pypto.tile<i32, [4, 8]>, !pypto.tile<i32, [4, 8]> to !pypto.tile<i32, [4, 8]>
  %8 = pypto.or %ia, %ib : !pypto.tile<i32, [4, 8]>, !pypto.tile<i32, [4, 8]> to !pypto.tile<i32, [4, 8]>
  %9 = pypto.xor %ia, %ib : !pypto.tile<i32, [4, 8]>, !pypto.tile<i32, [4, 8]> to !pypto.tile<i32, [4, 8]>
  %10 = pypto.shl %ia, %ib : !pypto.tile<i32, [4, 8]>, !pypto.tile<i32, [4, 8]> to !pypto.tile<i32, [4, 8]>
  %11 = pypto.shr %ia, %ib : !pypto.tile<i32, [4, 8]>, !pypto.tile<i32, [4, 8]> to !pypto.tile<i32, [4, 8]>
  return
}

// -----

// Elementwise unaries.

func.func @tile_unary_ops(%a : !pypto.tile<f32, [4, 8]>) {
  // CHECK-LABEL: func.func @tile_unary_ops(
  // CHECK: emitpy.call_opaque "pl.neg"(%arg0)
  // CHECK: emitpy.call_opaque "pl.abs"(%arg0)
  // CHECK: emitpy.call_opaque "pl.exp"(%arg0)
  // CHECK: emitpy.call_opaque "pl.log"(%arg0)
  // CHECK: emitpy.call_opaque "pl.sqrt"(%arg0)
  // CHECK: emitpy.call_opaque "pl.rsqrt"(%arg0)
  // CHECK: emitpy.call_opaque "pl.sin"(%arg0)
  // CHECK: emitpy.call_opaque "pl.cos"(%arg0)

  %0 = pypto.neg %a : !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %1 = pypto.abs %a : !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %2 = pypto.exp %a : !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %3 = pypto.log %a : !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %4 = pypto.sqrt %a : !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %5 = pypto.rsqrt %a : !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %6 = pypto.sin %a : !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  %7 = pypto.cos %a : !pypto.tile<f32, [4, 8]> to !pypto.tile<f32, [4, 8]>
  return
}

// -----

// Scalar combines; the tile forms are pypto.maximum / pypto.minimum.

func.func @scalar_max_min(%a : f32, %b : f32) {
  // CHECK-LABEL: func.func @scalar_max_min(
  // CHECK: %[[M:.*]] = emitpy.call_opaque "pl.max"(%arg0, %arg1) : (f32, f32) -> f32
  // CHECK: %[[N:.*]] = emitpy.call_opaque "pl.min"(%arg0, %arg1) : (f32, f32) -> f32

  %m = pypto.max %a, %b : f32, f32 to f32
  %n = pypto.min %a, %b : f32, f32 to f32
  return
}

// -----

// Reductions: axis selects the pl.row_* / pl.col_* family, kind the suffix.
// The fold is keepdim, so the folded axis drops to extent 1.

func.func @tile_reduction(%v : !pypto.tile<f32, [4, 8]>) {
  // CHECK-LABEL: func.func @tile_reduction(
  // CHECK: %[[R:.*]] = emitpy.call_opaque "pl.create_tile"() {args = [#emitpy.opaque<"[4, 8]">, #emitpy.opaque<"pl.FP32">, #emitpy.opaque<"pl.MemorySpace.Vec">], keyword_args = ["", "dtype", "target_memory"]}
  // CHECK: %[[S:.*]] = emitpy.call_opaque "pl.row_sum"(%arg0, %[[R]])
  // CHECK: %[[M:.*]] = emitpy.call_opaque "pl.col_max"(%arg0, %[[R]])

  %r = pypto.create_tile : !pypto.tile<f32, [4, 8]>
  %s = pypto.reduction sum, row, %v, %r : !pypto.tile<f32, [4, 8]>, !pypto.tile<f32, [4, 8]> -> !pypto.tile<f32, [4, 1]>
  %m = pypto.reduction max, col, %v, %r : !pypto.tile<f32, [4, 8]>, !pypto.tile<f32, [4, 8]> -> !pypto.tile<f32, [1, 8]>
  return
}

// -----

// Scalar IO.

func.func @tile_read_write(%a : !pypto.tile<f32, [4, 8]>) {
  // CHECK-LABEL: func.func @tile_read_write(
  // CHECK: %[[S:.*]] = emitpy.call_opaque "pl.read"(%arg0) {args = [0 : index, #emitpy.opaque<"{{\[1, 2\]}}">], keyword_args = ["", ""]} : (!emitpy.opaque<"pl.Tile{{[^"]*}}">) -> f32
  // CHECK: emitpy.call_opaque "pl.write"(%arg0, %[[S]]) {args = [0 : index, #emitpy.opaque<"{{\[0, 3\]}}">, 1 : index], keyword_args = ["", "", ""]}

  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %s = pypto.read %a[1, 2] : !pypto.tile<f32, [4, 8]> -> f32
  pypto.write %a[%c0, %c3], %s : !pypto.tile<f32, [4, 8]>, f32
  return
}

// -----

func.func @matmul_fp16(%a: tensor<8x16xf16>, %b: tensor<16x12xf16>) -> tensor<8x12xf16> {
  // CHECK-LABEL: func.func @matmul_fp16(
  // CHECK: %[[OUT:.*]] = emitpy.call_opaque "pl.create_tensor"
  // CHECK: emitpy.with "pl.at(level=pl.Level.CORE_GROUP)"
  // CHECK: "pl.slice"(%arg0
  // CHECK-SAME: "pad_value", "clamp"
  // CHECK: "pl.fillpad_expand"
  // CHECK: "pl.matmul"
  // CHECK-SAME: #emitpy.opaque<"pl.FP32">
  // CHECK: "pl.cast"
  // CHECK-SAME: #emitpy.opaque<"pl.FP16">
  // CHECK: %[[TARGET:.*]] = emitpy.tensor_slice %[[OUT]]
  // CHECK: emitpy.assign %[[TARGET]]
  // CHECK: return %[[OUT]]
  %r = pypto.matmul %a, %b : (tensor<8x16xf16>, tensor<16x12xf16>) -> tensor<8x12xf16>
  return %r : tensor<8x12xf16>
}

func.func @matmul_fp16_init(%a: tensor<130x129xf16>, %b: tensor<129x70xf16>,
                          %init: tensor<130x70xf16>) -> tensor<130x70xf16> {
  // CHECK-LABEL: func.func @matmul_fp16_init(
  // CHECK: %[[OUT:.*]] = emitpy.call_opaque "pl.create_tensor"
  // CHECK: %[[M:.*]] = emitpy.call_opaque "pl.matmul"
  // CHECK-SAME: #emitpy.opaque<"pl.FP32">
  // CHECK: emitpy.for
  // CHECK: %[[NEXT:[a-zA-Z0-9_]+]] = emitpy.call_opaque "pl.matmul_acc"(%[[M]]
  // CHECK-NEXT: emitpy.assign %[[M]] = %[[NEXT]]
  // CHECK: "pl.slice"(%arg2
  // CHECK: %[[I:.*]] = emitpy.call_opaque "pl.cast"
  // CHECK-SAME: #emitpy.opaque<"pl.FP32">
  // CHECK-NEXT: %[[A:.*]] = emitpy.call_opaque "pl.add"(%[[M]], %[[I]])
  // CHECK-NEXT: emitpy.call_opaque "pl.cast"(%[[A]])
  // CHECK-SAME: #emitpy.opaque<"pl.FP16">
  // CHECK: "pl.set_validshape"
  // CHECK: emitpy.assign
  // CHECK: return %[[OUT]]
  // PYTHON-LABEL: def matmul_fp16_init(
  // PYTHON: for [[TM:.*]] in pl.parallel(0, 130, 64):
  // PYTHON-NEXT: for [[TN:.*]] in pl.parallel(0, 70, 64):
  // PYTHON-NEXT: with pl.at(level=pl.Level.CORE_GROUP):
  // PYTHON-NEXT: {{.*}} = pl.slice(inputs, [64, 64], {{\[}}[[TM]], 0], pad_value=0, clamp=True)
  // PYTHON: pl.matmul(
  // PYTHON-NEXT: for [[TK:.*]] in pl.range(64, 129, 64):
  // PYTHON-NEXT: {{.*}} = pl.slice(inputs, [64, 64], {{\[}}[[TM]], [[TK]]], pad_value=0, clamp=True)
  // PYTHON: pl.matmul_acc(
  // PYTHON: pl.slice(inputs_1, [64, 64], {{\[}}[[TM]], [[TN]]], pad_value=0, clamp=True)
  // PYTHON: pl.set_validshape(
  // PYTHON-NOT: for
  // PYTHON-NOT: if
  // PYTHON: return
  %r = pypto.matmul %a, %b init %init : (tensor<130x129xf16>, tensor<129x70xf16>, tensor<130x70xf16>) -> tensor<130x70xf16>
  return %r : tensor<130x70xf16>
}

func.func @matmul_large(%a: tensor<384x384xf16>, %b: tensor<384x384xf16>) -> tensor<384x384xf16> {
  // CHECK-LABEL: func.func @matmul_large(
  // CHECK: emitpy.for
  // CHECK: emitpy.for
  // CHECK: emitpy.with
  // CHECK: "pl.matmul"
  // CHECK: emitpy.for
  // CHECK: "pl.matmul_acc"
  // CHECK: emitpy.assign
  // PYTHON-LABEL: def matmul_large(
  // PYTHON-NOT: pl.slice
  // PYTHON-NOT: pl.fillpad_expand
  // PYTHON: for [[M:.*]] in pl.parallel(0, 384, 64):
  // PYTHON-NEXT: for [[N:.*]] in pl.parallel(0, 384, 64):
  // PYTHON-NEXT: with pl.at(level=pl.Level.CORE_GROUP):
  // PYTHON-NEXT: [[ACC:[a-zA-Z0-9_]+]] = pl.matmul(inputs{{\[}}[[M]]:[[M]] + 64, 0:64], inputs_0{{\[}}0:64, [[N]]:[[N]] + 64], out_dtype=pl.FP32)
  // PYTHON-NEXT: for [[K:.*]] in pl.range(64, 384, 64):
  // PYTHON-NEXT: [[NEXT:[a-zA-Z0-9_]+]] = pl.matmul_acc([[ACC]], inputs{{\[}}[[M]]:[[M]] + 64, [[K]]:[[K]] + 64], inputs_0{{\[}}[[K]]:[[K]] + 64, [[N]]:[[N]] + 64])
  // PYTHON-NEXT: [[ACC]] = [[NEXT]]
  // PYTHON-NEXT: [[RESULT:[a-zA-Z0-9_]+]] = pl.cast([[ACC]], pl.FP16, mode="rint")
  // PYTHON-NEXT: {{.*}}[[M]]:[[M]] + 64, [[N]]:[[N]] + 64] = [[RESULT]]
  // PYTHON-NEXT: return
  %r = pypto.matmul %a, %b : (tensor<384x384xf16>, tensor<384x384xf16>) -> tensor<384x384xf16>
  return %r : tensor<384x384xf16>
}

// Mixed boundary: two row blocks and one padded column/reduction block.
func.func @matmul_65x63(%a: tensor<65x63xf16>, %b: tensor<63x63xf16>) -> tensor<65x63xf16> {
  // CHECK-LABEL: func.func @matmul_65x63(
  // CHECK: "pl.slice"(%arg0
  // CHECK-SAME: "pad_value", "clamp"
  // CHECK: "pl.fillpad_expand"
  // CHECK: "pl.slice"(%arg1
  // CHECK: "pl.matmul"
  // CHECK-NOT: "pl.matmul_acc"
  // CHECK: "pl.set_validshape"
  // CHECK: emitpy.assign
  // PYTHON-LABEL: def matmul_65x63(
  // PYTHON: for [[ROW:.*]] in pl.parallel(0, 65, 64):
  // PYTHON-NEXT: for {{.*}} in pl.parallel(0, 63, 64):
  // PYTHON: pl.slice(inputs, [64, 64], {{\[}}[[ROW]], 0], pad_value=0, clamp=True)
  // PYTHON: pl.matmul(
  // PYTHON-NOT: pl.matmul_acc
  // PYTHON: [[REMAIN:[a-zA-Z0-9_]+]] = (65 - [[ROW]])
  // PYTHON-NEXT: [[VALID:[a-zA-Z0-9_]+]] = pl.min(64, [[REMAIN]])
  // PYTHON-NEXT: {{.*}} = pl.set_validshape({{.*}}, [[VALID]], 63)
  // PYTHON-NOT: for
  // PYTHON-NOT: if
  // PYTHON: return
  %r = pypto.matmul %a, %b : (tensor<65x63xf16>, tensor<63x63xf16>) -> tensor<65x63xf16>
  return %r : tensor<65x63xf16>
}
