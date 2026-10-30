// RUN: akg-opt %s --npuvector-vectorize --mlir-disable-threading --allow-unregistered-dialect | FileCheck %s
// RUN: akg-opt %s --npuvector-vectorize --mlir-disable-threading --allow-unregistered-dialect | FileCheck %s --check-prefix=NO-SCRATCH

// CHECK-DAG: #[[SKIP_MIDDLE_MAP:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[SCRATCH_LB_MAP:map[0-9]*]] = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 2) * 4)>
// CHECK-DAG: #[[SCRATCH_UB_MAP:map[0-9]*]] = affine_map<(d0) -> (d0 * 2 - (d0 floordiv 2) * 4 + 2)>

// ============================================================================
// 1. 1-D Elementwise static and dynamic
// ============================================================================

// CHECK-LABEL: func.func @test_skip_host_func
// CHECK-NOT: npuvector.transfer_read
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: npuvector.transfer_write
// CHECK: return
func.func @test_skip_host_func(
    %X: memref<1024xf32>,
    %Y: memref<1024xf32>) attributes {hacc.function_kind = #hacc.function_kind<HOST>} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index

  scf.for %i = %c0 to %c1024 step %c1 {
    %x = memref.load %X[%i] : memref<1024xf32>
    memref.store %x, %Y[%i] : memref<1024xf32>
  } {vector=1024}
  return
}

// -----

// CHECK-LABEL: func.func @test_elementwise_static
func.func @test_elementwise_static(
    %X: memref<1024xf32>,
    %Y: memref<1024xf32>,
    %Z: memref<1024xf32>,
    %alpha: f32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[MAX:.*]] = arith.constant 1024 : index
  // CHECK: %[[X_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [{{.*}}] [{{.*}}], %{{.*}} : memref<1024xf32>, !npuvector<1024xf32>
  // CHECK: %[[Y_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [{{.*}}] [{{.*}}], %{{.*}} : memref<1024xf32>, !npuvector<1024xf32>
  // CHECK: %[[ALPHA_VEC:.*]] = npuvector.broadcast %{{.*}}[{{.*}}] [{{.*}}] : f32 to !npuvector<1024xf32>
  // CHECK: %[[AX_VEC:.*]] = arith.mulf %[[ALPHA_VEC]], %[[X_VEC]] : !npuvector<1024xf32>
  // CHECK: %[[RESULT_VEC:.*]] = arith.addf %[[AX_VEC]], %[[Y_VEC]] : !npuvector<1024xf32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<1024xf32>, memref<1024xf32>
  // CHECK-NOT: scf.for
  // CHECK: return

  scf.for %i = %c0 to %c1024 step %c1 {
    %x = memref.load %X[%i] : memref<1024xf32>
    %y = memref.load %Y[%i] : memref<1024xf32>
    %ax = arith.mulf %alpha, %x : f32
    %result = arith.addf %ax, %y : f32
    memref.store %result, %Z[%i] : memref<1024xf32>
  } {vector=1024}
  return
}

// -----

// CHECK-LABEL: func.func @test_elementwise_dynamic
func.func @test_elementwise_dynamic(
    %input: memref<?xf32>,
    %output: memref<?xf32>,
    %scale: f32,
    %bias: f32) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %N = memref.dim %input, %c0 : memref<?xf32>

  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
  // CHECK: %[[MAX:.*]] = arith.constant 4096 : index
  // CHECK: %[[VEC_SIZE:.*]] = affine.min
  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VEC_SIZE]]] [%[[MAX]]], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK: %[[SCALE_VEC:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
  // CHECK: %[[SCALED_VEC:.*]] = arith.mulf %[[V_VEC]], %[[SCALE_VEC]] : !npuvector<?xf32>
  // CHECK: %[[BIAS_VEC:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
  // CHECK: %[[RESULT_VEC:.*]] = arith.addf %[[SCALED_VEC]], %[[BIAS_VEC]] : !npuvector<?xf32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<?xf32>, memref<?xf32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %input[%i] : memref<?xf32>
    %scaled = arith.mulf %v, %scale : f32
    %result = arith.addf %scaled, %bias : f32
    memref.store %result, %output[%i] : memref<?xf32>
  } {vector=4096}
  return
}

// -----

// ============================================================================
// 2. 1-D Reduction static and dynamic
// ============================================================================

// CHECK-LABEL: func.func @test_reduction_static
func.func @test_reduction_static(%input: memref<1024xf32>, %output: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32

  // CHECK: %[[VF:.*]] = arith.constant 128 : index
  // CHECK: %[[NEUTRAL_SCALAR:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[NEUTRAL:.*]] = npuvector.broadcast %[[NEUTRAL_SCALAR]]{{.*}} : f32 to !npuvector<128xf32>
  // CHECK: %[[SUM_VEC:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC_VEC:.*]] = %[[NEUTRAL]]) -> (!npuvector<128xf32>) {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VF]]], %{{.*}} : memref<1024xf32>, !npuvector<128xf32>
  // CHECK:   %[[NEW_ACC_VEC:.*]] = arith.addf %[[ACC_VEC]], %[[V_VEC]] {reduction_type = "all"} : !npuvector<128xf32>
  // CHECK:   scf.yield %[[NEW_ACC_VEC]] : !npuvector<128xf32>
  // CHECK: }
  // CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %[[SUM_VEC]] {reduction_dims = array<i64: 0>} : !npuvector<128xf32> into !npuvector.f32
  // CHECK: npuvector.transfer_write %[[SUM]], %{{.*}}[]{{.*}}: !npuvector.f32, memref<f32>

  %sum = scf.for %i = %c0 to %c1024 step %c1 iter_args(%acc = %init) -> f32 {
    %v = memref.load %input[%i] : memref<1024xf32>
    %new_acc = arith.addf %acc, %v {reduction_type = "all"} : f32
    scf.yield %new_acc : f32
  } {reduction_all=128}

  memref.store %sum, %output[] : memref<f32>
  return
}

// -----

// CHECK-LABEL: func.func @test_reduction_static_truncf
// CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %{{.*}} {reduction_dims = array<i64: 0>} : !npuvector<128xf32> into !npuvector.f32
// CHECK: %[[ONE:.*]] = npuvector.broadcast %{{.*}} : f32 to !npuvector.f32
// CHECK: %[[BIASED:.*]] = arith.addf %[[SUM]], %[[ONE]] : !npuvector.f32
// CHECK: %[[TRUNC:.*]] = npuvector.truncf %[[BIASED]] {round_mode = #hivm.round_mode<round>} : !npuvector.f32 to !npuvector.bf16
// CHECK: npuvector.transfer_write %[[TRUNC]], %{{.*}}[]{{.*}}: !npuvector.bf16, memref<bf16>
func.func @test_reduction_static_truncf(%input: memref<1024xf32>, %output: memref<bf16>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32

  %sum = scf.for %i = %c0 to %c1024 step %c1 iter_args(%acc = %init) -> f32 {
    %v = memref.load %input[%i] : memref<1024xf32>
    %new_acc = arith.addf %acc, %v {reduction_type = "all"} : f32
    scf.yield %new_acc : f32
  } {reduction_all=128}

  %biased = arith.addf %sum, %one : f32
  %truncated = arith.truncf %biased {round_mode = #hivm.round_mode<round>} : f32 to bf16
  memref.store %truncated, %output[] : memref<bf16>
  return
}

// -----

// CHECK-LABEL: func.func @test_reduction_dynamic
func.func @test_reduction_dynamic(%input: memref<?xf32>, %output: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %N = memref.dim %input, %c0 : memref<?xf32>
  %init = arith.constant 0.0 : f32

  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[VF:.*]] = arith.constant 128 : index
  // CHECK: %[[VEC_SIZE:.*]] = affine.min
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[NEUTRAL:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[VF]]] : f32 to !npuvector<?xf32>
  // CHECK-NOT: arith.divsi
  // CHECK: %[[SUM_VEC:.*]] = scf.for %[[IV:.*]] = %{{.*}} to %[[DIM]] step %[[VEC_SIZE]] iter_args(%[[ACC_VEC:.*]] = %[[NEUTRAL]]) -> (!npuvector<?xf32>) {
  // CHECK:   %[[REMAIN:.*]] = affine.apply
  // CHECK:   %[[TILE:.*]] = affine.min
  // CHECK:   %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK:   %[[STRIDE:.*]] = arith.constant 1 : index
  // CHECK:   %[[ACC_TILE:.*]] = npuvector.extract_slice %[[ACC_VEC]][%[[OFFSET]]] [%[[TILE]]] [%[[STRIDE]]] {keep_dims = array<i64: 0>} : !npuvector<?xf32> to !npuvector<?xf32>
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%[[IV]]] [%[[TILE]]] [%[[VF]]], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   %[[NEW_ACC_TILE:.*]] = arith.addf %[[ACC_TILE]], %[[V_VEC]] {reduction_type = "all"} : !npuvector<?xf32>
  // CHECK:   %[[NEW_ACC_VEC:.*]] = npuvector.insert_slice %[[NEW_ACC_TILE]] into %[[ACC_VEC]][%[[OFFSET]]] [%[[TILE]]] [%[[STRIDE]]] : !npuvector<?xf32> into !npuvector<?xf32>
  // CHECK:   scf.yield %[[NEW_ACC_VEC]] : !npuvector<?xf32>
  // CHECK: }
  // CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %[[SUM_VEC]] {{.*}} : !npuvector<?xf32> into !npuvector.f32
  // CHECK-NOT: arith.remsi
  // CHECK-NOT: scf.if
  // CHECK: npuvector.transfer_write %[[SUM]], %{{.*}}[]{{.*}}: !npuvector.f32, memref<f32>

  %sum = scf.for %i = %c0 to %N step %c1 iter_args(%acc = %init) -> f32 {
    %v = memref.load %input[%i] : memref<?xf32>
    %new_acc = arith.addf %acc, %v {reduction_type = "all"} : f32
    scf.yield %new_acc : f32
  } {reduction_all=128}

  memref.store %sum, %output[] : memref<f32>
  return
}

// -----

// Two independent reductions along the same axis (e.g. sum and sum-of-squares), both reduction_x lanes.

// CHECK-LABEL: func.func @test_reduction_x_two_results
func.func @test_reduction_x_two_results(
    %input: memref<1024xf32>,
    %mask: memref<1024xf32>,
    %out_sum: memref<f32>,
    %out_sq: memref<f32>) attributes {arch = "Ascend950DT_9581", hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32

  // CHECK: %[[VF:.*]] = arith.constant 128 : index
  // CHECK: %[[N0_SCALAR:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[N0:.*]] = npuvector.broadcast %[[N0_SCALAR]]{{.*}} : f32 to !npuvector<128xf32>
  // CHECK: %[[N1_SCALAR:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[N1:.*]] = npuvector.broadcast %[[N1_SCALAR]]{{.*}} : f32 to !npuvector<128xf32>
  // CHECK: %{{.*}}:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC0:.*]] = %[[N0]], %[[ACC1:.*]] = %[[N1]]) -> (!npuvector<128xf32>, !npuvector<128xf32>) {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VF]]], %{{.*}} : memref<1024xf32>, !npuvector<128xf32>
  // CHECK:   %[[M_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VF]]], %{{.*}} : memref<1024xf32>, !npuvector<128xf32>
  // CHECK:   %[[VM:.*]] = arith.mulf %[[V_VEC]], %[[M_VEC]] : !npuvector<128xf32>
  // CHECK:   %[[VV:.*]] = arith.mulf %[[V_VEC]], %[[V_VEC]] : !npuvector<128xf32>
  // CHECK:   %[[A0:.*]] = arith.addf %[[ACC0]], %[[VM]] {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<128xf32>
  // CHECK:   %[[A1:.*]] = arith.addf %[[ACC1]], %[[VV]] {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<128xf32>
  // CHECK:   scf.yield %[[A0]], %[[A1]] : !npuvector<128xf32>, !npuvector<128xf32>
  // CHECK: }
  // CHECK: %[[R0:.*]] = npuvector.reduction <add>, %{{.*}} {{.*}} : !npuvector<128xf32> into !npuvector.f32
  // CHECK: %[[R1:.*]] = npuvector.reduction <add>, %{{.*}} {{.*}} : !npuvector<128xf32> into !npuvector.f32
  // CHECK: npuvector.transfer_write %[[R0]], %{{.*}}[]{{.*}}: !npuvector.f32, memref<f32>
  // CHECK: npuvector.transfer_write %[[R1]], %{{.*}}[]{{.*}}: !npuvector.f32, memref<f32>

  %pair:2 = scf.for %i = %c0 to %c1024 step %c1 iter_args(%a0 = %init, %a1 = %init) -> (f32, f32) {
    %v = memref.load %input[%i] : memref<1024xf32>
    %m = memref.load %mask[%i] : memref<1024xf32>
    %vm = arith.mulf %v, %m : f32
    %vv = arith.mulf %v, %v : f32
    %n0 = arith.addf %a0, %vm {reduction_axes = [0 : index], reduction_type = "x"} : f32
    %n1 = arith.addf %a1, %vv {reduction_axes = [0 : index], reduction_type = "x"} : f32
    scf.yield %n0, %n1 : f32, f32
  } {reduction_x = 128 : i64}

  memref.store %pair#0, %out_sum[] : memref<f32>
  memref.store %pair#1, %out_sq[] : memref<f32>
  return
}

// -----
// Two reduction_x lanes: body [0, 256) at VF 128, then a static 44-element tail.
// Each lane reuses its full accumulator and reduces once after its static tail.
// CHECK-LABEL: func.func @test_reduction_x_two_results_static_tail(
// CHECK: %[[PAIR_INIT0:.*]] = arith.constant 3.000000e+00 : f32
// CHECK: %[[PAIR_INIT1:.*]] = arith.constant 5.000000e+00 : f32
// CHECK: %[[PAIR_BSCALAR0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[PAIR_BZERO0:.*]] = npuvector.broadcast %[[PAIR_BSCALAR0]]{{.*}} : f32 to !npuvector<128xf32>
// CHECK: %[[PAIR_BSCALAR1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[PAIR_BZERO1:.*]] = npuvector.broadcast %[[PAIR_BSCALAR1]]{{.*}} : f32 to !npuvector<128xf32>
// CHECK: %[[PAIR_SPLIT:.*]] = arith.constant 256 : index
// CHECK: %[[PAIR_BODY:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %[[PAIR_SPLIT]] step %{{.*}} iter_args(%[[PAIR_BACC0:.*]] = %[[PAIR_BZERO0]], %[[PAIR_BACC1:.*]] = %[[PAIR_BZERO1]]) -> (!npuvector<128xf32>, !npuvector<128xf32>) {
// CHECK: %[[PAIR_BV:.*]] = npuvector.transfer_read {{.*}} : memref<300xf32>, !npuvector<128xf32>
// CHECK: %[[PAIR_BM:.*]] = npuvector.transfer_read {{.*}} : memref<300xf32>, !npuvector<128xf32>
// CHECK: %[[PAIR_BVM:.*]] = arith.mulf %[[PAIR_BV]], %[[PAIR_BM]] : !npuvector<128xf32>
// CHECK: %[[PAIR_BVV:.*]] = arith.mulf %[[PAIR_BV]], %[[PAIR_BV]] : !npuvector<128xf32>
// CHECK: %[[PAIR_BNEXT0:.*]] = arith.addf %[[PAIR_BACC0]], %[[PAIR_BVM]] {{.*}} : !npuvector<128xf32>
// CHECK: %[[PAIR_BNEXT1:.*]] = arith.addf %[[PAIR_BACC1]], %[[PAIR_BVV]] {{.*}} : !npuvector<128xf32>
// CHECK: scf.yield %[[PAIR_BNEXT0]], %[[PAIR_BNEXT1]] : !npuvector<128xf32>, !npuvector<128xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[PAIR_TAIL_WIDTH:.*]] = arith.constant 44 : index
// CHECK: %[[PAIR_TAIL:.*]]:2 = scf.for %{{.*}} = %[[PAIR_SPLIT]] to %{{.*}} step %[[PAIR_TAIL_WIDTH]] iter_args(%[[PAIR_TACC0:.*]] = %[[PAIR_BODY]]#0, %[[PAIR_TACC1:.*]] = %[[PAIR_BODY]]#1) -> (!npuvector<128xf32>, !npuvector<128xf32>) {
// CHECK: %[[PAIR_SLICE0:.*]] = npuvector.extract_slice %[[PAIR_TACC0]][%[[PAIR_OFF0:.*]]] [%[[PAIR_TAIL_WIDTH]]] [%[[PAIR_STRIDE0:.*]]] {keep_dims = array<i64: 0>} : !npuvector<128xf32> to !npuvector<44xf32>
// CHECK: %[[PAIR_SLICE1:.*]] = npuvector.extract_slice %[[PAIR_TACC1]][%[[PAIR_OFF1:.*]]] [%[[PAIR_TAIL_WIDTH]]] [%[[PAIR_STRIDE1:.*]]] {keep_dims = array<i64: 0>} : !npuvector<128xf32> to !npuvector<44xf32>
// CHECK: %[[PAIR_TV:.*]] = npuvector.transfer_read {{.*}} : memref<300xf32>, !npuvector<44xf32>
// CHECK: %[[PAIR_TM:.*]] = npuvector.transfer_read {{.*}} : memref<300xf32>, !npuvector<44xf32>
// CHECK: %[[PAIR_TVM:.*]] = arith.mulf %[[PAIR_TV]], %[[PAIR_TM]] : !npuvector<44xf32>
// CHECK: %[[PAIR_TVV:.*]] = arith.mulf %[[PAIR_TV]], %[[PAIR_TV]] : !npuvector<44xf32>
// CHECK: %[[PAIR_TNEXT0:.*]] = arith.addf %[[PAIR_SLICE0]], %[[PAIR_TVM]] {{.*}} : !npuvector<44xf32>
// CHECK: %[[PAIR_TNEXT1:.*]] = arith.addf %[[PAIR_SLICE1]], %[[PAIR_TVV]] {{.*}} : !npuvector<44xf32>
// CHECK: %[[PAIR_INSERT0:.*]] = npuvector.insert_slice %[[PAIR_TNEXT0]] into %[[PAIR_TACC0]][%[[PAIR_OFF0]]] [%[[PAIR_TAIL_WIDTH]]] [%[[PAIR_STRIDE0]]] : !npuvector<44xf32> into !npuvector<128xf32>
// CHECK: %[[PAIR_INSERT1:.*]] = npuvector.insert_slice %[[PAIR_TNEXT1]] into %[[PAIR_TACC1]][%[[PAIR_OFF1]]] [%[[PAIR_TAIL_WIDTH]]] [%[[PAIR_STRIDE1]]] : !npuvector<44xf32> into !npuvector<128xf32>
// CHECK-NEXT: scf.yield %[[PAIR_INSERT0]], %[[PAIR_INSERT1]] : !npuvector<128xf32>, !npuvector<128xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[PAIR_REDUCED0:.*]] = npuvector.reduction <add>, %[[PAIR_TAIL]]#0 {{.*}} : !npuvector<128xf32> into !npuvector.f32
// CHECK-NEXT: %[[PAIR_INIT_VEC0:.*]] = npuvector.broadcast %[[PAIR_INIT0]] : f32 to !npuvector.f32
// CHECK-NEXT: %[[PAIR_SUM0:.*]] = arith.addf %[[PAIR_REDUCED0]], %[[PAIR_INIT_VEC0]] : !npuvector.f32
// CHECK-NEXT: %[[PAIR_REDUCED1:.*]] = npuvector.reduction <add>, %[[PAIR_TAIL]]#1 {{.*}} : !npuvector<128xf32> into !npuvector.f32
// CHECK-NEXT: %[[PAIR_INIT_VEC1:.*]] = npuvector.broadcast %[[PAIR_INIT1]] : f32 to !npuvector.f32
// CHECK-NEXT: %[[PAIR_SUM1:.*]] = arith.addf %[[PAIR_REDUCED1]], %[[PAIR_INIT_VEC1]] : !npuvector.f32
// CHECK-NOT: npuvector.reduction
// CHECK: npuvector.transfer_write %[[PAIR_SUM0]], {{.*}} : !npuvector.f32, memref<f32>
// CHECK: npuvector.transfer_write %[[PAIR_SUM1]], {{.*}} : !npuvector.f32, memref<f32>
// CHECK: return
func.func @test_reduction_x_two_results_static_tail(
    %input: memref<300xf32>,
    %mask: memref<300xf32>,
    %out_sum: memref<f32>,
    %out_sq: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c300 = arith.constant 300 : index
  %c1 = arith.constant 1 : index
  %init0 = arith.constant 3.0 : f32
  %init1 = arith.constant 5.0 : f32

  %pair:2 = scf.for %i = %c0 to %c300 step %c1 iter_args(%a0 = %init0, %a1 = %init1) -> (f32, f32) {
    %v = memref.load %input[%i] : memref<300xf32>
    %m = memref.load %mask[%i] : memref<300xf32>
    %vm = arith.mulf %v, %m : f32
    %vv = arith.mulf %v, %v : f32
    %n0 = arith.addf %a0, %vm {reduction_axes = [0 : index], reduction_type = "x"} : f32
    %n1 = arith.addf %a1, %vv {reduction_axes = [0 : index], reduction_type = "x"} : f32
    scf.yield %n0, %n1 : f32, f32
  } {reduction_x = 128 : i64}

  memref.store %pair#0, %out_sum[] : memref<f32>
  memref.store %pair#1, %out_sq[] : memref<f32>
  return
}

// -----
// Two reduction_x lanes; dynamic dim uses in-loop min instead of per-lane tail replay.

// CHECK-LABEL: func.func @test_reduction_x_two_results_dynamic_tail
func.func @test_reduction_x_two_results_dynamic_tail(
    %input: memref<?xf32>,
    %mask: memref<?xf32>,
    %out_sum: memref<f32>,
    %out_sq: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %N = memref.dim %input, %c0 : memref<?xf32>
  %init = arith.constant 0.0 : f32
  %scale = arith.constant 2.0 : f32

  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SCALE:.*]] = arith.constant 2.000000e+00 : f32
  // CHECK: %[[VF:.*]] = arith.constant 128 : index
  // CHECK: %[[VEC_SIZE:.*]] = affine.min
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[N0:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[VF]]] : f32 to !npuvector<?xf32>
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[N1:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[VF]]] : f32 to !npuvector<?xf32>
  // CHECK-NOT: arith.divsi
  // CHECK: %{{.*}}:2 = scf.for %[[IV:.*]] = %{{.*}} to %[[DIM]] step %[[VEC_SIZE]] iter_args(%[[ACC0:.*]] = %[[N0]], %[[ACC1:.*]] = %[[N1]]) -> (!npuvector<?xf32>, !npuvector<?xf32>) {
  // CHECK:   %[[REMAIN:.*]] = affine.apply
  // CHECK:   %[[TILE:.*]] = affine.min
  // CHECK:   %[[ACC0_OFFSET:.*]] = arith.constant 0 : index
  // CHECK:   %[[ACC0_STRIDE:.*]] = arith.constant 1 : index
  // CHECK:   %[[ACC0_TILE:.*]] = npuvector.extract_slice %[[ACC0]][%[[ACC0_OFFSET]]] [%[[TILE]]] [%[[ACC0_STRIDE]]] {keep_dims = array<i64: 0>} : !npuvector<?xf32> to !npuvector<?xf32>
  // CHECK:   %[[ACC1_OFFSET:.*]] = arith.constant 0 : index
  // CHECK:   %[[ACC1_STRIDE:.*]] = arith.constant 1 : index
  // CHECK:   %[[ACC1_TILE:.*]] = npuvector.extract_slice %[[ACC1]][%[[ACC1_OFFSET]]] [%[[TILE]]] [%[[ACC1_STRIDE]]] {keep_dims = array<i64: 0>} : !npuvector<?xf32> to !npuvector<?xf32>
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%[[IV]]] [%[[TILE]]] [%[[VF]]], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   %[[M_VEC:.*]] = npuvector.transfer_read %{{.*}}[%[[IV]]] [%[[TILE]]] [%[[VF]]], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   %[[VM0:.*]] = arith.mulf %[[V_VEC]], %[[M_VEC]] : !npuvector<?xf32>
  // CHECK:   %[[SCALE_VEC:.*]] = npuvector.broadcast %[[SCALE]][%[[TILE]]] [%[[VF]]] : f32 to !npuvector<?xf32>
  // CHECK:   %[[VM:.*]] = arith.mulf %[[VM0]], %[[SCALE_VEC]] : !npuvector<?xf32>
  // CHECK:   %[[VV:.*]] = arith.mulf %[[V_VEC]], %[[V_VEC]] : !npuvector<?xf32>
  // CHECK:   %[[A0:.*]] = arith.addf %[[ACC0_TILE]], %[[VM]] {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<?xf32>
  // CHECK:   %[[A1:.*]] = arith.addf %[[ACC1_TILE]], %[[VV]] {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<?xf32>
  // CHECK:   %[[Y0:.*]] = npuvector.insert_slice %[[A0]] into %[[ACC0]][%[[ACC0_OFFSET]]] [%[[TILE]]] [%[[ACC0_STRIDE]]] : !npuvector<?xf32> into !npuvector<?xf32>
  // CHECK:   %[[Y1:.*]] = npuvector.insert_slice %[[A1]] into %[[ACC1]][%[[ACC1_OFFSET]]] [%[[TILE]]] [%[[ACC1_STRIDE]]] : !npuvector<?xf32> into !npuvector<?xf32>
  // CHECK:   scf.yield %[[Y0]], %[[Y1]] : !npuvector<?xf32>, !npuvector<?xf32>
  // CHECK: }
  // CHECK: npuvector.reduction <add>, %{{.*}} {{.*}} : !npuvector<?xf32> into !npuvector.f32
  // CHECK-NOT: arith.remsi
  // CHECK-NOT: scf.if
  // CHECK: npuvector.reduction <add>, %{{.*}} {{.*}} : !npuvector<?xf32> into !npuvector.f32
  // CHECK-NOT: arith.remsi
  // CHECK-NOT: scf.if
  // CHECK: npuvector.transfer_write {{.*}}[]{{.*}}: !npuvector.f32, memref<f32>
  // CHECK: npuvector.transfer_write {{.*}}[]{{.*}}: !npuvector.f32, memref<f32>

  %pair:2 = scf.for %i = %c0 to %N step %c1 iter_args(%a0 = %init, %a1 = %init) -> (f32, f32) {
    %v = memref.load %input[%i] : memref<?xf32>
    %m = memref.load %mask[%i] : memref<?xf32>
    %vm0 = arith.mulf %v, %m : f32
    %vm = arith.mulf %vm0, %scale : f32
    %vv = arith.mulf %v, %v : f32
    %n0 = arith.addf %a0, %vm {reduction_axes = [0 : index], reduction_type = "x"} : f32
    %n1 = arith.addf %a1, %vv {reduction_axes = [0 : index], reduction_type = "x"} : f32
    scf.yield %n0, %n1 : f32, f32
  } {reduction_x = 128 : i64}

  memref.store %pair#0, %out_sum[] : memref<f32>
  memref.store %pair#1, %out_sq[] : memref<f32>
  return
}

// -----

// ============================================================================
// 3. 1-D tail handling static and dynamic
// Static: 300 elements, VF 128 -> body [0, 256), then a static 44-element tail.
// ============================================================================
// CHECK-LABEL: func.func @test_reduction_static_with_tail(
// CHECK: %[[SUM_INIT:.*]] = arith.constant 3.000000e+00 : f32
// CHECK: %[[SUM_BSCALAR:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[SUM_BZERO:.*]] = npuvector.broadcast %[[SUM_BSCALAR]]{{.*}} : f32 to !npuvector<128xf32>
// CHECK: %[[SUM_SPLIT:.*]] = arith.constant 256 : index
// CHECK: %[[SUM_BODY:.*]] = scf.for %{{.*}} = %{{.*}} to %[[SUM_SPLIT]] step %{{.*}} iter_args(%[[SUM_BACC:.*]] = %[[SUM_BZERO]]) -> (!npuvector<128xf32>) {
// CHECK: %[[SUM_BVAL:.*]] = npuvector.transfer_read {{.*}} : memref<300xf32>, !npuvector<128xf32>
// CHECK: %[[SUM_BNEXT:.*]] = arith.addf %[[SUM_BACC]], %[[SUM_BVAL]] {{.*}} : !npuvector<128xf32>
// CHECK-NEXT: scf.yield %[[SUM_BNEXT]] : !npuvector<128xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[SUM_TAIL_WIDTH:.*]] = arith.constant 44 : index
// CHECK: %[[SUM_TAIL:.*]] = scf.for %{{.*}} = %[[SUM_SPLIT]] to %{{.*}} step %[[SUM_TAIL_WIDTH]] iter_args(%[[SUM_TACC:.*]] = %[[SUM_BODY]]) -> (!npuvector<128xf32>) {
// CHECK: %[[SUM_SLICE:.*]] = npuvector.extract_slice %[[SUM_TACC]][%[[SUM_OFFSET:.*]]] [%[[SUM_TAIL_WIDTH]]] [%[[SUM_STRIDE:.*]]] {keep_dims = array<i64: 0>} : !npuvector<128xf32> to !npuvector<44xf32>
// CHECK: %[[SUM_TVAL:.*]] = npuvector.transfer_read {{.*}} : memref<300xf32>, !npuvector<44xf32>
// CHECK: %[[SUM_TNEXT:.*]] = arith.addf %[[SUM_SLICE]], %[[SUM_TVAL]] {{.*}} : !npuvector<44xf32>
// CHECK-NEXT: %[[SUM_INSERT:.*]] = npuvector.insert_slice %[[SUM_TNEXT]] into %[[SUM_TACC]][%[[SUM_OFFSET]]] [%[[SUM_TAIL_WIDTH]]] [%[[SUM_STRIDE]]] : !npuvector<44xf32> into !npuvector<128xf32>
// CHECK-NEXT: scf.yield %[[SUM_INSERT]] : !npuvector<128xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[SUM_REDUCED:.*]] = npuvector.reduction <add>, %[[SUM_TAIL]] {{.*}} : !npuvector<128xf32> into !npuvector.f32
// CHECK-NEXT: %[[SUM_INIT_VEC:.*]] = npuvector.broadcast %[[SUM_INIT]] : f32 to !npuvector.f32
// CHECK-NEXT: %[[SUM_RESULT:.*]] = arith.addf %[[SUM_REDUCED]], %[[SUM_INIT_VEC]] : !npuvector.f32
// CHECK-NEXT: npuvector.transfer_write %[[SUM_RESULT]], {{.*}} : !npuvector.f32, memref<f32>
// CHECK-NOT: npuvector.reduction
// CHECK: return
func.func @test_reduction_static_with_tail(%input: memref<300xf32>, %output: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c300 = arith.constant 300 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 3.0 : f32

  %sum = scf.for %i = %c0 to %c300 step %c1 iter_args(%acc = %init) -> f32 {
    %v = memref.load %input[%i] : memref<300xf32>
    %new_acc = arith.addf %acc, %v {reduction_type = "all"} : f32
    scf.yield %new_acc : f32
  } {reduction_all=128}

  memref.store %sum, %output[] : memref<f32>
  return
}

// ReductionX takes precedence over vector; the tail maximum is rewritten to one.
// The tail updates a slice of the shared product before one final reduction.
// CHECK-LABEL: func.func @product65_body_tail(
// CHECK: %[[PROD_INIT:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[PROD_ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[PROD_IDENTITY:.*]] = npuvector.broadcast %[[PROD_ONE]]{{.*}} : f32 to !npuvector<32xf32>
// CHECK: %[[PROD_SPLIT:.*]] = arith.constant 64 : index
// CHECK: %[[PROD_BODY:.*]] = scf.for %{{.*}} = %{{.*}} to %[[PROD_SPLIT]] step %{{.*}} iter_args(%{{.*}} = %[[PROD_IDENTITY]]) -> (!npuvector<32xf32>) {
// CHECK: npuvector.transfer_read {{.*}} : memref<65xf32>, !npuvector<32xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[PROD_TAIL_MAX:.*]] = arith.constant 1 : index
// CHECK: %[[PROD_TAIL:.*]] = scf.for %{{.*}} = %[[PROD_SPLIT]] to %{{.*}} step %[[PROD_TAIL_MAX]] iter_args(%[[PROD_TACC:.*]] = %[[PROD_BODY]]) -> (!npuvector<32xf32>) {
// CHECK: %[[PROD_SLICE:.*]] = npuvector.extract_slice %[[PROD_TACC]][%[[PROD_OFFSET:.*]]] [%[[PROD_TAIL_MAX]]] [%[[PROD_STRIDE:.*]]] {keep_dims = array<i64: 0>} : !npuvector<32xf32> to !npuvector<1xf32>
// CHECK: %[[PROD_VALUE:.*]] = npuvector.transfer_read {{.*}} [%[[PROD_TAIL_MAX]]] [%[[PROD_TAIL_MAX]]], %{{.*}} : memref<65xf32>, !npuvector<1xf32>
// CHECK: %[[PROD_NEXT:.*]] = arith.mulf %[[PROD_SLICE]], %[[PROD_VALUE]] {{.*}} : !npuvector<1xf32>
// CHECK: %[[PROD_INSERT:.*]] = npuvector.insert_slice %[[PROD_NEXT]] into %[[PROD_TACC]][%[[PROD_OFFSET]]] [%[[PROD_TAIL_MAX]]] [%[[PROD_STRIDE]]] : !npuvector<1xf32> into !npuvector<32xf32>
// CHECK-NEXT: scf.yield %[[PROD_INSERT]] : !npuvector<32xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[PROD_REDUCED:.*]] = npuvector.reduction <mul>, %[[PROD_TAIL]] {{.*}} : !npuvector<32xf32> into !npuvector.f32
// CHECK: %[[PROD_INIT_VECTOR:.*]] = npuvector.broadcast %[[PROD_INIT]] : f32 to !npuvector.f32
// CHECK: %[[PROD_RESULT:.*]] = arith.mulf %[[PROD_REDUCED]], %[[PROD_INIT_VECTOR]] : !npuvector.f32
// CHECK: npuvector.transfer_write %[[PROD_RESULT]], {{.*}} : !npuvector.f32, memref<f32>
// CHECK-NOT: npuvector.reduction
// CHECK: return
func.func @product65_body_tail(%input: memref<65xf32>, %output: memref<f32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c65 = arith.constant 65 : index
  %init = arith.constant 2.0 : f32
  %product = scf.for %i = %c0 to %c65 step %c1 iter_args(%acc = %init) -> f32 {
    %value = memref.load %input[%i] : memref<65xf32>
    %next = arith.mulf %acc, %value {reduction_type = "x"} : f32
    scf.yield %next : f32
  } {vector = 16 : i64, reduction_x = 32 : i64}
  memref.store %product, %output[] : memref<f32>
  return
}

// -----

// The outer body and tail keep scalar broadcasts for their inner reduction's
// neutral values instead of materializing full-tile constants.
// CHECK-LABEL: func.func @test_elementwise_body_tail_inner_reduction_init(
// CHECK: scf.for
// CHECK: %[[BT_BODY_ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[BT_BODY_INIT:.*]] = npuvector.broadcast %[[BT_BODY_ZERO]]{{.*}} : f32 to !npuvector<2x8xf32>
// CHECK: %[[BT_BODY:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[BT_BODY_INIT]]) -> (!npuvector<2x8xf32>) {
// CHECK: %[[BT_BODY_SUM:.*]] = npuvector.reduction <add>, %[[BT_BODY]] {reduction_dims = array<i64: 1>} : !npuvector<2x8xf32> into !npuvector<2xf32>
// CHECK: npuvector.transfer_write %[[BT_BODY_SUM]], {{.*}} : !npuvector<2xf32>, memref<3xf32>
// CHECK: %[[BT_TAIL_ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[BT_TAIL_INIT:.*]] = npuvector.broadcast %[[BT_TAIL_ZERO]]{{.*}} : f32 to !npuvector<1x8xf32>
// CHECK: %[[BT_TAIL:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %[[BT_TAIL_INIT]]) -> (!npuvector<1x8xf32>) {
// CHECK: %[[BT_TAIL_SUM:.*]] = npuvector.reduction <add>, %[[BT_TAIL]] {reduction_dims = array<i64: 1>} : !npuvector<1x8xf32> into !npuvector<1xf32>
// CHECK: npuvector.transfer_write %[[BT_TAIL_SUM]], {{.*}} : !npuvector<1xf32>, memref<3xf32>
// CHECK: return
// NO-SCRATCH-LABEL: func.func @test_elementwise_body_tail_inner_reduction_init(
// NO-SCRATCH-NOT: arith.constant dense
// NO-SCRATCH: return
func.func @test_elementwise_body_tail_inner_reduction_init(
    %input: memref<3x8xf32>, %output: memref<3xf32>)
    attributes {arch = "Ascend910B1", hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.000000e+00 : f32
  scf.for %row = %c0 to %c3 step %c1 {
    %sum = scf.for %column = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> f32 {
      %value = memref.load %input[%row, %column] : memref<3x8xf32>
      %square = arith.mulf %value, %value : f32
      %next = arith.addf %acc, %square {reduction_type = "x"} : f32
      scf.yield %next : f32
    } {reduction_x = 8 : i64}
    memref.store %sum, %output[%row] : memref<3xf32>
  } {vector = 2 : i64}
  return
}

// -----

// Both reduction axes have tails; each updates a slice of the shared full tile.
// CHECK-LABEL: func.func @test_nested_reduction_tail_neutral_uses_outer_tile(
// CHECK: npuvector.broadcast %{{.*}} : f32 to !npuvector<15x64xf32>
// CHECK: %[[NESTED_SPLIT:.*]] = arith.constant 30 : index
// CHECK: %[[NESTED_BODY:.*]] = scf.for %{{.*}} = %{{.*}} to %[[NESTED_SPLIT]] step %{{.*}} iter_args(%[[NESTED_BACC:.*]] = %{{.*}}) -> (!npuvector<15x64xf32>) {
// CHECK: %[[NESTED_INNER_BODY:.*]] = scf.for {{.*}} -> (!npuvector<15x64xf32>) {
// CHECK: npuvector.transfer_read {{.*}} : memref<32x65xf32>, !npuvector<15x64xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[NESTED_INNER_TAIL:.*]] = scf.for {{.*}} iter_args(%[[NESTED_INNER_ACC:.*]] = %[[NESTED_INNER_BODY]]) -> (!npuvector<15x64xf32>) {
// CHECK: %[[NESTED_INNER_SLICE:.*]] = npuvector.extract_slice %[[NESTED_INNER_ACC]]{{.*}} : !npuvector<15x64xf32> to !npuvector<15x1xf32>
// CHECK: %[[NESTED_INNER_VALUE:.*]] = npuvector.transfer_read {{.*}} : memref<32x65xf32>, !npuvector<15x1xf32>
// CHECK: %[[NESTED_INNER_NEXT:.*]] = arith.addf %[[NESTED_INNER_SLICE]], %[[NESTED_INNER_VALUE]] {{.*}} : !npuvector<15x1xf32>
// CHECK: %[[NESTED_INNER_INSERT:.*]] = npuvector.insert_slice %[[NESTED_INNER_NEXT]] into %[[NESTED_INNER_ACC]]{{.*}} : !npuvector<15x1xf32> into !npuvector<15x64xf32>
// CHECK-NEXT: scf.yield %[[NESTED_INNER_INSERT]] : !npuvector<15x64xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[NESTED_BNEXT:.*]] = arith.addf %[[NESTED_BACC]], %[[NESTED_INNER_TAIL]] {{.*}} : !npuvector<15x64xf32>
// CHECK-NEXT: scf.yield %[[NESTED_BNEXT]] : !npuvector<15x64xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[NESTED_TAIL:.*]] = scf.for %{{.*}} = %[[NESTED_SPLIT]] to %{{.*}} step %{{.*}} iter_args(%[[NESTED_TACC:.*]] = %[[NESTED_BODY]]) -> (!npuvector<15x64xf32>) {
// CHECK: %[[NESTED_OUTER_SLICE:.*]] = npuvector.extract_slice %[[NESTED_TACC]]{{.*}} : !npuvector<15x64xf32> to !npuvector<2x64xf32>
// CHECK: %[[NESTED_SMALL_BODY:.*]] = scf.for {{.*}} -> (!npuvector<2x64xf32>) {
// CHECK: npuvector.transfer_read {{.*}} : memref<32x65xf32>, !npuvector<2x64xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[NESTED_SMALL_TAIL:.*]] = scf.for {{.*}} iter_args(%[[NESTED_CORNER_ACC:.*]] = %[[NESTED_SMALL_BODY]]) -> (!npuvector<2x64xf32>) {
// CHECK: %[[NESTED_CORNER_SLICE:.*]] = npuvector.extract_slice %[[NESTED_CORNER_ACC]]{{.*}} : !npuvector<2x64xf32> to !npuvector<2x1xf32>
// CHECK: %[[NESTED_CORNER_VALUE:.*]] = npuvector.transfer_read {{.*}} : memref<32x65xf32>, !npuvector<2x1xf32>
// CHECK: %[[NESTED_CORNER_NEXT:.*]] = arith.addf %[[NESTED_CORNER_SLICE]], %[[NESTED_CORNER_VALUE]] {{.*}} : !npuvector<2x1xf32>
// CHECK: %[[NESTED_CORNER_INSERT:.*]] = npuvector.insert_slice %[[NESTED_CORNER_NEXT]] into %[[NESTED_CORNER_ACC]]{{.*}} : !npuvector<2x1xf32> into !npuvector<2x64xf32>
// CHECK-NEXT: scf.yield %[[NESTED_CORNER_INSERT]] : !npuvector<2x64xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[NESTED_TNEXT:.*]] = arith.addf %[[NESTED_OUTER_SLICE]], %[[NESTED_SMALL_TAIL]] {{.*}} : !npuvector<2x64xf32>
// CHECK: %[[NESTED_OUTER_INSERT:.*]] = npuvector.insert_slice %[[NESTED_TNEXT]] into %[[NESTED_TACC]]{{.*}} : !npuvector<2x64xf32> into !npuvector<15x64xf32>
// CHECK-NEXT: scf.yield %[[NESTED_OUTER_INSERT]] : !npuvector<15x64xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[NESTED_RESULT:.*]] = npuvector.reduction <add>, %[[NESTED_TAIL]] {reduction_dims = array<i64: 0, 1>} : !npuvector<15x64xf32> into !npuvector.f32
// CHECK: npuvector.transfer_write %[[NESTED_RESULT]], {{.*}} : !npuvector.f32, memref<f32>
// CHECK-NOT: npuvector.reduction
// CHECK: return
func.func @test_nested_reduction_tail_neutral_uses_outer_tile(
    %input: memref<32x65xf32>,
    %output: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c65 = arith.constant 65 : index
  %zero = arith.constant 0.000000e+00 : f32

  %sum_i = scf.for %i = %c0 to %c32 step %c1 iter_args(%acc_i = %zero) -> f32 {
    %sum_j = scf.for %j = %c0 to %c65 step %c1 iter_args(%acc_j = %zero) -> f32 {
      %value = memref.load %input[%i, %j] : memref<32x65xf32>
      %next_j = arith.addf %acc_j, %value {reduction_type = "all"} : f32
      scf.yield %next_j : f32
    } {reduction_all=64}
    %next_i = arith.addf %acc_i, %sum_j {reduction_type = "all"} : f32
    scf.yield %next_i : f32
  } {reduction_all=15}

  memref.store %sum_i, %output[] : memref<f32>
  return
}

// -----

// CHECK-LABEL: func.func @test_nested_reduction_all_x_merge_to_single_multidim_reduce
// CHECK: scf.for {{.*}} -> (!npuvector<4x8x16xf32>) {
// CHECK: scf.for {{.*}} -> (!npuvector<4x8x16xf32>) {
// CHECK: scf.for {{.*}} -> (!npuvector<4x8x16xf32>) {
// CHECK: npuvector.transfer_read {{.*}} : memref<4x8x16xf32>, !npuvector<4x8x16xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %{{.*}} {reduction_dims = array<i64: 0, 1, 2>} : !npuvector<4x8x16xf32> into !npuvector.f32
// CHECK-NOT: npuvector.reduction
// CHECK: npuvector.transfer_write %[[SUM]], %{{.*}}[]{{.*}}: !npuvector.f32, memref<f32>
func.func @test_nested_reduction_all_x_merge_to_single_multidim_reduce(
    %input: memref<4x8x16xf32>,
    %output: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.000000e+00 : f32

  %sum_i = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc_i = %zero) -> f32 {
    %sum_j = scf.for %j = %c0 to %c8 step %c1 iter_args(%acc_j = %zero) -> f32 {
      %sum_k = scf.for %k = %c0 to %c16 step %c1 iter_args(%acc_k = %zero) -> f32 {
        %value = memref.load %input[%i, %j, %k] : memref<4x8x16xf32>
        %next_k = arith.addf %acc_k, %value {reduction_type = "all"} : f32
        scf.yield %next_k : f32
      } {reduction_all=16}
      %next_j = arith.addf %acc_j, %sum_k {reduction_axes = [1 : index], reduction_type = "x"} : f32
      scf.yield %next_j : f32
    } {reduction_x = 8 : i64}
    %next_i = arith.addf %acc_i, %sum_j {reduction_type = "all"} : f32
    scf.yield %next_i : f32
  } {reduction_all=4}

  memref.store %sum_i, %output[] : memref<f32>
  return
}

// -----

// CHECK-LABEL: func.func @test_nested_reduction_no_merge_when_child_result_is_not_direct_operand
// CHECK-NOT: {reduction_dims = array<i64: 0, 1, 2>}
// CHECK: npuvector.reduction <add>
// CHECK-NOT: {reduction_dims = array<i64: 0, 1, 2>}
// CHECK: return
func.func @test_nested_reduction_no_merge_when_child_result_is_not_direct_operand(
    %input: memref<4x8x16xf32>,
    %output: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.000000e+00 : f32
  %scale = arith.constant 2.000000e+00 : f32

  %sum_i = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc_i = %zero) -> f32 {
    %sum_j = scf.for %j = %c0 to %c8 step %c1 iter_args(%acc_j = %zero) -> f32 {
      %sum_k = scf.for %k = %c0 to %c16 step %c1 iter_args(%acc_k = %zero) -> f32 {
        %value = memref.load %input[%i, %j, %k] : memref<4x8x16xf32>
        %next_k = arith.addf %acc_k, %value {reduction_type = "all"} : f32
        scf.yield %next_k : f32
      } {reduction_all=16}
      %scaled = arith.mulf %sum_k, %scale : f32
      %next_j = arith.addf %acc_j, %scaled {reduction_axes = [1 : index], reduction_type = "x"} : f32
      scf.yield %next_j : f32
    } {reduction_x = 8 : i64}
    %next_i = arith.addf %acc_i, %sum_j {reduction_type = "all"} : f32
    scf.yield %next_i : f32
  } {reduction_all=4}

  memref.store %sum_i, %output[] : memref<f32>
  return
}

// -----

// ============================================================================
// 4. ReductionY / Broadcast (incl. unit attrs without integer), static and dynamic
// ============================================================================

// ReductionY without iter_args (load-modify-store), static
// acc[j] += sum_i input[i,j]
// Inner loop uses unit attr `reduction_y` (no integer; defaults in pass via kVectorSize when needed).

// CHECK-LABEL: func.func @test_reduction_y_no_iter_args
func.func @test_reduction_y_no_iter_args(
    %acc: memref<64xf32>,
    %input: memref<32x64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index

  // CHECK: npuvector.transfer_read
  // CHECK: npuvector.transfer_read
  // CHECK: arith.addf {{.*}} {reduction_axes = [0 : index], reduction_type = "y"}
  // CHECK: npuvector.transfer_write
  scf.for %j = %c0 to %c64 step %c1 {
    scf.for %i = %c0 to %c32 step %c1 {
      %acc_val = memref.load %acc[%j] : memref<64xf32>
      %x = memref.load %input[%i, %j] : memref<32x64xf32>
      %new = arith.addf %acc_val, %x {reduction_axes = [0 : index], reduction_type = "y"} : f32
      memref.store %new, %acc[%j] : memref<64xf32>
    } {reduction_y}
  } {vector=64}
  return
}

// -----

// Broadcast inner: unit attr `broadcast` (no integer). getVectorizationMode falls back to
// kVectorSize when the attr is not IntegerAttr; nested `broadcast`/`reduction_y` reuse the
// outer `vector` loop’s step / max-step in handleNestedForOp.

// CHECK-LABEL: func.func @test_broadcast_inner_unit_attr
func.func @test_broadcast_inner_unit_attr(
    %out: memref<32x64xf32>,
    %input: memref<32x64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index

  // CHECK: npuvector.transfer_read
  // CHECK: npuvector.transfer_write
  scf.for %j = %c0 to %c64 step %c1 {
    scf.for %i = %c0 to %c32 step %c1 {
      %x = memref.load %input[%i, %j] : memref<32x64xf32>
      memref.store %x, %out[%i, %j] : memref<32x64xf32>
    } {broadcast}
  } {vector=64}
  return
}

// -----

// CHECK-LABEL: func.func @test_reduction_y_no_iter_args_dynamic
func.func @test_reduction_y_no_iter_args_dynamic(
    %acc: memref<?xf32>,
    %input: memref<?x?xf32>,
    %M: index, %N: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: npuvector.transfer_read
  // CHECK: arith.addf {{.*}} {reduction_axes = [0 : index], reduction_type = "y"}
  // CHECK: npuvector.transfer_write
  scf.for %j = %c0 to %N step %c1 {
    scf.for %i = %c0 to %M step %c1 {
      %acc_val = memref.load %acc[%j] : memref<?xf32>
      %x = memref.load %input[%i, %j] : memref<?x?xf32>
      %new = arith.addf %acc_val, %x {reduction_axes = [0 : index], reduction_type = "y"} : f32
      memref.store %new, %acc[%j] : memref<?xf32>
    } {reduction_y}
  } {vector=4096}
  return
}

// -----

// ReductionY with iter_args, static
// output[j] = sum_i input[i,j]

// CHECK-LABEL: func.func @test_reduction_y_with_iter_args
func.func @test_reduction_y_with_iter_args(
    %input: memref<32x64xf32>,
    %output: memref<64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32

  // CHECK: npuvector.broadcast %{{.*}} : f32 to !npuvector<64xf32>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!npuvector<64xf32>) {
  // CHECK:   npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}] [%{{.*}}], %{{.*}} : memref<32x64xf32>, !npuvector<64xf32>
  // CHECK:   arith.addf %{{.*}}, %{{.*}} {reduction_axes = [0 : index], reduction_type = "y"} : !npuvector<64xf32>
  // CHECK:   scf.yield %{{.*}} : !npuvector<64xf32>
  // CHECK: }
  // CHECK: npuvector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}]{{.*}}: !npuvector<64xf32>, memref<64xf32>
  scf.for %j = %c0 to %c64 step %c1 {
    %sum = scf.for %i = %c0 to %c32 step %c1 iter_args(%acc = %init) -> (f32) {
      %x = memref.load %input[%i, %j] : memref<32x64xf32>
      %new = arith.addf %acc, %x {reduction_axes = [0 : index], reduction_type = "y"} : f32
      scf.yield %new : f32
    } {reduction_y}
    memref.store %sum, %output[%j] : memref<64xf32>
  } {vector=64}
  return
}

// -----

// CHECK-LABEL: func.func @test_reduction_y_with_iter_args_dynamic
func.func @test_reduction_y_with_iter_args_dynamic(
    %input: memref<?x?xf32>,
    %output: memref<?xf32>,
    %M: index, %N: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32

  // CHECK: arith.constant 4096 : index
  // CHECK: affine.min
  // CHECK: npuvector.broadcast %{{.*}}[%{{.*}}] [%{{.*}}] : f32 to !npuvector<?xf32>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!npuvector<?xf32>) {
  // CHECK:   npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}] [%{{.*}}] [%{{.*}}], %{{.*}} : memref<?x?xf32>, !npuvector<?xf32>
  // CHECK:   arith.addf %{{.*}}, %{{.*}} {reduction_axes = [0 : index], reduction_type = "y"} : !npuvector<?xf32>
  // CHECK:   scf.yield %{{.*}} : !npuvector<?xf32>
  // CHECK: }
  // CHECK: npuvector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}]{{.*}}: !npuvector<?xf32>, memref<?xf32>
  scf.for %j = %c0 to %N step %c1 {
    %sum = scf.for %i = %c0 to %M step %c1 iter_args(%acc = %init) -> (f32) {
      %x = memref.load %input[%i, %j] : memref<?x?xf32>
      %new = arith.addf %acc, %x {reduction_axes = [0 : index], reduction_type = "y"} : f32
      scf.yield %new : f32
    } {reduction_y}
    memref.store %sum, %output[%j] : memref<?xf32>
  } {vector=4096}
  return
}

// -----

// ============================================================================
// 5. Multi-dim elementwise static and dynamic
// ============================================================================

// CHECK-LABEL: func.func @test_multidim_elementwise_static
func.func @test_multidim_elementwise_static(
    %a: memref<32x64xf32>, %b: memref<32x64xf32>, %c: memref<32x64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[A_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<32x64xf32>, !npuvector<32x64xf32>
  // CHECK: %[[B_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<32x64xf32>, !npuvector<32x64xf32>
  // CHECK: %[[C_VEC:.*]] = arith.addf %[[A_VEC]], %[[B_VEC]] : !npuvector<32x64xf32>
  // CHECK: npuvector.transfer_write %[[C_VEC]], %{{.*}}[%{{.*}}, %{{.*}}]{{.*}}: !npuvector<32x64xf32>, memref<32x64xf32>
  // CHECK-NOT: scf.for
  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c64 step %c1 {
      %va = memref.load %a[%i, %j] : memref<32x64xf32>
      %vb = memref.load %b[%i, %j] : memref<32x64xf32>
      %vc = arith.addf %va, %vb : f32
      memref.store %vc, %c[%i, %j] : memref<32x64xf32>
    } {vector=64}
  } {vector=32}
  return
}

// -----

// CHECK-LABEL: func.func @test_multidim_elementwise_transpose_load
func.func @test_multidim_elementwise_transpose_load(
    %a: memref<32x64xf32>, %b: memref<32x64xf32>, %c: memref<32x64xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[A_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<32x64xf32>, !npuvector<32x64xf32>
  // CHECK: %[[B_READ:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<32x64xf32>, !npuvector<64x32xf32>
  // CHECK: %[[B_VEC:.*]] = npuvector.transpose %[[B_READ]], [1, 0]
  // CHECK: %[[C_VEC:.*]] = arith.addf %[[A_VEC]], %[[B_VEC]] : !npuvector<32x64xf32>
  // CHECK: npuvector.transfer_write %[[C_VEC]], %{{.*}}[%{{.*}}, %{{.*}}]{{.*}}: !npuvector<32x64xf32>, memref<32x64xf32>
  // CHECK-NOT: scf.for
  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c64 step %c1 {
      %va = memref.load %a[%i, %j] : memref<32x64xf32>
      %vb = memref.load %b[%j, %i] : memref<32x64xf32>
      %vc = arith.addf %va, %vb : f32
      memref.store %vc, %c[%i, %j] : memref<32x64xf32>
    } {vector=64}
  } {vector=32}
  return
}

// -----

// CHECK-LABEL: func.func @test_multidim_elementwise_dynamic
func.func @test_multidim_elementwise_dynamic(
    %a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>,
    %M: index, %N: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: arith.constant 4096
  // CHECK: affine.min
  // CHECK: arith.constant 4096
  // CHECK: affine.min
  // CHECK: npuvector.transfer_read
  // CHECK: npuvector.transfer_read
  // CHECK: arith.addf
  // CHECK: npuvector.transfer_write
  // CHECK-NOT: scf.for
  scf.for %i = %c0 to %M step %c1 {
    scf.for %j = %c0 to %N step %c1 {
      %va = memref.load %a[%i, %j] : memref<?x?xf32>
      %vb = memref.load %b[%i, %j] : memref<?x?xf32>
      %vc = arith.addf %va, %vb : f32
      memref.store %vc, %c[%i, %j] : memref<?x?xf32>
    } {vector=4096}
  } {vector=4096}
  return
}

// -----

// CHECK-LABEL: func.func @test_multidim_elementwise_skip_middle_axis_static
func.func @test_multidim_elementwise_skip_middle_axis_static(
    %a: memref<4x8x16xf32>, %b: memref<4x8x16xf32>, %c: memref<4x8x16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // CHECK: scf.for %[[J:.*]] =
  // CHECK: %[[A_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %[[J]], %{{.*}}]{{.*}}, %{{.*}} {permutation_map = #[[SKIP_MIDDLE_MAP]]} : memref<4x8x16xf32>, !npuvector<4x16xf32>
  // CHECK: %[[B_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %[[J]], %{{.*}}]{{.*}}, %{{.*}} {permutation_map = #[[SKIP_MIDDLE_MAP]]} : memref<4x8x16xf32>, !npuvector<4x16xf32>
  // CHECK: %[[C_VEC:.*]] = arith.addf %[[A_VEC]], %[[B_VEC]] : !npuvector<4x16xf32>
  // CHECK: npuvector.transfer_write %[[C_VEC]], %{{.*}}[%{{.*}}, %[[J]], %{{.*}}] {permutation_map = #[[SKIP_MIDDLE_MAP]]} : !npuvector<4x16xf32>, memref<4x8x16xf32>
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      scf.for %k = %c0 to %c16 step %c1 {
        %va = memref.load %a[%i, %j, %k] : memref<4x8x16xf32>
        %vb = memref.load %b[%i, %j, %k] : memref<4x8x16xf32>
        %vc = arith.addf %va, %vb : f32
        memref.store %vc, %c[%i, %j, %k] : memref<4x8x16xf32>
      } {vector=16}
    }
  } {vector=4}
  return
}

// -----

// CHECK-LABEL: func.func @test_multidim_elementwise_skip_middle_axis_dynamic
func.func @test_multidim_elementwise_skip_middle_axis_dynamic(
    %a: memref<?x?x?xf32>, %b: memref<?x?x?xf32>, %c: memref<?x?x?xf32>,
    %M: index, %N: index, %K: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: scf.for %[[J:.*]] =
  // CHECK: npuvector.transfer_read %{{.*}}[%{{.*}}, %[[J]], %{{.*}}]{{.*}}, %{{.*}} {permutation_map = #[[SKIP_MIDDLE_MAP]]} : memref<?x?x?xf32>, !npuvector<?x?xf32>
  // CHECK: npuvector.transfer_read %{{.*}}[%{{.*}}, %[[J]], %{{.*}}]{{.*}}, %{{.*}} {permutation_map = #[[SKIP_MIDDLE_MAP]]} : memref<?x?x?xf32>, !npuvector<?x?xf32>
  // CHECK: arith.addf
  // CHECK: npuvector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}, %[[J]], %{{.*}}] {permutation_map = #[[SKIP_MIDDLE_MAP]]} : !npuvector<?x?xf32>, memref<?x?x?xf32>
  scf.for %i = %c0 to %M step %c1 {
    scf.for %j = %c0 to %N step %c1 {
      scf.for %k = %c0 to %K step %c1 {
        %va = memref.load %a[%i, %j, %k] : memref<?x?x?xf32>
        %vb = memref.load %b[%i, %j, %k] : memref<?x?x?xf32>
        %vc = arith.addf %va, %vb : f32
        memref.store %vc, %c[%i, %j, %k] : memref<?x?x?xf32>
      } {vector=4096}
    }
  } {vector=4096}
  return
}

// -----

// CHECK-LABEL: func.func @test_multidim_arith_disjoint_operand_axes
func.func @test_multidim_arith_disjoint_operand_axes(
    %input: memref<512xi64>, %out: memref<512x512xi1>, %M: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c512 = arith.constant 512 : index

  // CHECK: %[[OUTER:.*]] = npuvector.transfer_read {{.*}} : memref<512xi64>, !npuvector<?xi64>
  // CHECK: %[[INNER:.*]] = npuvector.transfer_read {{.*}} : memref<512xi64>, !npuvector<512xi64>
  // CHECK: %[[INNER_BCAST:.*]] = npuvector.broadcast %[[INNER]]{{.*}} : !npuvector<512xi64> to !npuvector<?x512xi64> {dimension = array<i64: 1>}
  // CHECK: %[[OUTER_BCAST:.*]] = npuvector.broadcast %[[OUTER]]{{.*}} : !npuvector<?xi64> to !npuvector<?x512xi64> {dimension = array<i64: 0>}
  // CHECK: %[[CMP:.*]] = npuvector.cmpi sle, %[[INNER_BCAST]], %[[OUTER_BCAST]] : !npuvector<?x512xi64> to !npuvector<?x512xi1>
  // CHECK: npuvector.transfer_write %[[CMP]], {{.*}} : !npuvector<?x512xi1>, memref<512x512xi1>
  scf.for %i = %c0 to %M step %c1 {
    %outer = memref.load %input[%i] : memref<512xi64>
    scf.for %j = %c0 to %c512 step %c1 {
      %inner = memref.load %input[%j] : memref<512xi64>
      %cmp = arith.cmpi sle, %inner, %outer : i64
      memref.store %cmp, %out[%i, %j] : memref<512x512xi1>
    } {vector=512}
  } {vector=8}
  return
}

// -----

// CHECK-LABEL: func.func @test_multidim_arith_partially_overlapping_operand_axes
func.func @test_multidim_arith_partially_overlapping_operand_axes(
    %lhs: memref<4x8xf32>, %rhs: memref<8x16xf32>, %out: memref<4x8x16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index

  // CHECK: %[[LHS:.*]] = npuvector.transfer_read {{.*}} : memref<4x8xf32>, !npuvector<4x8xf32>
  // CHECK: %[[RHS:.*]] = npuvector.transfer_read {{.*}} : memref<8x16xf32>, !npuvector<8x16xf32>
  // CHECK: %[[LHS_BCAST:.*]] = npuvector.broadcast %[[LHS]]{{.*}} : !npuvector<4x8xf32> to !npuvector<4x8x16xf32> {dimension = array<i64: 0, 1>}
  // CHECK: %[[RHS_BCAST:.*]] = npuvector.broadcast %[[RHS]]{{.*}} : !npuvector<8x16xf32> to !npuvector<4x8x16xf32> {dimension = array<i64: 1, 2>}
  // CHECK: %[[ADD:.*]] = arith.addf %[[LHS_BCAST]], %[[RHS_BCAST]] : !npuvector<4x8x16xf32>
  // CHECK: npuvector.transfer_write %[[ADD]], {{.*}} : !npuvector<4x8x16xf32>, memref<4x8x16xf32>
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      %a = memref.load %lhs[%i, %j] : memref<4x8xf32>
      scf.for %k = %c0 to %c16 step %c1 {
        %b = memref.load %rhs[%j, %k] : memref<8x16xf32>
        %sum = arith.addf %a, %b : f32
        memref.store %sum, %out[%i, %j, %k] : memref<4x8x16xf32>
      } {vector=16}
    } {vector=8}
  } {vector=4}
  return
}

// -----

// ============================================================================
// 6. Multi-dim with Transpose static and dynamic
// ============================================================================

// Store-only transpose: load a[i,j], b[i,j], store out[j,i]

// CHECK-LABEL: func.func @test_multidim_transpose_store_only
func.func @test_multidim_transpose_store_only(
    %a: memref<32x64xf32>, %b: memref<32x64xf32>, %out: memref<64x32xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[A_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<32x64xf32>, !npuvector<32x64xf32>
  // CHECK: %[[B_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<32x64xf32>, !npuvector<32x64xf32>
  // CHECK: %[[SUM_VEC:.*]] = arith.addf %[[A_VEC]], %[[B_VEC]] : !npuvector<32x64xf32>
  // CHECK: %[[TRANS:.*]] = npuvector.transpose %[[SUM_VEC]], [1, 0] : !npuvector<32x64xf32> to !npuvector<64x32xf32>
  // CHECK: npuvector.transfer_write %[[TRANS]], %{{.*}}[%{{.*}}, %{{.*}}] : !npuvector<64x32xf32>, memref<64x32xf32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c64 step %c1 {
      %va = memref.load %a[%i, %j] : memref<32x64xf32>
      %vb = memref.load %b[%i, %j] : memref<32x64xf32>
      %sum = arith.addf %va, %vb : f32
      memref.store %sum, %out[%j, %i] : memref<64x32xf32>
    } {vector=64}
  } {vector=32}
  return
}

// -----

// Canonical + transpose: c[i,j]=a[i,j]+b[i,j], d[j,i]=a1[j,i]+b1[j,i]

// CHECK-LABEL: func.func @test_multidim_transpose_static
func.func @test_multidim_transpose_static(
    %a: memref<32x64xf32>, %b: memref<32x64xf32>, %c: memref<32x64xf32>,
    %a1: memref<64x32xf32>, %b1: memref<64x32xf32>, %d: memref<64x32xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[A_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<32x64xf32>, !npuvector<32x64xf32>
  // CHECK: %[[B_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<32x64xf32>, !npuvector<32x64xf32>
  // CHECK: %[[C_VEC:.*]] = arith.addf %[[A_VEC]], %[[B_VEC]] : !npuvector<32x64xf32>
  // CHECK: npuvector.transfer_write %[[C_VEC]], %{{.*}}[%{{.*}}, %{{.*}}]{{.*}}: !npuvector<32x64xf32>, memref<32x64xf32>
  // CHECK: %[[A1_READ:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<64x32xf32>, !npuvector<64x32xf32>
  // CHECK: %[[A1_VEC:.*]] = npuvector.transpose %[[A1_READ]], [1, 0]
  // CHECK: %[[B1_READ:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}} : memref<64x32xf32>, !npuvector<64x32xf32>
  // CHECK: %[[B1_VEC:.*]] = npuvector.transpose %[[B1_READ]], [1, 0]
  // CHECK: %[[D_VEC:.*]] = arith.addf %[[A1_VEC]], %[[B1_VEC]] : !npuvector<32x64xf32>
  // CHECK: %[[D_TRANS:.*]] = npuvector.transpose %[[D_VEC]], [1, 0]
  // CHECK: npuvector.transfer_write %[[D_TRANS]], %{{.*}}[%{{.*}}, %{{.*}}] : !npuvector<64x32xf32>, memref<64x32xf32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c64 step %c1 {
      %va = memref.load %a[%i, %j] : memref<32x64xf32>
      %vb = memref.load %b[%i, %j] : memref<32x64xf32>
      %vc = arith.addf %va, %vb : f32
      memref.store %vc, %c[%i, %j] : memref<32x64xf32>

      %va1 = memref.load %a1[%j, %i] : memref<64x32xf32>
      %vb1 = memref.load %b1[%j, %i] : memref<64x32xf32>
      %vd = arith.addf %va1, %vb1 : f32
      memref.store %vd, %d[%j, %i] : memref<64x32xf32>
    } {vector=64}
  } {vector=32}
  return
}

// -----

// Three-layer loop: store transpose [k,j,i]

// CHECK-LABEL: func.func @test_three_layer_outer_inner_tag
func.func @test_three_layer_outer_inner_tag(
    %a: memref<8x16x32xf32>, %out: memref<8x16x32xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}], %{{.*}} : memref<8x16x32xf32>, !npuvector<8x16x32xf32>
  // CHECK: %[[TRANS:.*]] = npuvector.transpose %[[VEC]], [2, 1, 0] : !npuvector<8x16x32xf32> to !npuvector<32x16x8xf32>
  // CHECK: npuvector.transfer_write %[[TRANS]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}]{{.*}}: !npuvector<32x16x8xf32>, memref<8x16x32xf32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c8 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      scf.for %k = %c0 to %c32 step %c1 {
        %v = memref.load %a[%i, %j, %k] : memref<8x16x32xf32>
        memref.store %v, %out[%k, %j, %i] : memref<8x16x32xf32>
      } {vector=32}
    } {vector=16}
  } {vector=8}
  return
}

// -----

// Transpose with scf.if (no else)

// CHECK-LABEL: func.func @test_multidim_transpose_with_scf_if
func.func @test_multidim_transpose_with_scf_if(
    %a: memref<32x64xf32>, %out: memref<64x32xf32>, %threshold: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32

  // CHECK: npuvector.transfer_read
  // CHECK: scf.if
  // CHECK: npuvector.transpose {{.*}} [1, 0]
  // CHECK: npuvector.transfer_write
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c64 step %c1 {
      %v = memref.load %a[%i, %j] : memref<32x64xf32>
      %cond = arith.cmpi ult, %i, %threshold {skip_vectorize} : index
      scf.if %cond {
        %scaled = arith.mulf %v, %c2 : f32
        memref.store %scaled, %out[%j, %i] : memref<64x32xf32>
      }
    } {vector=64}
  } {vector=32}
  return
}

// -----

// CHECK-LABEL: func.func @test_multidim_if_result_transpose_store
func.func @test_multidim_if_result_transpose_store(
    %a: memref<4x5x6xf32>, %b: memref<4x5x6xf32>, %c: memref<4x5x6xf32>,
    %out: memref<5x3x4x6xf32>, %cond0: i1, %cond1: i1) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index

  // CHECK: %[[OUTER_IF:.*]] = scf.if {{.*}} -> (!npuvector<4x5x6xf32>) {
  // CHECK: npuvector.transfer_read
  // CHECK: scf.yield
  // CHECK: } else {
  // CHECK: %[[INNER_IF:.*]] = scf.if {{.*}} -> (!npuvector<4x5x6xf32>) {
  // CHECK: npuvector.transfer_read
  // CHECK: scf.yield
  // CHECK: } else {
  // CHECK: npuvector.transfer_read
  // CHECK: scf.yield
  // CHECK: }
  // CHECK: scf.yield %[[INNER_IF]]
  // CHECK: }
  // CHECK: %[[TRANS:.*]] = npuvector.transpose %[[OUTER_IF]], [1, 0, 2] : !npuvector<4x5x6xf32> to !npuvector<5x4x6xf32>
  // CHECK: npuvector.transfer_write %[[TRANS]]
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c5 step %c1 {
      scf.for %k = %c0 to %c6 step %c1 {
        %v = scf.if %cond0 -> f32 {
          %va = memref.load %a[%i, %j, %k] : memref<4x5x6xf32>
          scf.yield %va : f32
        } else {
          %inner = scf.if %cond1 -> f32 {
            %vb = memref.load %b[%i, %j, %k] : memref<4x5x6xf32>
            scf.yield %vb : f32
          } else {
            %vc = memref.load %c[%i, %j, %k] : memref<4x5x6xf32>
            scf.yield %vc : f32
          }
          scf.yield %inner : f32
        }
        memref.store %v, %out[%j, %c0, %i, %k] : memref<5x3x4x6xf32>
      } {vector=6}
    } {vector=5}
  } {vector=4}
  return
}

// -----

// CHECK-LABEL: func.func @test_multidim_transpose_dynamic
func.func @test_multidim_transpose_dynamic(
    %a: memref<?x?xf32>, %out: memref<?x?xf32>, %M: index, %N: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: arith.constant 4096
  // CHECK: affine.min
  // CHECK: arith.constant 4096
  // CHECK: affine.min
  // CHECK: npuvector.transfer_read
  // CHECK: npuvector.transpose
  // CHECK: npuvector.transfer_write
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %M step %c1 {
    scf.for %j = %c0 to %N step %c1 {
      %v = memref.load %a[%i, %j] : memref<?x?xf32>
      memref.store %v, %out[%j, %i] : memref<?x?xf32>
    } {vector=4096}
  } {vector=4096}
  return
}

// -----

// CHECK-LABEL: func.func @test_multidim_transpose_dynamic_with_scf_if
func.func @test_multidim_transpose_dynamic_with_scf_if(
    %a: memref<?x?xf32>, %out: memref<?x?xf32>, %M: index, %N: index, %thresh: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32

  // CHECK: arith.constant 4096
  // CHECK: affine.min
  // CHECK: arith.constant 4096
  // CHECK: affine.min
  // CHECK: npuvector.transfer_read
  // CHECK: arith.cmpi
  // CHECK: scf.if
  // CHECK: npuvector.broadcast
  // CHECK: arith.mulf
  // CHECK: npuvector.transpose
  // CHECK: npuvector.transfer_write
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %M step %c1 {
    scf.for %j = %c0 to %N step %c1 {
      %v = memref.load %a[%i, %j] : memref<?x?xf32>
      %cond = arith.cmpi ult, %i, %thresh {skip_vectorize} : index
      scf.if %cond {
        %scaled = arith.mulf %v, %c2 : f32
        memref.store %scaled, %out[%j, %i] : memref<?x?xf32>
      }
    } {vector=4096}
  } {vector=4096}
  return
}

// -----

// ============================================================================
// 6.1 Transpose with affine bounds
// ============================================================================

// 2D transpose: outer tile loops 0..1, inner affine.min bounds, load a[i,j] store out[j,i]

#map_2d_mul = affine_map<(d0)[s0] -> (d0 * s0)>
#map_2d_min = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @test_transpose_affine_bounds_2d
func.func @test_transpose_affine_bounds_2d(
    %a: memref<32x64xf32>, %out: memref<64x32xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index

  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: npuvector.transfer_read
  // CHECK: npuvector.transpose {{.*}} [1, 0]
  // CHECK: npuvector.transfer_write

  scf.for %bi = %c0 to %c1 step %c1 {
    scf.for %bj = %c0 to %c1 step %c1 {
      %lb_i = affine.apply #map_2d_mul(%bi)[%c32]
      %ub_i_raw = arith.addi %bi, %c1 : index
      %ub_i = affine.apply #map_2d_mul(%ub_i_raw)[%c32]
      %lb_i_clamped = affine.min #map_2d_min(%lb_i, %c32)
      %ub_i_clamped = affine.min #map_2d_min(%ub_i, %c32)
      scf.for %i = %lb_i_clamped to %ub_i_clamped step %c1 {
        %lb_j = affine.apply #map_2d_mul(%bj)[%c64]
        %ub_j_raw = arith.addi %bj, %c1 : index
        %ub_j = affine.apply #map_2d_mul(%ub_j_raw)[%c64]
        %lb_j_clamped = affine.min #map_2d_min(%lb_j, %c64)
        %ub_j_clamped = affine.min #map_2d_min(%ub_j, %c64)
        scf.for %j = %lb_j_clamped to %ub_j_clamped step %c1 {
          %v = memref.load %a[%i, %j] : memref<32x64xf32>
          memref.store %v, %out[%j, %i] : memref<64x32xf32>
        } {vector=64}
      } {vector=32}
    }
  }
  return
}

// -----

// 3D transpose: outer tile loops, inner affine bounds, load a[i,j,k] store out[k,j,i]

#map_3d_mul = affine_map<(d0)[s0] -> (d0 * s0)>
#map_3d_min = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @test_transpose_affine_bounds_3d
func.func @test_transpose_affine_bounds_3d(
    %a: memref<8x16x32xf32>, %out: memref<32x16x8xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index

  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: npuvector.transfer_read
  // CHECK: npuvector.transpose {{.*}} [2, 1, 0]
  // CHECK: npuvector.transfer_write

  scf.for %bi = %c0 to %c1 step %c1 {
    scf.for %bj = %c0 to %c1 step %c1 {
      scf.for %bk = %c0 to %c1 step %c1 {
        %lb_i = affine.apply #map_3d_mul(%bi)[%c8]
        %ub_i_raw = arith.addi %bi, %c1 : index
        %ub_i = affine.apply #map_3d_mul(%ub_i_raw)[%c8]
        %lb_i_c = affine.min #map_3d_min(%lb_i, %c8)
        %ub_i_c = affine.min #map_3d_min(%ub_i, %c8)
        scf.for %i = %lb_i_c to %ub_i_c step %c1 {
          %lb_j = affine.apply #map_3d_mul(%bj)[%c16]
          %ub_j_raw = arith.addi %bj, %c1 : index
          %ub_j = affine.apply #map_3d_mul(%ub_j_raw)[%c16]
          %lb_j_c = affine.min #map_3d_min(%lb_j, %c16)
          %ub_j_c = affine.min #map_3d_min(%ub_j, %c16)
          scf.for %j = %lb_j_c to %ub_j_c step %c1 {
            %lb_k = affine.apply #map_3d_mul(%bk)[%c32]
            %ub_k_raw = arith.addi %bk, %c1 : index
            %ub_k = affine.apply #map_3d_mul(%ub_k_raw)[%c32]
            %lb_k_c = affine.min #map_3d_min(%lb_k, %c32)
            %ub_k_c = affine.min #map_3d_min(%ub_k, %c32)
            scf.for %k = %lb_k_c to %ub_k_c step %c1 {
              %v = memref.load %a[%i, %j, %k] : memref<8x16x32xf32>
              memref.store %v, %out[%k, %j, %i] : memref<32x16x8xf32>
            } {vector=32}
          } {vector=16}
        } {vector=8}
      }
    }
  }
  return
}

// -----

// 4D transpose: expand_shape + transpose pattern, load src[i,j,k,l] store dst[i,k,l,j]
// load src[i,j,k,l] store dst[i,k,l,j] (swap dim 1 and 3)

#map_4d_mul = affine_map<(d0)[s0] -> (d0 * s0)>
#map_4d_min = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @test_transpose_affine_bounds_4d
func.func @test_transpose_affine_bounds_4d(
    %src: memref<4x8x16x32xf32>, %dst: memref<4x16x32x8xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index

  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: npuvector.transfer_read
  // CHECK: npuvector.transpose
  // CHECK: npuvector.transfer_write

  scf.for %b0 = %c0 to %c1 step %c1 {
    scf.for %b1 = %c0 to %c1 step %c1 {
      scf.for %b2 = %c0 to %c1 step %c1 {
        scf.for %b3 = %c0 to %c1 step %c1 {
          %lb0 = affine.apply #map_4d_mul(%b0)[%c4]
          %b0p1 = arith.addi %b0, %c1 : index
          %ub0_raw = affine.apply #map_4d_mul(%b0p1)[%c4]
          %ub0 = affine.min #map_4d_min(%ub0_raw, %c4)
          scf.for %i = %lb0 to %ub0 step %c1 {
            %lb1 = affine.apply #map_4d_mul(%b1)[%c8]
            %b1p1 = arith.addi %b1, %c1 : index
            %ub1_raw = affine.apply #map_4d_mul(%b1p1)[%c8]
            %ub1 = affine.min #map_4d_min(%ub1_raw, %c8)
            scf.for %j = %lb1 to %ub1 step %c1 {
              %lb2 = affine.apply #map_4d_mul(%b2)[%c16]
              %b2p1 = arith.addi %b2, %c1 : index
              %ub2_raw = affine.apply #map_4d_mul(%b2p1)[%c16]
              %ub2 = affine.min #map_4d_min(%ub2_raw, %c16)
              scf.for %k = %lb2 to %ub2 step %c1 {
                %lb3 = affine.apply #map_4d_mul(%b3)[%c32]
                %b3p1 = arith.addi %b3, %c1 : index
                %ub3_raw = affine.apply #map_4d_mul(%b3p1)[%c32]
                %ub3 = affine.min #map_4d_min(%ub3_raw, %c32)
                scf.for %l = %lb3 to %ub3 step %c1 {
                  %v = memref.load %src[%i, %j, %k, %l] : memref<4x8x16x32xf32>
                  memref.store %v, %dst[%i, %k, %l, %j] : memref<4x16x32x8xf32>
                } {vector=32}
              } {vector=16}
            } {vector=8}
          } {vector=4}
        }
      }
    }
  }
  return
}

// -----

// Mixed: outer static bounds, inner affine bounds, 2D transpose

#map_mixed_mul = affine_map<(d0)[s0] -> (d0 * s0)>
#map_mixed_min = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @test_transpose_mixed_static_affine_inner
func.func @test_transpose_mixed_static_affine_inner(
    %a: memref<16x32xf32>, %out: memref<32x16xf32>, %tile: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index

  // CHECK: affine.apply
  // CHECK: affine.min
  // CHECK: npuvector.transfer_read
  // CHECK: npuvector.transpose {{.*}} [1, 0]
  // CHECK: npuvector.transfer_write

  scf.for %i = %c0 to %c16 step %c1 {
    %lb_mul = affine.apply #map_mixed_mul(%c0)[%tile]
    %lb = affine.min #map_mixed_min(%lb_mul, %c32)
    %ub_mul = affine.apply #map_mixed_mul(%c1)[%tile]
    %ub = affine.min #map_mixed_min(%ub_mul, %c32)
    scf.for %j = %lb to %ub step %c1 {
      %v = memref.load %a[%i, %j] : memref<16x32xf32>
      memref.store %v, %out[%j, %i] : memref<32x16xf32>
    } {vector=32}
  } {vector=16}
  return
}

// -----

// ============================================================================
// 7. Custom OP examples
// ============================================================================

// Type Conversion: Float Extension

// CHECK-LABEL: func.func @test_extf_static
func.func @test_extf_static(%input: memref<128xbf16>, %output: memref<128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xbf16>, !npuvector<128xbf16>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.extf %[[V_VEC]] {round_mode = #hivm.round_mode<round>} : !npuvector<128xbf16> to !npuvector<128xf32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xf32>, memref<128xf32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xbf16>
    %result = arith.extf %v {round_mode = #hivm.round_mode<round>} : bf16 to f32
    memref.store %result, %output[%i] : memref<128xf32>
  } {vector=128}
  return
}

// -----

// Type Conversion: Float Truncation

// CHECK-LABEL: func.func @test_truncf_static
func.func @test_truncf_static(%input: memref<128xf32>, %output: memref<128xbf16>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.truncf %[[V_VEC]] {round_mode = #hivm.round_mode<round>} : !npuvector<128xf32> to !npuvector<128xbf16>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xbf16>, memref<128xbf16>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xf32>
    %result = arith.truncf %v {round_mode = #hivm.round_mode<round>} : f32 to bf16
    memref.store %result, %output[%i] : memref<128xbf16>
  } {vector=128}
  return
}

// -----

// Type Conversion: Integer Extension (Signed)

// CHECK-LABEL: func.func @test_extsi_static
func.func @test_extsi_static(%input: memref<128xi8>, %output: memref<128xi32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xi8>, !npuvector<128xi8>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.extsi %[[V_VEC]] : !npuvector<128xi8> to !npuvector<128xi32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xi32>, memref<128xi32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xi8>
    %result = arith.extsi %v : i8 to i32
    memref.store %result, %output[%i] : memref<128xi32>
  } {vector=128}
  return
}

// -----

// Type Conversion: Integer Extension (Unsigned)

// CHECK-LABEL: func.func @test_extui_static
func.func @test_extui_static(%input: memref<128xi8>, %output: memref<128xi32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xi8>, !npuvector<128xi8>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.extui %[[V_VEC]] : !npuvector<128xi8> to !npuvector<128xi32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xi32>, memref<128xi32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xi8>
    %result = arith.extui %v : i8 to i32
    memref.store %result, %output[%i] : memref<128xi32>
  } {vector=128}
  return
}

// -----

// Type Conversion: Integer Truncation

// CHECK-LABEL: func.func @test_trunci_static
func.func @test_trunci_static(%input: memref<128xi32>, %output: memref<128xi8>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xi32>, !npuvector<128xi32>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.trunci %[[V_VEC]] : !npuvector<128xi32> to !npuvector<128xi8>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xi8>, memref<128xi8>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xi32>
    %result = arith.trunci %v : i32 to i8
    memref.store %result, %output[%i] : memref<128xi8>
  } {vector=128}
  return
}

// -----

// Type Conversion: Integer to Float (Signed)

// CHECK-LABEL: func.func @test_sitofp_static
func.func @test_sitofp_static(%input: memref<128xi32>, %output: memref<128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xi32>, !npuvector<128xi32>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.sitofp %[[V_VEC]] : !npuvector<128xi32> to !npuvector<128xf32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xf32>, memref<128xf32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xi32>
    %result = arith.sitofp %v : i32 to f32
    memref.store %result, %output[%i] : memref<128xf32>
  } {vector=128}
  return
}

// -----

// Type Conversion: Integer to Float (Unsigned)

// CHECK-LABEL: func.func @test_uitofp_static
func.func @test_uitofp_static(%input: memref<128xi32>, %output: memref<128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xi32>, !npuvector<128xi32>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.uitofp %[[V_VEC]] : !npuvector<128xi32> to !npuvector<128xf32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xf32>, memref<128xf32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xi32>
    %result = arith.uitofp %v : i32 to f32
    memref.store %result, %output[%i] : memref<128xf32>
  } {vector=128}
  return
}

// -----

// Type Conversion: Float to Integer (Signed)

// CHECK-LABEL: func.func @test_fptosi_static
func.func @test_fptosi_static(%input: memref<128xf32>, %output: memref<128xi32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.fptosi %[[V_VEC]] : !npuvector<128xf32> to !npuvector<128xi32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xi32>, memref<128xi32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xf32>
    %result = arith.fptosi %v : f32 to i32
    memref.store %result, %output[%i] : memref<128xi32>
  } {vector=128}
  return
}

// -----

// Type Conversion: Float to Integer (Unsigned)

// CHECK-LABEL: func.func @test_fptoui_static
func.func @test_fptoui_static(%input: memref<128xf32>, %output: memref<128xi32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.fptoui %[[V_VEC]] : !npuvector<128xf32> to !npuvector<128xi32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xi32>, memref<128xi32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xf32>
    %result = arith.fptoui %v : f32 to i32
    memref.store %result, %output[%i] : memref<128xi32>
  } {vector=128}
  return
}

// -----

// Type Conversion: Bitcast

// CHECK-LABEL: func.func @test_bitcast_static
func.func @test_bitcast_static(%input: memref<128xbf16>, %output: memref<128xi16>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xbf16>, !npuvector<128xbf16>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.bitcast %[[V_VEC]] : !npuvector<128xbf16> to !npuvector<128xi16>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xi16>, memref<128xi16>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xbf16>
    %result = arith.bitcast %v : bf16 to i16
    memref.store %result, %output[%i] : memref<128xi16>
  } {vector=128}
  return
}

// -----

// Type Conversion: Mixed Operations

// CHECK-LABEL: func.func @test_mixed_type_conversion
func.func @test_mixed_type_conversion(%input: memref<128xbf16>, %output: memref<128xbf16>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %scale = arith.constant 2.0 : f32

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xbf16>, !npuvector<128xbf16>
  // CHECK: %[[EXT_VEC:.*]] = npuvector.extf %[[V_VEC]] : !npuvector<128xbf16> to !npuvector<128xf32>
  // CHECK: %[[SCALE_VEC:.*]] = npuvector.broadcast %{{.*}} : f32 to !npuvector<128xf32>
  // CHECK: %[[MUL_VEC:.*]] = arith.mulf %[[EXT_VEC]], %[[SCALE_VEC]] : !npuvector<128xf32>
  // CHECK: %[[TRUNC_VEC:.*]] = npuvector.truncf %[[MUL_VEC]] : !npuvector<128xf32> to !npuvector<128xbf16>
  // CHECK: npuvector.transfer_write %[[TRUNC_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xbf16>, memref<128xbf16>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xbf16>
    %v_f32 = arith.extf %v : bf16 to f32
    %scaled = arith.mulf %v_f32, %scale : f32
    %result = arith.truncf %scaled : f32 to bf16
    memref.store %result, %output[%i] : memref<128xbf16>
  } {vector=128}
  return
}

// -----

// Type Conversion: Dynamic Shape

// CHECK-LABEL: func.func @test_extf_dynamic
func.func @test_extf_dynamic(%input: memref<?xbf16>, %output: memref<?xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %N = memref.dim %input, %c0 : memref<?xbf16>

  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xbf16>
  // CHECK: %[[MAX:.*]] = arith.constant 4096 : index
  // CHECK: %[[VEC_SIZE:.*]] = affine.min
  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VEC_SIZE]]] [%[[MAX]]], %{{.*}} : memref<?xbf16>, !npuvector<?xbf16>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.extf %[[V_VEC]] : !npuvector<?xbf16> to !npuvector<?xf32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<?xf32>, memref<?xf32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %N step %c1 {
    %v = memref.load %input[%i] : memref<?xbf16>
    %result = arith.extf %v : bf16 to f32
    memref.store %result, %output[%i] : memref<?xf32>
  } {vector=4096}
  return
}

// -----

// CmpIOp

// CHECK-LABEL: func.func @test_cmpi_static
func.func @test_cmpi_static(%input: memref<128xi32>, %output: memref<128xi1>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %threshold = arith.constant 100 : i32

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xi32>, !npuvector<128xi32>
  // CHECK: %[[THRESH_VEC:.*]] = npuvector.broadcast %{{.*}} : i32 to !npuvector<128xi32>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.cmpi sgt, %[[V_VEC]], %[[THRESH_VEC]] : !npuvector<128xi32> to !npuvector<128xi1>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xi1>, memref<128xi1>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xi32>
    %result = arith.cmpi sgt, %v, %threshold : i32
    memref.store %result, %output[%i] : memref<128xi1>
  } {vector=128}
  return
}

// -----

// CmpFOp

// CHECK-LABEL: func.func @test_cmpf_static
func.func @test_cmpf_static(%input: memref<128xf32>, %output: memref<128xi1>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32

  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK: %[[ZERO_VEC:.*]] = npuvector.broadcast %{{.*}} : f32 to !npuvector<128xf32>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.cmpf ogt, %[[V_VEC]], %[[ZERO_VEC]] : !npuvector<128xf32> to !npuvector<128xi1>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xi1>, memref<128xi1>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xf32>
    %result = arith.cmpf ogt, %v, %zero : f32
    memref.store %result, %output[%i] : memref<128xi1>
  } {vector=128}
  return
}

// -----

// SelectOp

// CHECK-LABEL: func.func @test_select_static
func.func @test_select_static(
    %cond: memref<128xi1>,
    %input_a: memref<128xf32>,
    %input_b: memref<128xf32>,
    %output: memref<128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[COND_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xi1>, !npuvector<128xi1>
  // CHECK: %[[A_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK: %[[B_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.select %[[COND_VEC]], %[[A_VEC]], %[[B_VEC]] : !npuvector<128xi1>, !npuvector<128xf32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xf32>, memref<128xf32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %c = memref.load %cond[%i] : memref<128xi1>
    %a = memref.load %input_a[%i] : memref<128xf32>
    %b = memref.load %input_b[%i] : memref<128xf32>
    %result = arith.select %c, %a, %b : f32
    memref.store %result, %output[%i] : memref<128xf32>
  } {vector=128}
  return
}

// -----

// SCF.If: Conditional operation with side effects (store)

// CHECK-LABEL: func.func @test_scf_if_with_store
func.func @test_scf_if_with_store(
    %input: memref<128xf32>,
    %output: memref<128xf32>,
    %threshold: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32

  // CHECK: %[[INPUT_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK: scf.if %{{.*}} {
  // CHECK:   %[[SCALE_VEC:.*]] = npuvector.broadcast %{{.*}} : f32 to !npuvector<128xf32>
  // CHECK:   %[[SCALED_VEC:.*]] = arith.mulf %[[INPUT_VEC]], %[[SCALE_VEC]] : !npuvector<128xf32>
  // CHECK:   npuvector.transfer_write %[[SCALED_VEC]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector<128xf32>, memref<128xf32>
  // CHECK: }
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %val = memref.load %input[%i] : memref<128xf32>
    %cond = arith.cmpi ult, %i, %threshold {skip_vectorize} : index
    scf.if %cond {
      %scaled = arith.mulf %val, %c2 : f32
      memref.store %scaled, %output[%i] : memref<128xf32>
    }
  } {vector=128}
  return
}

// -----

// Four tests for if-conditions along the vectorization axis (slice path):
// (1) sge: affine(-IV+117) >= 0, lb=0, cmpi RHS 0
// (2) sle: IV <= 117, lb=0, cmpi RHS 117
// (3) Non-zero lb: same predicate as (1), lb=32..160 (trip 128), cmpi RHS still 0
// (4) Non-zero RHS constant: IV <= 100, lb=0, cmpi RHS 100

#map_if_iv = affine_map<(d0) -> (-d0 + 117)>
#map_if_iv_id = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func.func @test_scf_if_iv_affine_upper_bound
func.func @test_scf_if_iv_affine_upper_bound(%m: memref<128xf32>, %out: memref<128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: affine.min
  // CHECK: affine.max
  // CHECK: affine.apply
  // CHECK: arith.cmpi ne
  // CHECK: scf.if
  // CHECK: npuvector.transfer_read %{{.*}}[%{{.*}}] [{{.*}}] [{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<?xf32>
  // CHECK: npuvector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}]{{.*}}: !npuvector<?xf32>, memref<128xf32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %t = affine.apply #map_if_iv(%i)
    %cond = arith.cmpi sge, %t, %c0 {skip_vectorize} : index
    scf.if %cond {
      %v = memref.load %m[%i] : memref<128xf32>
      memref.store %v, %out[%i] : memref<128xf32>
    }
  } {vector=128}
  return
}

// -----

// CHECK-LABEL: func.func @test_scf_if_iv_affine_sle
func.func @test_scf_if_iv_affine_sle(%m: memref<128xf32>, %out: memref<128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c117 = arith.constant 117 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index

  // CHECK: affine.min
  // CHECK: affine.max
  // CHECK: affine.apply
  // CHECK: arith.cmpi ne
  // CHECK: scf.if
  // CHECK: npuvector.transfer_read
  // CHECK: npuvector.transfer_write
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %t = affine.apply #map_if_iv_id(%i)
    %cond = arith.cmpi sle, %t, %c117 {skip_vectorize} : index
    scf.if %cond {
      %v = memref.load %m[%i] : memref<128xf32>
      memref.store %v, %out[%i] : memref<128xf32>
    }
  } {vector=128}
  return
}

// -----

// Tile [32,160) ∩ [0,118) = [32,118), hence extent=86.
// CHECK-LABEL: func.func @test_scf_if_iv_nonzero_lb
// CHECK-DAG: arith.constant 32 : index
// CHECK: affine.min
// CHECK: affine.max
// CHECK: affine.apply
// CHECK: arith.cmpi ne
// CHECK: scf.if
// CHECK: npuvector.transfer_read
// CHECK: npuvector.transfer_write
// CHECK-NOT: scf.for
func.func @test_scf_if_iv_nonzero_lb(%m: memref<256xf32>, %out: memref<256xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c160 = arith.constant 160 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c32 to %c160 step %c1 {
    %t = affine.apply #map_if_iv(%i)
    %cond = arith.cmpi sge, %t, %c0 {skip_vectorize} : index
    scf.if %cond {
      %v = memref.load %m[%i] : memref<256xf32>
      memref.store %v, %out[%i] : memref<256xf32>
    }
  } {vector=128}
  return
}

// -----

// Tile [0,128) ∩ [0,101) = [0,101), hence extent=101; slice constant 101 from RHS 100 (sle).
// CHECK-LABEL: func.func @test_scf_if_iv_nonzero_rhs
// CHECK: affine.min
// CHECK: affine.max
// CHECK: affine.apply
// CHECK: arith.cmpi ne
// CHECK: scf.if
// CHECK: npuvector.transfer_read
// CHECK: npuvector.transfer_write
// CHECK-NOT: scf.for
func.func @test_scf_if_iv_nonzero_rhs(%m: memref<128xf32>, %out: memref<128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c128 step %c1 {
    %t = affine.apply #map_if_iv_id(%i)
    %cond = arith.cmpi sle, %t, %c100 {skip_vectorize} : index
    scf.if %cond {
      %v = memref.load %m[%i] : memref<128xf32>
      memref.store %v, %out[%i] : memref<128xf32>
    }
  } {vector=128}
  return
}

// ============================================================================
// Phase 2 (rank-0 sweep): no tagged loops — scalar memref.load chain only
// ============================================================================

// -----

#map_vector_axis_minus_base = affine_map<(d0, d1) -> (d0 - d1)>

// CHECK-LABEL: func.func @test_vector_axis_affine_derived_store
// CHECK: %[[IDX:.*]] = affine.apply
// CHECK: npuvector.transfer_read
// CHECK: npuvector.transfer_write %{{.*}}, %{{.*}}[%[[IDX]]]
// CHECK-NOT: memref.store
// CHECK-NOT: scf.for
func.func @test_vector_axis_affine_derived_store(%input: memref<128xf32>, %tmp: memref<128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c16 = arith.constant 16 : index
  %c80 = arith.constant 80 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c16 to %c80 step %c1 {
    %idx = affine.apply #map_vector_axis_minus_base(%i, %c16)
    %value = memref.load %input[%i] : memref<128xf32>
    memref.store %value, %tmp[%idx] : memref<128xf32>
  } {vector = 64 : i64}
  return
}

// -----

// CHECK-LABEL: func.func @test_vector_axis_affine_derived_load
// CHECK: %[[IDX:.*]] = affine.apply
// CHECK: %[[READ:.*]] = npuvector.transfer_read %{{.*}}[%[[IDX]]]
// CHECK: npuvector.transfer_write %[[READ]], %{{.*}}[%{{.*}}]
// CHECK-NOT: memref.load
// CHECK-NOT: scf.for
func.func @test_vector_axis_affine_derived_load(%tmp: memref<128xf32>, %output: memref<128xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c16 = arith.constant 16 : index
  %c80 = arith.constant 80 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c16 to %c80 step %c1 {
    %idx = affine.apply #map_vector_axis_minus_base(%i, %c16)
    %value = memref.load %tmp[%idx] : memref<128xf32>
    memref.store %value, %output[%i] : memref<128xf32>
  } {vector = 64 : i64}
  return
}

// CHECK-LABEL: func.func @test_vf1_sweep_no_tagged_loops
// CHECK: npuvector.transfer_read {{.*}} : memref<f32>, !npuvector.f32
// CHECK: arith.mulf {{.*}} : !npuvector.f32
// CHECK: npuvector.transfer_write {{.*}} : !npuvector.f32, memref<f32>
func.func @test_vf1_sweep_no_tagged_loops(%in: memref<f32>, %out: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %x = memref.load %in[] : memref<f32>
  %c2 = arith.constant 2.000000e+00 : f32
  %m = arith.mulf %x, %c2 : f32
  memref.store %m, %out[] : memref<f32>
  return
}

// -----

// Phase 2 promotes the loop-carried lane when a promoted load chain reaches scf.yield.
// CHECK-LABEL: func.func @test_vf1_sweep_promotes_for_iter_arg
// CHECK: %[[INIT_VEC:.*]] = npuvector.broadcast %{{.*}} : f16 to !npuvector.f16
// CHECK: %[[SUM:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT_VEC]]) -> (!npuvector.f16) {
// CHECK:   %[[READ:.*]] = npuvector.transfer_read %{{.*}}[]{{.*}} : memref<f16>, !npuvector.f16
// CHECK:   %[[NEXT:.*]] = arith.addf %[[ACC]], %[[READ]] : !npuvector.f16
// CHECK:   scf.yield %[[NEXT]] : !npuvector.f16
// CHECK: }
// CHECK: %[[BIASED:.*]] = arith.addf %[[SUM]], %{{.*}} : !npuvector.f16
// CHECK: npuvector.transfer_write %[[BIASED]], %{{.*}}[]{{.*}}: !npuvector.f16, memref<f16>
// CHECK-NOT: memref.load
// CHECK-NOT: memref.store
func.func @test_vf1_sweep_promotes_for_iter_arg(%input: memref<f16>, %output: memref<f16>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.000000e+00 : f16
  %one = arith.constant 1.000000e+00 : f16
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> f16 {
    %value = memref.load %input[] : memref<f16>
    %next = arith.addf %acc, %value : f16
    scf.yield %next : f16
  }
  %biased = arith.addf %sum, %one : f16
  memref.store %biased, %output[] : memref<f16>
  return
}

// -----

// Phase 2 also enters a loop-carried lane when the promoted load is the init operand.
// CHECK-LABEL: func.func @test_vf1_sweep_promotes_for_load_init
// CHECK: %[[INIT:.*]] = npuvector.transfer_read %{{.*}}[]{{.*}} : memref<f16>, !npuvector.f16
// CHECK-NOT: npuvector.broadcast %[[INIT]]
// CHECK: %[[SUM:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (!npuvector.f16) {
// CHECK:   %[[NEXT:.*]] = arith.addf %[[ACC]], %{{.*}} : !npuvector.f16
// CHECK:   scf.yield %[[NEXT]] : !npuvector.f16
// CHECK: }
// CHECK: npuvector.transfer_write %[[SUM]], %{{.*}}[]{{.*}}: !npuvector.f16, memref<f16>
// CHECK-NOT: memref.load
// CHECK-NOT: memref.store
func.func @test_vf1_sweep_promotes_for_load_init(%input: memref<f16>, %output: memref<f16>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %one = arith.constant 1.000000e+00 : f16
  %init = memref.load %input[] : memref<f16>
  %sum = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> f16 {
    %next = arith.addf %acc, %one : f16
    scf.yield %next : f16
  }
  memref.store %sum, %output[] : memref<f16>
  return
}

// -----

// Phase 1 inlines vector=1; 0-d load stays scalar until Phase 2 promotes to !npuvector.f32.
// CHECK-LABEL: func.func @test_vf1_sweep_after_vector1_collapse
// CHECK: memref.collapse_shape
// CHECK: npuvector.transfer_read {{.*}} : memref<f32>, !npuvector.f32
// CHECK: arith.mulf {{.*}} : !npuvector.f32
// CHECK: npuvector.transfer_write {{.*}} : !npuvector.f32, memref<f32>
func.func @test_vf1_sweep_after_vector1_collapse(%a: memref<1xf32>, %b: memref<1xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %ca = memref.collapse_shape %a [] : memref<1xf32> into memref<f32>
    %cb = memref.collapse_shape %b [] : memref<1xf32> into memref<f32>
    %x = memref.load %ca[] : memref<f32>
    %y = arith.mulf %x, %x : f32
    memref.store %y, %cb[] : memref<f32>
  } {vector = 1 : i64}
  return
}

// -----

// Phase 1: single tagged elementwise loop (vector=4). Phase 2: scalar memref.load chains outside
// that loop (bias prologue + tail) promote to !npuvector.f32 (batch-norm style).
// CHECK-LABEL: func.func @test_vf1_scalar_before_after_tagged_elementwise
// CHECK: npuvector.transfer_read {{.*}} : memref<4xf32>, !npuvector.f32
// CHECK: arith.mulf {{.*}} : !npuvector.f32
// CHECK: npuvector.transfer_read {{.*}} : memref<4xf32>, !npuvector<4xf32>
// CHECK: npuvector.transfer_write {{.*}} : !npuvector<4xf32>, memref<4xf32>
// CHECK: npuvector.transfer_read {{.*}} : memref<4xf32>, !npuvector.f32
// CHECK: arith.addf {{.*}} : !npuvector.f32
// CHECK: npuvector.transfer_write {{.*}} : !npuvector.f32, memref<4xf32>
func.func @test_vf1_scalar_before_after_tagged_elementwise(
    %bias: memref<4xf32>,
    %in: memref<4xf32>,
    %out: memref<4xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %pre = memref.load %bias[%c0] : memref<4xf32>
  %preScaled = arith.mulf %pre, %pre : f32
  scf.for %i = %c0 to %c4 step %c1 {
    %v = memref.load %in[%i] : memref<4xf32>
    %x = arith.mulf %v, %preScaled : f32
    memref.store %x, %out[%i] : memref<4xf32>
  } {vector = 4 : i64}
  %post = memref.load %out[%c0] : memref<4xf32>
  %postAcc = arith.addf %post, %preScaled : f32
  memref.store %postAcc, %bias[%c0] : memref<4xf32>
  return
}

// -----

// Phase 2's store sweep writes a rank-0 reduction result without a scalar load seed.
// CHECK-LABEL: func.func @test_rank_zero_store_after_reduction
// CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %{{.*}} {reduction_dims = array<i64: 0>} : !npuvector<64xf32> into !npuvector.f32
// CHECK: npuvector.transfer_write %[[SUM]], %{{.*}}[%{{.*}}]{{.*}}: !npuvector.f32, memref<128xf32>
func.func @test_rank_zero_store_after_reduction(%input: memref<64xf32>, %output: memref<128xf32>, %idx: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.000000e+00 : f32

  %sum = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init) -> f32 {
    %value = memref.load %input[%i] : memref<64xf32>
    %next = arith.addf %acc, %value {reduction_axes = [0 : index], reduction_type = "x"} : f32
    scf.yield %next : f32
  } {reduction_x = 64 : i64}

  memref.store %sum, %output[%idx] : memref<128xf32>
  return
}

// -----

// Phase 2 promotes dangling rank-0 arith whose source is a reduction result after load/store sweeps.
// CHECK-LABEL: func.func @test_vf1_arith_after_reduction_without_load_or_store_seed
// CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %{{.*}} {reduction_dims = array<i64: 0>} : !npuvector<64xf32> into !npuvector.f32
// CHECK: %[[HALF_VEC:.*]] = npuvector.broadcast %{{.*}} : f32 to !npuvector.f32
// CHECK: %[[SCALED:.*]] = arith.mulf %[[SUM]], %[[HALF_VEC]] : !npuvector.f32
// CHECK: %[[SEED:.*]] = npuvector.transfer_read %{{.*}}[]{{.*}} : memref<f32>, !npuvector.f32
// CHECK: %[[SEED_SQ:.*]] = arith.mulf %[[SEED]], %[[SEED]] : !npuvector.f32
// CHECK: %[[MIX:.*]] = arith.mulf {{.*}}, %[[SEED_SQ]] : !npuvector.f32
// CHECK: npuvector.transfer_write %[[MIX]], %{{.*}}[]{{.*}}: !npuvector.f32, memref<f32>
func.func @test_vf1_arith_after_reduction_without_load_or_store_seed(
    %input: memref<64xf32>,
    %seed: memref<f32>,
    %output: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.000000e+00 : f32
  %half = arith.constant 5.000000e-01 : f32

  %sum = scf.for %i = %c0 to %c64 step %c1 iter_args(%acc = %init) -> f32 {
    %value = memref.load %input[%i] : memref<64xf32>
    %next = arith.addf %acc, %value {reduction_axes = [0 : index], reduction_type = "x"} : f32
    scf.yield %next : f32
  } {reduction_x = 64 : i64}

  %scaled = arith.mulf %sum, %half : f32
  %seed_value = memref.load %seed[] : memref<f32>
  %seed_square = arith.mulf %seed_value, %seed_value : f32
  %mix = arith.mulf %scaled, %seed_square : f32
  memref.store %mix, %output[] : memref<f32>
  return
}

// -----

// CHECK-LABEL: func.func @test_nested_vector_reduction_x_rank_lift_accumulator_dim_order
// CHECK: %[[VEC_ADD:.*]] = arith.addf {{.*}} : !npuvector<64xf16>
// CHECK: %[[LIFT_ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[NEUTRAL:.*]] = npuvector.broadcast %[[LIFT_ZERO]]{{.*}} : f16 to !npuvector<64x8xf16>
// CHECK: %[[LIFT_BODY:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[NEUTRAL]]) -> (!npuvector<64x8xf16>) {
// CHECK: %[[LIFTED:.*]] = npuvector.broadcast %[[VEC_ADD]][{{.*}}] [{{.*}}] : !npuvector<64xf16> to !npuvector<64x8xf16>
// CHECK-SAME: {dimension = array<i64: 0>}
// CHECK: arith.addf %[[LIFTED]], %[[ACC]] {{.*}} : !npuvector<64x8xf16>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[LIFT_TAIL:.*]] = scf.for {{.*}} iter_args(%[[LIFT_TACC:.*]] = %[[LIFT_BODY]]) -> (!npuvector<64x8xf16>) {
// CHECK: %[[LIFT_SLICE:.*]] = npuvector.extract_slice %[[LIFT_TACC]]{{.*}} : !npuvector<64x8xf16> to !npuvector<64x1xf16>
// CHECK: %[[LIFT_TAIL_VALUE:.*]] = npuvector.broadcast %[[VEC_ADD]][{{.*}}] [{{.*}}] : !npuvector<64xf16> to !npuvector<64x1xf16> {dimension = array<i64: 0>}
// CHECK: %[[LIFT_NEXT:.*]] = arith.addf %[[LIFT_TAIL_VALUE]], %[[LIFT_SLICE]] {{.*}} : !npuvector<64x1xf16>
// CHECK: %[[LIFT_INSERT:.*]] = npuvector.insert_slice %[[LIFT_NEXT]] into %[[LIFT_TACC]]{{.*}} : !npuvector<64x1xf16> into !npuvector<64x8xf16>
// CHECK-NEXT: scf.yield %[[LIFT_INSERT]] : !npuvector<64x8xf16>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[LIFT_REDUCED:.*]] = npuvector.reduction <add>, %[[LIFT_TAIL]] {reduction_dims = array<i64: 1>} : !npuvector<64x8xf16> into !npuvector<64xf16>
// CHECK: npuvector.extf %[[LIFT_REDUCED]] : !npuvector<64xf16> to !npuvector<64xf32>
func.func @test_nested_vector_reduction_x_rank_lift_accumulator_dim_order(
    %input: memref<32x64xf16>,
    %output: memref<64x32xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0.000000e+00 : f16
  %one = arith.constant 1.000000e+00 : f16

  scf.for %row = %c0 to %c32 step %c1 {
    scf.for %col = %c0 to %c64 step %c1 {
      %value = memref.load %input[%row, %col] : memref<32x64xf16>
      %plus = arith.addf %value, %one : f16
      %sum = scf.for %red = %c0 to %c9 step %c1 iter_args(%acc = %zero) -> (f16) {
        %next = arith.addf %plus, %acc {reduction_axes = [2 : index], reduction_type = "x"} : f16
        scf.yield %next : f16
      } {reduction_x = 8 : i64}
      %cast = arith.extf %sum : f16 to f32
      memref.store %cast, %output[%col, %row] : memref<64x32xf32>
    } {vector = 64 : i64}
  }
  return
}

// -----

// AllocBufferShrink scratch: an inner copy loop writes a high-rank tile into a
// shrunk local buffer; fixed-channel reads must be forwarded via
// npuvector.extract_slice instead of a (impossible) write to the 1x2 buffer.
// CHECK-LABEL: func.func @test_scratch_extract_slice
// CHECK: %[[TILE:.*]] = npuvector.transfer_read %{{.*}} : memref<8x2xf32>, !npuvector<8x2xf32>
// CHECK: %[[S0:.*]] = npuvector.extract_slice %[[TILE]]{{.*}} {keep_dims = array<i64: 0>} : !npuvector<8x2xf32> to !npuvector<8xf32>
// CHECK: %[[S1:.*]] = npuvector.extract_slice %[[TILE]]{{.*}} {keep_dims = array<i64: 0>} : !npuvector<8x2xf32> to !npuvector<8xf32>
// CHECK: arith.addf %[[S0]], %[[S1]] : !npuvector<8xf32>
// CHECK: npuvector.transfer_write %{{.*}}, %{{.*}} : !npuvector<8xf32>, memref<8xf32>
// CHECK-NOT: npuvector.transfer_write %{{.*}} : !npuvector{{.*}}, memref<1x2xf32>
func.func @test_scratch_extract_slice(
    %A: memref<8x2xf32>,
    %O: memref<8xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %alloc = memref.alloc() : memref<1x2xf32>
  %ch0 = memref.subview %alloc[0, 0] [1, 1] [1, 1] : memref<1x2xf32> to memref<1x1xf32, strided<[2, 1]>>
  %ch1 = memref.subview %alloc[0, 1] [1, 1] [1, 1] : memref<1x2xf32> to memref<1x1xf32, strided<[2, 1], offset: 1>>
  scf.for %i = %c0 to %c8 step %c1 {
    scf.for %k = %c0 to %c2 step %c1 {
      %a = memref.load %A[%i, %k] : memref<8x2xf32>
      memref.store %a, %alloc[%c0, %k] : memref<1x2xf32>
    } {vector = 16 : i64}
    %v0 = memref.load %ch0[%c0, %c0] : memref<1x1xf32, strided<[2, 1]>>
    %v1 = memref.load %ch1[%c0, %c0] : memref<1x1xf32, strided<[2, 1], offset: 1>>
    %s = arith.addf %v0, %v1 : f32
    memref.store %s, %O[%i] : memref<8xf32>
  } {vector = 16 : i64}
  return
}

// -----

// A value that is invariant along all scratch indices is replicated by the
// scalar store. A same-index load can be forwarded directly from the tile.
// CHECK-LABEL: func.func @test_scratch_replicated_store_load
// CHECK: %[[TILE:.*]] = npuvector.transfer_read %{{.*}} : memref<8xf32>, !npuvector<8xf32>
// CHECK-NOT: npuvector.transfer_write %[[TILE]]{{.*}}memref<1x2x16xf32>
// CHECK-NOT: npuvector.transfer_read %{{.*}} : memref<1x2x16xf32>
// CHECK: npuvector.broadcast %[[TILE]]
func.func @test_scratch_replicated_store_load(
    %A: memref<8xf32>,
    %O: memref<8xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<1x2x16xf32>
  scf.for %i = %c0 to %c8 step %c1 {
    %value = memref.load %A[%i] : memref<8xf32>
    %outer = scf.for %j = %c0 to %c2 step %c1 iter_args(%outerAcc = %zero) -> (f32) {
      %inner = scf.for %k = %c0 to %c16 step %c1 iter_args(%innerAcc = %zero) -> (f32) {
        memref.store %value, %alloc[%c0, %j, %k] : memref<1x2x16xf32>
        %loaded = memref.load %alloc[%c0, %j, %k] : memref<1x2x16xf32>
        %next = arith.addf %loaded, %innerAcc {reduction_axes = [1 : index, 2 : index], reduction_type = "x"} : f32
        scf.yield %next : f32
      } {reduction_x = 16 : i64}
      %nextOuter = arith.addf %inner, %outerAcc : f32
      scf.yield %nextOuter : f32
    } {reduction_x = 2 : i64}
    memref.store %outer, %O[%i] : memref<8xf32>
  } {vector = 8 : i64}
  return
}

// -----

// Inner reduction_x only; outer iter_arg loop has no vectorization attr. Phase 2 arith VF1
// sweep must promote the outer loop lane when combining rank-0 reduction results.
// CHECK-LABEL: func.func @test_reduction_x_outer_iter_arg_vf1_lane_promotion
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = {{.*}}) -> (!npuvector.f32) {
// CHECK: npuvector.reduction
// CHECK: scf.yield %{{.*}} : !npuvector.f32
// CHECK-NOT: scf.yield %{{.*}} : f32
func.func @test_reduction_x_outer_iter_arg_vf1_lane_promotion(
    %data: memref<2x2x16xf32>,
    %weight: memref<2x2xf32>,
    %out: memref<2xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32.0 : f32
  %zero = arith.constant 0.000000e+00 : f32
  scf.for %g = %c0 to %c2 step %c1 {
    %sum = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %zero) -> (f32) {
      %w = memref.load %weight[%g, %i] : memref<2x2xf32>
      %inner = scf.for %k = %c0 to %c16 step %c1 iter_args(%innerAcc = %zero) -> (f32) {
        %x = memref.load %data[%g, %i, %k] : memref<2x2x16xf32>
        %t = arith.addf %x, %w : f32
        %next = arith.addf %t, %innerAcc {reduction_axes = [1 : index, 2 : index], reduction_type = "x"} : f32
        scf.yield %next : f32
      } {reduction_x = 16 : i64}
      %nextOuter = arith.addf %inner, %acc : f32
      scf.yield %nextOuter : f32
    }
    %avg = arith.divf %sum, %c32 : f32
    memref.store %avg, %out[%g] : memref<2xf32>
  }
  return
}

// -----

// Softmax-like: multi-dim reduction stores into a shrunk local scratch, then a
// sibling nest reloads the same alloc with a different tiling IV. Scratch state
// must propagate across createChild and forward via extract_slice (no physical
// transfer_read from the never-written alloc).
// CHECK-LABEL: func.func @test_scratch_reduction_store_sibling_load
// CHECK: %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<1x16xf32>
// CHECK: %[[RED:.*]] = npuvector.reduction
// CHECK: npuvector.extract_slice %[[RED]]
// CHECK-NOT: npuvector.transfer_read %[[ALLOC]]
func.func @test_scratch_reduction_store_sibling_load(
    %input: memref<2x2x16xf32>,
    %output: memref<2x2x16xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %neg_inf = arith.constant 0xFF800000 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16xf32>

  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %mx = scf.for %r = %c0 to %c2 step %c1 iter_args(%acc = %neg_inf) -> (f32) {
        %v = memref.load %input[%i, %r, %j] : memref<2x2x16xf32>
        %n = arith.maximumf %v, %acc {reduction_axes = [1 : index], reduction_type = "y"} : f32
        scf.yield %n : f32
      } {reduction_x = 8 : i64}
      memref.store %mx, %alloc[%c0, %j] : memref<1x16xf32>
    } {vector = 16 : i64}

    scf.for %t = %c0 to %c2 step %c1 {
      %lo = affine.apply affine_map<(d0) -> (d0 * 8)>(%t)
      %hi = affine.apply affine_map<(d0) -> (d0 * 8 + 8)>(%t)
      scf.for %r = %c0 to %c2 step %c1 {
        scf.for %k = %lo to %hi step %c1 {
          %m = memref.load %alloc[%c0, %k] : memref<1x16xf32>
          %x = memref.load %input[%i, %r, %k] : memref<2x2x16xf32>
          %d = arith.subf %x, %m : f32
          memref.store %d, %output[%i, %r, %k] : memref<2x2x16xf32>
        } {vector = 8 : i64}
      } {vector = 16 : i64}
    }
  } {vector = 2 : i64}
  return
}

// -----

// Reproduce a sliced-input reduction with a shrunk scratch buffer. Keep the
// original SSA block bounds (with proven constant difference 2), collapsed
// 7x7 dimensions, input offset 970,
// vector width 56 over 49 elements, and independently computed scratch +10
// indices. The transposed static tile is forwarded directly to the reduction;
// the scratch address omits both parallel vector axes.
// CHECK-LABEL: func.func @scratch_offset_slice_reduction
// CHECK: %[[INPUT_BIAS:.*]] = arith.constant 970 : index
// CHECK: %[[SRC_A:.*]] = memref.collapse_shape {{.*}} into memref<4x1024x49xf32>
// CHECK: %[[SRC_B:.*]] = memref.collapse_shape {{.*}} into memref<4x992x49xf32>
// CHECK: %[[DST:.*]] = memref.collapse_shape {{.*}} into memref<4x49xf32>
// CHECK: scf.for %[[BLOCK:.*]] =
// CHECK: %[[LB:.*]] = affine.apply #[[SCRATCH_LB_MAP]](%[[BLOCK]])
// CHECK: %[[UB:.*]] = affine.apply #[[SCRATCH_UB_MAP]](%[[BLOCK]])
// CHECK-NOT: affine.min
// CHECK: %[[LOOP_RESULT:.*]] = scf.for %[[RIV:.*]] = {{.*}} iter_args(%[[ACC0:.*]] = {{.*}}) -> (!npuvector<2x49x16xf32>) {
// CHECK: %[[INDEX_A:.*]] = arith.addi %[[RIV]], %[[INPUT_BIAS]] : index
// CHECK: %[[READ_A:.*]] = npuvector.transfer_read %[[SRC_A]][%[[LB]], %[[INDEX_A]], %{{.*}}] {{.*}} : memref<4x1024x49xf32>, !npuvector<2x16x49xf32>
// CHECK: %[[X0:.*]] = npuvector.transpose %[[READ_A]], [0, 2, 1] : !npuvector<2x16x49xf32> to !npuvector<2x49x16xf32>
// CHECK: %[[INDEX_B:.*]] = arith.addi %[[RIV]], %[[INPUT_BIAS]] : index
// CHECK: %[[READ_B:.*]] = npuvector.transfer_read %[[SRC_B]][%[[LB]], %[[INDEX_B]], %{{.*}}] {{.*}} : memref<4x992x49xf32>, !npuvector<2x16x49xf32>
// CHECK: %[[Y0:.*]] = npuvector.transpose %[[READ_B]], [0, 2, 1] : !npuvector<2x16x49xf32> to !npuvector<2x49x16xf32>
// CHECK: %[[TILE0:.*]] = arith.addf %[[X0]], %[[Y0]] : !npuvector<2x49x16xf32>
// CHECK-NOT: npuvector.extract_slice
// CHECK: arith.addf %[[TILE0]], %[[ACC0]] {{.*}} : !npuvector<2x49x16xf32>
// CHECK: %[[REDUCED:.*]] = npuvector.reduction <add>, %[[LOOP_RESULT]] {reduction_dims = array<i64: 2>} : !npuvector<2x49x16xf32> into !npuvector<2x49xf32>
// CHECK: npuvector.transfer_write %[[REDUCED]], %[[DST]][%[[LB]], %{{.*}}]
// NO-SCRATCH-LABEL: func.func @scratch_offset_slice_reduction
// NO-SCRATCH: %[[S0:.*]] = memref.alloc() {{.*}} : memref<1x32x1xf32>
// NO-SCRATCH-NOT: npuvector.transfer_read %[[S0]]
// NO-SCRATCH-NOT: npuvector.transfer_write {{.*}}, %[[S0]]
// NO-SCRATCH: return
func.func @scratch_offset_slice_reduction(%a: memref<4x1024x7x7xf32>, %b: memref<4x992x7x7xf32>,
                                         %out: memref<4x1x7x7xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c49 = arith.constant 49 : index
  %c10 = arith.constant 10 : index
  %c16 = arith.constant 16 : index
  %c970 = arith.constant 970 : index
  %zero = arith.constant 0.0 : f32
  %a_view = memref.collapse_shape %a [[0], [1], [2, 3]] : memref<4x1024x7x7xf32> into memref<4x1024x49xf32>
  %b_view = memref.collapse_shape %b [[0], [1], [2, 3]] : memref<4x992x7x7xf32> into memref<4x992x49xf32>
  %scratch = memref.alloc() {alignment = 64 : i64} : memref<1x32x1xf32>
  %out_view = memref.collapse_shape %out [[0], [1, 2, 3]] : memref<4x1x7x7xf32> into memref<4x49xf32>
  scf.for %block = %c0 to %c2 step %c1 {
    %lb = affine.apply affine_map<(d0) -> (d0 * 2 - (d0 floordiv 2) * 4)>(%block)
    %ub = affine.apply affine_map<(d0) -> (d0 * 2 - (d0 floordiv 2) * 4 + 2)>(%block)
    scf.for %i = %lb to %ub step %c1 {
      scf.for %j = %c0 to %c49 step %c1 {
        %sum = scf.for %k = %c0 to %c16 step %c1 iter_args(%acc = %zero) -> f32 {
          %input_index_a = arith.addi %k, %c970 : index
          %x = memref.load %a_view[%i, %input_index_a, %j] : memref<4x1024x49xf32>
          %input_index_b = arith.addi %k, %c970 : index
          %y = memref.load %b_view[%i, %input_index_b, %j] : memref<4x992x49xf32>
          %value = arith.addf %x, %y : f32
          %store_index = arith.addi %k, %c10 : index
          memref.store %value, %scratch[%c0, %store_index, %c0] : memref<1x32x1xf32>
          %load_index = arith.addi %k, %c10 : index
          %loaded = memref.load %scratch[%c0, %load_index, %c0] : memref<1x32x1xf32>
          %next = arith.addf %loaded, %acc {reduction_axes = [1 : index], reduction_type = "y"} : f32
          scf.yield %next : f32
        } {reduction_x = 16 : i64}
        memref.store %sum, %out_view[%i, %j] : memref<4x49xf32>
      } {vector = 56 : i64}
    } {vector = 2 : i64}
  } {map_for_to_forall}
  return
}

// -----

// With a nonzero LB, split [5, 24) at 21: body 2x8, tail 2x3.
// Scratch forwarding must use each segment's actual IV, preserve the parallel
// extent 2, and update the shared accumulator without materializing scratch.
// NO-SCRATCH-LABEL: func.func @scratch_offset_reduction_tiles_and_tail
// NO-SCRATCH: %[[S2:.*]] = memref.alloc() : memref<1x40xf32>
// NO-SCRATCH-NOT: npuvector.transfer_read %[[S2]]
// NO-SCRATCH-NOT: npuvector.transfer_write {{.*}}, %[[S2]]
// NO-SCRATCH: return
// CHECK-LABEL: func.func @scratch_offset_reduction_tiles_and_tail(
// CHECK: %[[S_LOW:.*]] = arith.constant 5 : index
// CHECK: %[[S_HIGH:.*]] = arith.constant 24 : index
// CHECK: %[[S_SPLIT:.*]] = arith.constant 21 : index
// CHECK: %[[S_BODY:.*]] = scf.for %[[S_BODY_IV:.*]] = %[[S_LOW]] to %[[S_SPLIT]] step %{{.*}} iter_args({{.*}}) -> (!npuvector<2x8xf32>) {
// CHECK: npuvector.transfer_read %{{.*}}[%{{.*}}, %[[S_BODY_IV]]]{{.*}} : memref<2x32xf32>, !npuvector<2x8xf32>
// CHECK-NOT: npuvector.extract_slice
// CHECK-NOT: npuvector.reduction
// CHECK: %[[S_TAIL:.*]] = scf.for %[[S_TAIL_IV:.*]] = %[[S_SPLIT]] to %[[S_HIGH]] step %{{.*}} iter_args(%[[S_TACC:.*]] = %[[S_BODY]]) -> (!npuvector<2x8xf32>) {
// CHECK: %[[S_SLICE:.*]] = npuvector.extract_slice %[[S_TACC]]{{.*}} {keep_dims = array<i64: 0, 1>} : !npuvector<2x8xf32> to !npuvector<2x3xf32>
// CHECK: %[[S_VALUE:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %[[S_TAIL_IV]]]{{.*}} : memref<2x32xf32>, !npuvector<2x3xf32>
// CHECK: %[[S_NEXT:.*]] = arith.addf %[[S_VALUE]], %[[S_SLICE]] {{.*}} : !npuvector<2x3xf32>
// CHECK: %[[S_INSERT:.*]] = npuvector.insert_slice %[[S_NEXT]] into %[[S_TACC]]{{.*}} : !npuvector<2x3xf32> into !npuvector<2x8xf32>
// CHECK-NEXT: scf.yield %[[S_INSERT]] : !npuvector<2x8xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[S_RESULT:.*]] = npuvector.reduction <add>, %[[S_TAIL]] {reduction_dims = array<i64: 1>} : !npuvector<2x8xf32> into !npuvector<2xf32>
// CHECK: npuvector.transfer_write %[[S_RESULT]], {{.*}} : !npuvector<2xf32>, memref<2xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: return
func.func @scratch_offset_reduction_tiles_and_tail(%a: memref<2x32xf32>, %out: memref<2xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %c24 = arith.constant 24 : index
  %zero = arith.constant 0.0 : f32
  %scratch = memref.alloc() : memref<1x40xf32>
  scf.for %i = %c0 to %c2 step %c1 {
    %sum = scf.for %k = %c5 to %c24 step %c1 iter_args(%acc = %zero) -> f32 {
      %value = memref.load %a[%i, %k] : memref<2x32xf32>
      %store_index = arith.addi %c10, %k : index
      memref.store %value, %scratch[%c0, %store_index] : memref<1x40xf32>
      %load_index = affine.apply affine_map<(d0) -> (d0 + 10)>(%k)
      %loaded = memref.load %scratch[%c0, %load_index] : memref<1x40xf32>
      %next = arith.addf %loaded, %acc {reduction_axes = [1 : index], reduction_type = "x"} : f32
      scf.yield %next : f32
    } {reduction_x = 8 : i64}
    memref.store %sum, %out[%i] : memref<2xf32>
  } {vector = 2 : i64}
  return
}

// -----

// Keep the six-iteration carrier inside each block. Each half-head owns 256
// rows: five 47-row tiles and one 21-row tile. The body and tail carriers
// must stay inside that carrier, including the original stride-two BF16 view.
// CHECK-LABEL: func.func @test_piecewise_clipped_elementwise_64blocks(
// CHECK-DAG: %[[E_ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[E_ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[E_SIX:.*]] = arith.constant 6 : index
// CHECK: scf.forall (%[[E_BLOCK:.*]]) in (64) {
// CHECK: %[[E_HEAD:.*]] = affine.apply
// CHECK: scf.for %[[E_TILE:.*]] = %[[E_ZERO]] to %[[E_SIX]] step %[[E_ONE]] {
// CHECK: %[[E_LOW:.*]] = affine.min
// CHECK: %[[E_HIGH:.*]] = affine.min
// CHECK: %[[E_WIDTH:.*]] = arith.constant 47 : index
// CHECK: scf.for %[[E_ROW:.*]] = %[[E_LOW]] to %[[E_BODY_END:.*]] step %[[E_WIDTH]] {
// CHECK-NOT: scf.if
// CHECK-NOT: !npuvector<{{[^>]*[?]}}
// CHECK: %[[E_BODY:.*]] = npuvector.transfer_read %{{.*}}[%[[E_HEAD]], %[[E_ROW]], %{{.*}}]{{.*}} : {{.*}}, !npuvector<47x32xbf16>
// CHECK-NOT: !npuvector<{{[^>]*[?]}}
// CHECK: arith.mulf {{.*}} : !npuvector<47x32xf32>
// CHECK-NOT: !npuvector<{{[^>]*[?]}}
// CHECK: npuvector.transfer_write {{.*}} : !npuvector<47x32xbf16>, memref<32x512x32xbf16>
// CHECK: {{^        }}}
// CHECK-NOT: scf.if
// CHECK: %[[E_TAIL_WIDTH:.*]] = arith.constant 21 : index
// CHECK: {{^        }}scf.for %[[E_TAIL_ROW:.*]] = %[[E_BODY_END]] to %[[E_HIGH]] step %[[E_TAIL_WIDTH]] {
// CHECK-NOT: scf.if
// CHECK-NOT: !npuvector<{{[^>]*[?]}}
// CHECK: npuvector.transfer_read %{{.*}}[%[[E_HEAD]], %[[E_TAIL_ROW]], %{{.*}}]{{.*}} : {{.*}}, !npuvector<21x32xbf16>
// CHECK-NOT: !npuvector<{{[^>]*[?]}}
// CHECK: arith.mulf {{.*}} : !npuvector<21x32xf32>
// CHECK-NOT: !npuvector<{{[^>]*[?]}}
// CHECK: npuvector.transfer_write {{.*}}, %{{.*}}[%[[E_HEAD]], %[[E_TAIL_ROW]], %{{.*}}] : !npuvector<21x32xbf16>, memref<32x512x32xbf16>
// CHECK: {{^        }}}
// CHECK-NEXT: {{^      }}}
// CHECK-NEXT: {{^    }}}
// CHECK: return
func.func @test_piecewise_clipped_elementwise_64blocks(
    %input: memref<32x512x32xbf16, strided<[65536, 128, 2], offset: 1>>,
    %output: memref<32x512x32xbf16>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %c32 = arith.constant 32 : index
  scf.forall (%block) in (64) {
    %head = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 32)>(%block)
    scf.for %tile = %c0 to %c6 step %c1 {
      %low = affine.min affine_map<(d0, d1) -> (d0 * 47 + d1 * 256 - (d1 floordiv 2) * 512, 512)>(%tile, %block)
      %high = affine.min affine_map<(d0, d1) -> (512, d1 * 256 - (d1 floordiv 2) * 512 + 256, d0 * 47 + d1 * 256 - (d1 floordiv 2) * 512 + 47)>(%tile, %block)
      scf.for %row = %low to %high step %c1 {
        scf.for %column = %c0 to %c32 step %c1 {
          %value = memref.load %input[%head, %row, %column] : memref<32x512x32xbf16, strided<[65536, 128, 2], offset: 1>>
          %wide = arith.extf %value : bf16 to f32
          %square = arith.mulf %wide, %wide : f32
          %result = arith.truncf %square : f32 to bf16
          memref.store %result, %output[%head, %row, %column] : memref<32x512x32xbf16>
        } {vector = 32 : i64}
      } {vector = 47 : i64}
    }
  } {mapping = [#hivm.block]}
  return
}

// -----

// Each 22-row tile has 16/6 rows; the final tile has 6/0. The clipped lower
// bound permits an empty tile, and the inactive 512 upper cap must be ignored.
// CHECK-LABEL: func.func @test_piecewise_clipped_lower_bound_empty(
// CHECK-DAG: %[[C_ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C_ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C_TWO:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C_TWENTY_FOUR:.*]] = arith.constant 24 : index
// CHECK: scf.for %[[C_TILE:.*]] = %[[C_ZERO]] to %[[C_TWENTY_FOUR]] step %[[C_ONE]] {
// CHECK: scf.for %[[C_PART:.*]] = %[[C_ZERO]] to %[[C_TWO]] step %[[C_ONE]] {
// CHECK: %[[C_LOW:.*]] = affine.min
// CHECK: %[[C_HIGH:.*]] = affine.min
// CHECK: %[[C_WIDTH:.*]] = arith.constant 16 : index
// CHECK: scf.for %[[C_ROW:.*]] = %[[C_LOW]] to %[[C_BODY_END:.*]] step %[[C_WIDTH]] {
// CHECK-NOT: scf.if
// CHECK-NOT: !npuvector<{{[^>]*[?]}}
// CHECK: %[[C_BODY:.*]] = npuvector.transfer_read %{{.*}}[%[[C_ROW]]]{{.*}} : memref<512xf32>, !npuvector<16xf32>
// CHECK: %[[C_BODY_SUM:.*]] = arith.addf %[[C_BODY]], %[[C_BODY]] : !npuvector<16xf32>
// CHECK: npuvector.transfer_write %[[C_BODY_SUM]], %{{.*}}[%[[C_ROW]]] : !npuvector<16xf32>, memref<512xf32>
// CHECK: {{^        }}}
// CHECK-NOT: scf.if
// CHECK: %[[C_TAIL_WIDTH:.*]] = arith.constant 6 : index
// CHECK: scf.for %[[C_TAIL_ROW:.*]] = %[[C_BODY_END]] to %[[C_HIGH]] step %[[C_TAIL_WIDTH]] {
// CHECK-NOT: scf.if
// CHECK-NOT: !npuvector<{{[^>]*[?]}}
// CHECK: %[[C_TAIL:.*]] = npuvector.transfer_read %{{.*}}[%[[C_TAIL_ROW]]]{{.*}} : memref<512xf32>, !npuvector<6xf32>
// CHECK: %[[C_TAIL_SUM:.*]] = arith.addf %[[C_TAIL]], %[[C_TAIL]] : !npuvector<6xf32>
// CHECK: npuvector.transfer_write %[[C_TAIL_SUM]], %{{.*}}[%[[C_TAIL_ROW]]] : !npuvector<6xf32>, memref<512xf32>
// CHECK: {{^        }}}
// CHECK-NEXT: {{^      }}}
// CHECK-NEXT: {{^    }}}
// CHECK: return
func.func @test_piecewise_clipped_lower_bound_empty(%input: memref<512xf32>, %output: memref<512xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c24 = arith.constant 24 : index
  scf.for %tile = %c0 to %c24 step %c1 {
    scf.for %part = %c0 to %c2 step %c1 {
      %low = affine.min affine_map<(d0, d1) -> (d0 * 22 + d1 * 16, 512)>(%tile, %part)
      %high = affine.min affine_map<(d0, d1) -> (512, d0 * 22 + 22, d0 * 22 + d1 * 16 + 16)>(%tile, %part)
      scf.for %row = %low to %high step %c1 {
        %value = memref.load %input[%row] : memref<512xf32>
        %result = arith.addf %value, %value : f32
        memref.store %result, %output[%row] : memref<512xf32>
      } {vector = 16 : i64}
    }
  }
  return
}

// -----

// A five-row base offset must survive both segments. For each (block, tile),
// update one full accumulator before reducing and adding the original seven.
// CHECK-LABEL: func.func @test_piecewise_clipped_reduction_nonzero_init(
// CHECK-DAG: %[[R_ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[R_ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[R_SIX:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[R_INIT:.*]] = arith.constant 7.000000e+00 : f32
// CHECK: scf.forall (%[[R_BLOCK:.*]]) in (64) {
// CHECK: scf.for %[[R_TILE:.*]] = %[[R_ZERO]] to %[[R_SIX]] step %[[R_ONE]] {
// CHECK: %[[R_LOW:.*]] = affine.min
// CHECK: %[[R_HIGH:.*]] = affine.min
// CHECK: %[[R_WIDTH:.*]] = arith.constant 47 : index
// CHECK: %[[R_NEUTRAL_SCALAR:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[R_NEUTRAL:.*]] = npuvector.broadcast %[[R_NEUTRAL_SCALAR]]{{.*}} : f32 to !npuvector<47xf32>
// CHECK: %[[R_BODY_END:.*]] = affine.apply
// CHECK: %[[R_BODY:.*]] = scf.for %{{.*}} = %[[R_LOW]] to %[[R_BODY_END]] step %{{.*}} iter_args(%[[R_BODY_ACC:.*]] = %[[R_NEUTRAL]]) -> (!npuvector<47xf32>) {
// CHECK-NOT: scf.if
// CHECK-NOT: !npuvector<{{[^>]*[?]}}
// CHECK: %[[R_BODY_VALUE:.*]] = npuvector.transfer_read {{.*}} : memref<32x522xf32>, !npuvector<47xf32>
// CHECK: %[[R_BODY_NEXT:.*]] = arith.addf %[[R_BODY_ACC]], %[[R_BODY_VALUE]] {{.*}} : !npuvector<47xf32>
// CHECK-NEXT: scf.yield %[[R_BODY_NEXT]] : !npuvector<47xf32>
// CHECK-NOT: npuvector.reduction
// CHECK-NOT: scf.if
// CHECK: %[[R_TAIL_WIDTH:.*]] = arith.constant 21 : index
// CHECK: %[[R_TAIL:.*]] = scf.for %[[R_TAIL_ROW:.*]] = %[[R_BODY_END]] to %[[R_HIGH]] step %[[R_TAIL_WIDTH]] iter_args(%[[R_TAIL_ACC:.*]] = %[[R_BODY]]) -> (!npuvector<47xf32>) {
// CHECK-NOT: scf.if
// CHECK-NOT: !npuvector<{{[^>]*[?]}}
// CHECK: %[[R_SLICE:.*]] = npuvector.extract_slice %[[R_TAIL_ACC]][%[[R_OFFSET:.*]]] [%[[R_TAIL_WIDTH]]] [%[[R_STRIDE:.*]]] {keep_dims = array<i64: 0>} : !npuvector<47xf32> to !npuvector<21xf32>
// CHECK: %[[R_TAIL_VALUE:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}, %[[R_TAIL_ROW]]]{{.*}} : memref<32x522xf32>, !npuvector<21xf32>
// CHECK: %[[R_TAIL_NEXT:.*]] = arith.addf %[[R_SLICE]], %[[R_TAIL_VALUE]] {{.*}} : !npuvector<21xf32>
// CHECK: %[[R_INSERT:.*]] = npuvector.insert_slice %[[R_TAIL_NEXT]] into %[[R_TAIL_ACC]][%[[R_OFFSET]]] [%[[R_TAIL_WIDTH]]] [%[[R_STRIDE]]] : !npuvector<21xf32> into !npuvector<47xf32>
// CHECK-NEXT: scf.yield %[[R_INSERT]] : !npuvector<47xf32>
// CHECK-NOT: npuvector.reduction
// CHECK: %[[R_REDUCED:.*]] = npuvector.reduction <add>, %[[R_TAIL]] {{.*}} : !npuvector<47xf32> into !npuvector.f32
// CHECK-NEXT: %[[R_INIT_VEC:.*]] = npuvector.broadcast %[[R_INIT]] : f32 to !npuvector.f32
// CHECK-NEXT: %[[R_RESULT:.*]] = arith.addf %[[R_REDUCED]], %[[R_INIT_VEC]] : !npuvector.f32
// CHECK-NEXT: npuvector.transfer_write %[[R_RESULT]], %{{.*}}[%[[R_BLOCK]], %[[R_TILE]]] : !npuvector.f32, memref<64x6xf32>
// CHECK: {{^      }}}
// CHECK-NEXT: {{^    }}}
// CHECK: return
func.func @test_piecewise_clipped_reduction_nonzero_init(
    %input: memref<32x522xf32>, %output: memref<64x6xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %init = arith.constant 7.0 : f32
  scf.forall (%block) in (64) {
    %head = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 32)>(%block)
    scf.for %tile = %c0 to %c6 step %c1 {
      %low = affine.min affine_map<(d0, d1) -> (d0 * 47 + d1 * 256 - (d1 floordiv 2) * 512 + 5, 517)>(%tile, %block)
      %high = affine.min affine_map<(d0, d1) -> (517, d1 * 256 - (d1 floordiv 2) * 512 + 261, d0 * 47 + d1 * 256 - (d1 floordiv 2) * 512 + 52)>(%tile, %block)
      %sum = scf.for %row = %low to %high step %c1 iter_args(%acc = %init) -> f32 {
        %value = memref.load %input[%head, %row] : memref<32x522xf32>
        %next = arith.addf %acc, %value {reduction_type = "x"} : f32
        scf.yield %next : f32
      } {reduction_x = 47 : i64}
      memref.store %sum, %output[%block, %tile] : memref<64x6xf32>
    }
  } {mapping = [#hivm.block]}
  return
}

// -----

// A non-affine condition keeps the original point loop scalar on fallback.
// CHECK-LABEL: func.func @test_piecewise_scalar_if_fallback(
// CHECK: %[[F_ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[F_ONE:.*]] = arith.constant 1 : index
// CHECK: %[[F_INIT:.*]] = arith.constant 7.000000e+00 : f32
// CHECK: scf.for
// CHECK: %[[F_LOW:.*]] = affine.apply
// CHECK: %[[F_HIGH:.*]] = affine.min
// CHECK-NOT: scf.for
// CHECK: %[[F_SUM:.*]] = scf.for %{{.*}} = %[[F_LOW]] to %[[F_HIGH]] step %[[F_ONE]] iter_args(%[[F_ACC:.*]] = %[[F_INIT]]) -> (f32) {
// CHECK: %[[F_PARITY:.*]] = arith.remui {{.*}} : index
// CHECK: %[[F_EVEN:.*]] = arith.cmpi eq, %[[F_PARITY]], {{.*}} : index
// CHECK: %[[F_CHOSEN:.*]] = scf.if %[[F_EVEN]] -> (f32) {
// CHECK: scf.yield {{.*}} : f32
// CHECK: } else {
// CHECK: scf.yield {{.*}} : f32
// CHECK: %[[F_NEXT:.*]] = arith.addf %[[F_ACC]], %[[F_CHOSEN]] {reduction_type = "x"} : f32
// CHECK-NEXT: scf.yield %[[F_NEXT]] : f32
// CHECK-NOT: scf.for
// CHECK: memref.store %[[F_SUM]], {{.*}} : memref<2xf32>
// CHECK: return
func.func @test_piecewise_scalar_if_fallback(%output: memref<2xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %init = arith.constant 7.0 : f32
  %zero = arith.constant 0.0 : f32
  scf.for %tile = %c0 to %c2 step %c1 {
    %low = affine.apply affine_map<(d0) -> (d0 * 4)>(%tile)
    %high = affine.min affine_map<(d0) -> (d0 + 4, 6)>(%low)
    %sum = scf.for %i = %low to %high step %c1 iter_args(%acc = %init) -> f32 {
      %value = arith.addf %init, %zero : f32
      %parity = arith.remui %i, %c2 : index
      %even = arith.cmpi eq, %parity, %c0 : index
      %chosen = scf.if %even -> (f32) {
        scf.yield %value : f32
      } else {
        scf.yield %zero : f32
      }
      %next = arith.addf %acc, %chosen {reduction_type = "x"} : f32
      scf.yield %next : f32
    } {reduction_x = 4 : i64}
    memref.store %sum, %output[%tile] : memref<2xf32>
  }
  return
}

// -----

// A single static tail cannot represent residual lengths 45, 35, 25 and 15.
// Keep the existing dynamic-vector path instead of selecting an arbitrary tail.
// CHECK-LABEL: func.func @test_piecewise_multiple_residuals_stays_dynamic(
// CHECK: scf.forall
// CHECK: scf.for
// CHECK-NOT: scf.if
// CHECK: npuvector.transfer_read {{.*}} : memref<32x512xf32>, !npuvector<?xf32>
// CHECK-NOT: scf.if
// CHECK: npuvector.transfer_write {{.*}} : !npuvector<?xf32>, memref<32x512xf32>
// CHECK-NOT: scf.if
// CHECK: return
func.func @test_piecewise_multiple_residuals_stays_dynamic(
    %input: memref<32x512xf32>, %output: memref<32x512xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.forall (%block) in (64) {
    %head = affine.apply affine_map<(d0) -> ((d0 floordiv 2) mod 32)>(%block)
    scf.for %tile = %c0 to %c6 step %c1 {
      %low = affine.apply affine_map<(d0, d1) -> (d0 * 47 + d1 * 256 - (d1 floordiv 2) * 512)>(%tile, %block)
      %high = affine.min affine_map<(d0, d1) -> (d0 * 47 + d1 * 256 - (d1 floordiv 2) * 512 + 47, d0 * 37 + d1 * 256 - (d1 floordiv 2) * 512 + 65, d1 * 256 - (d1 floordiv 2) * 512 + 256)>(%tile, %block)
      scf.for %row = %low to %high step %c1 {
        %value = memref.load %input[%head, %row] : memref<32x512xf32>
        %result = arith.addf %value, %value : f32
        memref.store %result, %output[%head, %row] : memref<32x512xf32>
      } {vector = 47 : i64}
    }
  } {mapping = [#hivm.block]}
  return
}

// -----

// The untagged outer loop stays scalar. Its inner constant range always has
// two full tiles and one tail, so the tail is emitted directly after the body.
// CHECK-LABEL: func.func @elementwise_body_tail_under_scalar_outer_2x65(
// CHECK-DAG: %[[SO_ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[SO_ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[SO_TWO:.*]] = arith.constant 2 : index
// CHECK: scf.for %[[SO_ROW:.*]] = %[[SO_ZERO]] to %[[SO_TWO]] step %[[SO_ONE]] {
// CHECK-DAG: %[[SO_SPLIT:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[SO_WIDTH:.*]] = arith.constant 32 : index
// CHECK: scf.for %[[SO_COLUMN:.*]] = %[[SO_ZERO]] to %[[SO_SPLIT]] step %[[SO_WIDTH]] {
// CHECK: %[[SO_BODY:.*]] = npuvector.transfer_read %{{.*}}[%[[SO_ROW]], %[[SO_COLUMN]]]{{.*}} : memref<2x65xf32>, !npuvector<32xf32>
// CHECK: %[[SO_BODY_SUM:.*]] = arith.addf %[[SO_BODY]], %[[SO_BODY]] : !npuvector<32xf32>
// CHECK: npuvector.transfer_write %[[SO_BODY_SUM]], %{{.*}}[%[[SO_ROW]], %[[SO_COLUMN]]] : !npuvector<32xf32>, memref<2x65xf32>
// CHECK: {{^      }}}
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK: %[[SO_TAIL:.*]] = npuvector.transfer_read %{{.*}}[%[[SO_ROW]], %[[SO_SPLIT]]]{{.*}} : memref<2x65xf32>, !npuvector<1xf32>
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK: %[[SO_TAIL_SUM:.*]] = arith.addf %[[SO_TAIL]], %[[SO_TAIL]] : !npuvector<1xf32>
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK: npuvector.transfer_write %[[SO_TAIL_SUM]], %{{.*}}[%[[SO_ROW]], %[[SO_SPLIT]]] : !npuvector<1xf32>, memref<2x65xf32>
// CHECK-NEXT: {{^    }}}
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK: return
func.func @elementwise_body_tail_under_scalar_outer_2x65(
    %input: memref<2x65xf32>, %output: memref<2x65xf32>)
    attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c65 = arith.constant 65 : index
  scf.for %row = %c0 to %c2 step %c1 {
    scf.for %column = %c0 to %c65 step %c1 {
      %value = memref.load %input[%row, %column] : memref<2x65xf32>
      %result = arith.addf %value, %value : f32
      memref.store %result, %output[%row, %column] : memref<2x65xf32>
    } {vector = 32 : i64}
  }
  return
}
