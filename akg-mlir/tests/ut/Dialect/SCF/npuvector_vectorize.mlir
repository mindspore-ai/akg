// RUN: akg-opt %s --npuvector-vectorize --mlir-disable-threading --allow-unregistered-dialect | FileCheck %s

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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<1024xf32>, memref<1024xf32>
  // CHECK-NOT: scf.for

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
  // CHECK: %[[TRIP:.*]] = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[VEC_SIZE:.*]] = arith.minsi %[[TRIP]], %[[MAX]] : index
  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VEC_SIZE]]] [%[[MAX]]], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK: %[[SCALE_VEC:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
  // CHECK: %[[SCALED_VEC:.*]] = arith.mulf %[[V_VEC]], %[[SCALE_VEC]] : !npuvector<?xf32>
  // CHECK: %[[BIAS_VEC:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
  // CHECK: %[[RESULT_VEC:.*]] = arith.addf %[[SCALED_VEC]], %[[BIAS_VEC]] : !npuvector<?xf32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<?xf32>, memref<?xf32>
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
  // CHECK: %[[NEUTRAL:.*]] = arith.constant dense<0.000000e+00> : !npuvector<128xf32>
  // CHECK: %[[SUM_VEC:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC_VEC:.*]] = %[[NEUTRAL]]) -> (!npuvector<128xf32>) {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VF]]], %{{.*}} : memref<1024xf32>, !npuvector<128xf32>
  // CHECK:   %[[NEW_ACC_VEC:.*]] = arith.addf %[[ACC_VEC]], %[[V_VEC]] {reduction_type = "all"} : !npuvector<128xf32>
  // CHECK:   scf.yield %[[NEW_ACC_VEC]] : !npuvector<128xf32>
  // CHECK: }
  // CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %[[SUM_VEC]] : !npuvector<128xf32> into f32
  // CHECK: memref.store %[[SUM]], %{{.*}}[] : memref<f32>

  %sum = scf.for %i = %c0 to %c1024 step %c1 iter_args(%acc = %init) -> f32 {
    %v = memref.load %input[%i] : memref<1024xf32>
    %new_acc = arith.addf %acc, %v {reduction_type = "all"} : f32
    scf.yield %new_acc : f32
  } {reduction_all=128}

  memref.store %sum, %output[] : memref<f32>
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
  // CHECK: %[[TRIP:.*]] = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[VEC_SIZE:.*]] = arith.minsi %[[TRIP]], %[[VF]] : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[NEUTRAL:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[VF]]] : f32 to !npuvector<?xf32>
  // CHECK: %{{.*}} = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[DIV:.*]] = arith.divsi %{{.*}}, %[[VEC_SIZE]] : index
  // CHECK: %[[MUL:.*]] = arith.muli %[[DIV]], %[[VEC_SIZE]] : index
  // CHECK: %[[ALIGNED_UB:.*]] = arith.addi %[[MUL]], %{{.*}} : index
  // CHECK: %[[SUM_VEC:.*]] = scf.for %{{.*}} = %{{.*}} to %[[ALIGNED_UB]] step %{{.*}} iter_args(%[[ACC_VEC:.*]] = %[[NEUTRAL]]) -> (!npuvector<?xf32>) {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%{{.*}}] [%{{.*}}], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   %[[NEW_ACC_VEC:.*]] = arith.addf %[[ACC_VEC]], %[[V_VEC]] {reduction_type = "all"} : !npuvector<?xf32>
  // CHECK:   scf.yield %[[NEW_ACC_VEC]] : !npuvector<?xf32>
  // CHECK: }
  // CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %[[SUM_VEC]] : !npuvector<?xf32> into f32
  // CHECK: %{{.*}} = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[REMAINDER:.*]] = arith.remsi %{{.*}}, %[[VEC_SIZE]] : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[NEED_TAIL:.*]] = arith.cmpi ne, %[[REMAINDER]], %[[C0]] : index
  // CHECK: %[[FINAL:.*]] = scf.if %[[NEED_TAIL]] -> (f32) {
  // CHECK:   %[[TAIL_READ:.*]] = npuvector.transfer_read %{{.*}}[%[[ALIGNED_UB]]] [%[[REMAINDER]]] [%{{.*}}], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   %[[TAIL_REDUCED:.*]] = npuvector.reduction <add>, %[[TAIL_READ]] : !npuvector<?xf32> into f32
  // CHECK:   %[[MERGED:.*]] = arith.addf %[[SUM]], %[[TAIL_REDUCED]] : f32
  // CHECK:   scf.yield %[[MERGED]] : f32
  // CHECK: } else {
  // CHECK:   scf.yield %[[SUM]] : f32
  // CHECK: }
  // CHECK: memref.store %[[FINAL]], %{{.*}}[] : memref<f32>

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
    %out_sq: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32

  // CHECK: %[[VF:.*]] = arith.constant 128 : index
  // CHECK: %[[N0:.*]] = arith.constant dense<0.000000e+00> : !npuvector<128xf32>
  // CHECK: %[[N1:.*]] = arith.constant dense<0.000000e+00> : !npuvector<128xf32>
  // CHECK: %{{.*}}:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC0:.*]] = %[[N0]], %[[ACC1:.*]] = %[[N1]]) -> (!npuvector<128xf32>, !npuvector<128xf32>) {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VF]]], %{{.*}} : memref<1024xf32>, !npuvector<128xf32>
  // CHECK:   %[[M_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VF]]], %{{.*}} : memref<1024xf32>, !npuvector<128xf32>
  // CHECK:   %[[VM:.*]] = arith.mulf %[[V_VEC]], %[[M_VEC]] : !npuvector<128xf32>
  // CHECK:   %[[VV:.*]] = arith.mulf %[[V_VEC]], %[[V_VEC]] : !npuvector<128xf32>
  // CHECK:   %[[A0:.*]] = arith.addf %[[ACC0]], %[[VM]] {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<128xf32>
  // CHECK:   %[[A1:.*]] = arith.addf %[[ACC1]], %[[VV]] {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<128xf32>
  // CHECK:   scf.yield %[[A0]], %[[A1]] : !npuvector<128xf32>, !npuvector<128xf32>
  // CHECK: }
  // CHECK: %[[R0:.*]] = npuvector.reduction <add>, %{{.*}} : !npuvector<128xf32> into f32
  // CHECK: %[[R1:.*]] = npuvector.reduction <add>, %{{.*}} : !npuvector<128xf32> into f32
  // CHECK: memref.store %[[R0]], %{{.*}}[] : memref<f32>
  // CHECK: memref.store %[[R1]], %{{.*}}[] : memref<f32>

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
// Two reduction_x lanes; static trip 300, VF 128 -> main 256 + tail 44 per lane (same remainder for both).

// CHECK-LABEL: func.func @test_reduction_x_two_results_static_tail
func.func @test_reduction_x_two_results_static_tail(
    %input: memref<300xf32>,
    %mask: memref<300xf32>,
    %out_sum: memref<f32>,
    %out_sq: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c300 = arith.constant 300 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32

  // CHECK: %[[VF:.*]] = arith.constant 128 : index
  // CHECK: %[[N0:.*]] = arith.constant dense<0.000000e+00> : !npuvector<128xf32>
  // CHECK: %[[N1:.*]] = arith.constant dense<0.000000e+00> : !npuvector<128xf32>
  // CHECK: %[[ALIGNED_UB:.*]] = arith.constant 256 : index
  // CHECK: %{{.*}}:2 = scf.for %{{.*}} = %{{.*}} to %[[ALIGNED_UB]] step %{{.*}} iter_args(%[[ACC0:.*]] = %[[N0]], %[[ACC1:.*]] = %[[N1]]) -> (!npuvector<128xf32>, !npuvector<128xf32>) {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%{{.*}}] [%[[VF]]], %{{.*}} : memref<300xf32>, !npuvector<128xf32>
  // CHECK:   %[[M_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%{{.*}}] [%[[VF]]], %{{.*}} : memref<300xf32>, !npuvector<128xf32>
  // CHECK:   arith.mulf %[[V_VEC]], %[[M_VEC]] : !npuvector<128xf32>
  // CHECK:   arith.mulf %[[V_VEC]], %[[V_VEC]] : !npuvector<128xf32>
  // CHECK:   arith.addf %[[ACC0]], {{.*}} {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<128xf32>
  // CHECK:   arith.addf %[[ACC1]], {{.*}} {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<128xf32>
  // CHECK:   scf.yield {{.*}} : !npuvector<128xf32>, !npuvector<128xf32>
  // CHECK: }
  // CHECK: npuvector.reduction <add>, %{{.*}} : !npuvector<128xf32> into f32
  // CHECK: arith.constant 44 : index
  // CHECK: npuvector.transfer_read {{.*}} : memref<300xf32>, !npuvector<44xf32>
  // CHECK: npuvector.transfer_read {{.*}} : memref<300xf32>, !npuvector<44xf32>
  // CHECK: npuvector.reduction <add>, %{{.*}} : !npuvector<44xf32> into f32
  // CHECK: arith.addf {{.*}} : f32
  // CHECK: npuvector.reduction <add>, %{{.*}} : !npuvector<128xf32> into f32
  // CHECK: arith.constant 44 : index
  // CHECK: npuvector.transfer_read {{.*}} : memref<300xf32>, !npuvector<44xf32>
  // CHECK: npuvector.transfer_read {{.*}} : memref<300xf32>, !npuvector<44xf32>
  // CHECK: npuvector.reduction <add>, %{{.*}} : !npuvector<44xf32> into f32
  // CHECK: arith.addf {{.*}} : f32
  // CHECK: memref.store {{.*}}[] : memref<f32>
  // CHECK: memref.store {{.*}}[] : memref<f32>

  %pair:2 = scf.for %i = %c0 to %c300 step %c1 iter_args(%a0 = %init, %a1 = %init) -> (f32, f32) {
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
// Two reduction_x lanes; dynamic dim — aligned main loop + optional tail each lane (may duplicate index math IR).

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

  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[VF:.*]] = arith.constant 128 : index
  // CHECK: %[[TRIP:.*]] = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[VEC_SIZE:.*]] = arith.minsi %[[TRIP]], %[[VF]] : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[N0:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[VF]]] : f32 to !npuvector<?xf32>
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[N1:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[VF]]] : f32 to !npuvector<?xf32>
  // CHECK: %{{.*}} = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[DIV:.*]] = arith.divsi %{{.*}}, %[[VEC_SIZE]] : index
  // CHECK: %[[MUL:.*]] = arith.muli %[[DIV]], %[[VEC_SIZE]] : index
  // CHECK: %[[ALIGNED_UB:.*]] = arith.addi %[[MUL]], %{{.*}} : index
  // CHECK: %{{.*}}:2 = scf.for %{{.*}} = %{{.*}} to %[[ALIGNED_UB]] step %{{.*}} iter_args(%{{.*}} = %[[N0]], %{{.*}} = %[[N1]]) -> (!npuvector<?xf32>, !npuvector<?xf32>) {
  // CHECK:   npuvector.transfer_read {{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   npuvector.transfer_read {{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   arith.mulf {{.*}} : !npuvector<?xf32>
  // CHECK:   arith.mulf {{.*}} : !npuvector<?xf32>
  // CHECK:   arith.addf {{.*}} {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<?xf32>
  // CHECK:   arith.addf {{.*}} {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<?xf32>
  // CHECK:   scf.yield {{.*}} : !npuvector<?xf32>, !npuvector<?xf32>
  // CHECK: }
  // CHECK: npuvector.reduction <add>, %{{.*}} : !npuvector<?xf32> into f32
  // CHECK: %{{.*}} = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: arith.remsi %{{.*}}, %[[VEC_SIZE]] : index
  // CHECK: scf.if %{{.*}} -> (f32) {
  // CHECK:   %{{.*}} = npuvector.transfer_read %{{.*}}[%[[ALIGNED_UB]]] [%{{.*}}] [%{{.*}}], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   %{{.*}} = npuvector.transfer_read %{{.*}}[%[[ALIGNED_UB]]] [%{{.*}}] [%{{.*}}], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   arith.mulf {{.*}} : !npuvector<?xf32>
  // CHECK:   arith.mulf {{.*}} : !npuvector<?xf32>
  // CHECK:   npuvector.reduction <add>, {{.*}} : !npuvector<?xf32> into f32
  // CHECK:   arith.addf {{.*}} : f32
  // CHECK:   scf.yield {{.*}} : f32
  // CHECK: } else {
  // CHECK:   scf.yield {{.*}} : f32
  // CHECK: }
  // CHECK: npuvector.reduction <add>, %{{.*}} : !npuvector<?xf32> into f32
  // CHECK: %{{.*}} = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: arith.remsi %{{.*}}, %[[VEC_SIZE]] : index
  // CHECK: scf.if %{{.*}} -> (f32) {
  // CHECK:   %{{.*}} = npuvector.transfer_read %{{.*}}[%[[ALIGNED_UB]]] [%{{.*}}] [%{{.*}}], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   %{{.*}} = npuvector.transfer_read %{{.*}}[%[[ALIGNED_UB]]] [%{{.*}}] [%{{.*}}], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   npuvector.reduction <add>, {{.*}} : !npuvector<?xf32> into f32
  // CHECK:   arith.addf {{.*}} : f32
  // CHECK:   scf.yield {{.*}} : f32
  // CHECK: } else {
  // CHECK:   scf.yield {{.*}} : f32
  // CHECK: }
  // CHECK: memref.store {{.*}}[] : memref<f32>
  // CHECK: memref.store {{.*}}[] : memref<f32>

  %pair:2 = scf.for %i = %c0 to %N step %c1 iter_args(%a0 = %init, %a1 = %init) -> (f32, f32) {
    %v = memref.load %input[%i] : memref<?xf32>
    %m = memref.load %mask[%i] : memref<?xf32>
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

// ============================================================================
// 3. 1-D tail handling static and dynamic
// Static: 300 elements, vf=128 -> 256+44 tail; dynamic tail see test_reduction_dynamic (section 2)
// ============================================================================

// CHECK-LABEL: func.func @test_reduction_static_with_tail
func.func @test_reduction_static_with_tail(%input: memref<300xf32>, %output: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c300 = arith.constant 300 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32

  // CHECK: %[[VF:.*]] = arith.constant 128 : index
  // CHECK: %[[NEUTRAL:.*]] = arith.constant dense<0.000000e+00> : !npuvector<128xf32>
  // CHECK: %[[ALIGNED_UB:.*]] = arith.constant 256 : index
  // CHECK: %[[SUM_VEC:.*]] = scf.for %{{.*}} = %{{.*}} to %[[ALIGNED_UB]] step %{{.*}} iter_args(%[[ACC_VEC:.*]] = %[[NEUTRAL]]) -> (!npuvector<128xf32>) {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%{{.*}}] [%[[VF]]], %{{.*}} : memref<300xf32>, !npuvector<128xf32>
  // CHECK:   %[[NEW_ACC_VEC:.*]] = arith.addf %[[ACC_VEC]], %[[V_VEC]] {reduction_type = "all"} : !npuvector<128xf32>
  // CHECK:   scf.yield %[[NEW_ACC_VEC]] : !npuvector<128xf32>
  // CHECK: }
  // CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %[[SUM_VEC]] : !npuvector<128xf32> into f32
  // CHECK: %[[TAIL_SIZE:.*]] = arith.constant 44 : index
  // CHECK: %[[TAIL_READ:.*]] = npuvector.transfer_read %{{.*}}[%[[ALIGNED_UB]]], %{{.*}} : memref<300xf32>, !npuvector<44xf32>
  // CHECK: %[[TAIL_REDUCED:.*]] = npuvector.reduction <add>, %[[TAIL_READ]] : !npuvector<44xf32> into f32
  // CHECK: %[[FINAL:.*]] = arith.addf %[[SUM]], %[[TAIL_REDUCED]] : f32
  // CHECK: memref.store %[[FINAL]], %{{.*}}[] : memref<f32>

  %sum = scf.for %i = %c0 to %c300 step %c1 iter_args(%acc = %init) -> f32 {
    %v = memref.load %input[%i] : memref<300xf32>
    %new_acc = arith.addf %acc, %v {reduction_type = "all"} : f32
    scf.yield %new_acc : f32
  } {reduction_all=128}

  memref.store %sum, %output[] : memref<f32>
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

  // CHECK: arith.constant dense<0.000000e+00> : !npuvector<64xf32>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!npuvector<64xf32>) {
  // CHECK:   npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}] [%{{.*}}], %{{.*}} : memref<32x64xf32>, !npuvector<64xf32>
  // CHECK:   arith.addf %{{.*}}, %{{.*}} {reduction_axes = [0 : index], reduction_type = "y"} : !npuvector<64xf32>
  // CHECK:   scf.yield %{{.*}} : !npuvector<64xf32>
  // CHECK: }
  // CHECK: npuvector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] : !npuvector<64xf32>, memref<64xf32>
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
  // CHECK: arith.subi %{{.*}}, %{{.*}} : index
  // CHECK: arith.minsi %{{.*}}, %{{.*}} : index
  // CHECK: npuvector.broadcast %{{.*}}[%{{.*}}] [%{{.*}}] : f32 to !npuvector<?xf32>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!npuvector<?xf32>) {
  // CHECK:   npuvector.transfer_read %{{.*}}[%{{.*}}, %{{.*}}] [%{{.*}}] [%{{.*}}], %{{.*}} : memref<?x?xf32>, !npuvector<?xf32>
  // CHECK:   arith.addf %{{.*}}, %{{.*}} {reduction_axes = [0 : index], reduction_type = "y"} : !npuvector<?xf32>
  // CHECK:   scf.yield %{{.*}} : !npuvector<?xf32>
  // CHECK: }
  // CHECK: npuvector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] : !npuvector<?xf32>, memref<?xf32>
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
  // CHECK: npuvector.transfer_write %[[C_VEC]], %{{.*}}[%{{.*}}, %{{.*}}] : !npuvector<32x64xf32>, memref<32x64xf32>
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
  // CHECK: npuvector.transfer_write %[[C_VEC]], %{{.*}}[%{{.*}}, %{{.*}}] : !npuvector<32x64xf32>, memref<32x64xf32>
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
  // CHECK: arith.subi
  // CHECK: arith.minsi
  // CHECK: arith.constant 4096
  // CHECK: arith.subi
  // CHECK: arith.minsi
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
  // CHECK: npuvector.transfer_write %[[C_VEC]], %{{.*}}[%{{.*}}, %{{.*}}] : !npuvector<32x64xf32>, memref<32x64xf32>
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
  // CHECK: npuvector.transfer_write %[[TRANS]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : !npuvector<32x16x8xf32>, memref<8x16x32xf32>
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

// CHECK-LABEL: func.func @test_multidim_transpose_dynamic
func.func @test_multidim_transpose_dynamic(
    %a: memref<?x?xf32>, %out: memref<?x?xf32>, %M: index, %N: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: arith.constant 4096
  // CHECK: arith.subi
  // CHECK: arith.minsi
  // CHECK: arith.constant 4096
  // CHECK: arith.subi
  // CHECK: arith.minsi
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
  // CHECK: arith.subi
  // CHECK: arith.minsi
  // CHECK: arith.constant 4096
  // CHECK: arith.subi
  // CHECK: arith.minsi
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
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.extf %[[V_VEC]] : !npuvector<128xbf16> to !npuvector<128xf32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xf32>, memref<128xf32>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xbf16>
    %result = arith.extf %v : bf16 to f32
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
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.truncf %[[V_VEC]] : !npuvector<128xf32> to !npuvector<128xbf16>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xbf16>, memref<128xbf16>
  // CHECK-NOT: scf.for

  scf.for %i = %c0 to %c128 step %c1 {
    %v = memref.load %input[%i] : memref<128xf32>
    %result = arith.truncf %v : f32 to bf16
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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xi32>, memref<128xi32>
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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xi32>, memref<128xi32>
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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xi8>, memref<128xi8>
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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xf32>, memref<128xf32>
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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xf32>, memref<128xf32>
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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xi32>, memref<128xi32>
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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xi32>, memref<128xi32>
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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xi16>, memref<128xi16>
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
  // CHECK: npuvector.transfer_write %[[TRUNC_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xbf16>, memref<128xbf16>
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
  // CHECK: %[[TRIP:.*]] = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[VEC_SIZE:.*]] = arith.minsi %[[TRIP]], %[[MAX]] : index
  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VEC_SIZE]]] [%[[MAX]]], %{{.*}} : memref<?xbf16>, !npuvector<?xbf16>
  // CHECK: %[[RESULT_VEC:.*]] = npuvector.extf %[[V_VEC]] : !npuvector<?xbf16> to !npuvector<?xf32>
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<?xf32>, memref<?xf32>
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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xi1>, memref<128xi1>
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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xi1>, memref<128xi1>
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
  // CHECK: npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xf32>, memref<128xf32>
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
  // CHECK:   npuvector.transfer_write %[[SCALED_VEC]], %{{.*}}[%{{.*}}] : !npuvector<128xf32>, memref<128xf32>
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

  // CHECK: arith.constant 118 : index
  // CHECK: arith.cmpi ne
  // CHECK: scf.if
  // CHECK: npuvector.transfer_read %{{.*}}[%{{.*}}] [{{.*}}] [{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<?xf32>
  // CHECK: npuvector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] : !npuvector<?xf32>, memref<128xf32>
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

  // CHECK: arith.constant 118 : index
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
// CHECK: arith.constant 118 : index
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
// CHECK: arith.constant 101 : index
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
// Phase 2 (VF=1 sweep): no tagged loops — scalar memref.load chain only
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
// CHECK: npuvector.transfer_read {{.*}} : memref<f32>, !npuvector<1xf32>
// CHECK: arith.mulf {{.*}} : !npuvector<1xf32>
// CHECK: npuvector.transfer_write {{.*}} : !npuvector<1xf32>, memref<f32>
func.func @test_vf1_sweep_no_tagged_loops(%in: memref<f32>, %out: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %x = memref.load %in[] : memref<f32>
  %c2 = arith.constant 2.000000e+00 : f32
  %m = arith.mulf %x, %c2 : f32
  memref.store %m, %out[] : memref<f32>
  return
}

// -----

// Phase 1 inlines vector=1; 0-d load stays scalar until Phase 2 promotes to !npuvector<1xf32>.
// CHECK-LABEL: func.func @test_vf1_sweep_after_vector1_collapse
// CHECK: memref.collapse_shape
// CHECK: npuvector.transfer_read {{.*}} : memref<f32>, !npuvector<1xf32>
// CHECK: arith.mulf {{.*}} : !npuvector<1xf32>
// CHECK: npuvector.transfer_write {{.*}} : !npuvector<1xf32>, memref<f32>
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
// that loop (bias prologue + tail) promote to !npuvector<1xf32> (batch-norm style).
// CHECK-LABEL: func.func @test_vf1_scalar_before_after_tagged_elementwise
// CHECK: npuvector.transfer_read {{.*}} : memref<4xf32>, !npuvector<1xf32>
// CHECK: arith.mulf {{.*}} : !npuvector<1xf32>
// CHECK: npuvector.transfer_read {{.*}} : memref<4xf32>, !npuvector<4xf32>
// CHECK: npuvector.transfer_write {{.*}} : !npuvector<4xf32>, memref<4xf32>
// CHECK: npuvector.transfer_read {{.*}} : memref<4xf32>, !npuvector<1xf32>
// CHECK: arith.addf {{.*}} : !npuvector<1xf32>
// CHECK: npuvector.transfer_write {{.*}} : !npuvector<1xf32>, memref<4xf32>
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

// Phase 2 must also mop up a remaining scalar store when there is no scalar memref.load
// left to seed the VF=1 chain.
// CHECK-LABEL: func.func @test_vf1_store_seed_after_reduction
// CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %{{.*}} : !npuvector<64xf32> into f32
// CHECK: %[[BCAST:.*]] = npuvector.broadcast %[[SUM]][%{{.*}}] [%{{.*}}] : f32 to !npuvector<1xf32>
// CHECK: npuvector.transfer_write %[[BCAST]], %{{.*}}[%{{.*}}] : !npuvector<1xf32>, memref<128xf32>
func.func @test_vf1_store_seed_after_reduction(%input: memref<64xf32>, %output: memref<128xf32>, %idx: index) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
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

// CHECK-LABEL: func.func @test_nested_vector_reduction_x_rank_lift_accumulator_dim_order
// CHECK: %[[VEC_ADD:.*]] = arith.addf {{.*}} : !npuvector<64xf16>
// CHECK: %[[NEUTRAL:.*]] = arith.constant dense<0.000000e+00> : !npuvector<64x8xf16>
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[NEUTRAL]]) -> (!npuvector<64x8xf16>) {
// CHECK: %[[LIFTED:.*]] = npuvector.broadcast %[[VEC_ADD]][{{.*}}] [{{.*}}] : !npuvector<64xf16> to !npuvector<64x8xf16>
// CHECK-SAME: {dimension = array<i64: 0>}
// CHECK: arith.addf %[[LIFTED]], %[[ACC]] {{.*}} : !npuvector<64x8xf16>
func.func @test_nested_vector_reduction_x_rank_lift_accumulator_dim_order(
    %input: memref<32x64xf16>,
    %output: memref<64x32xf32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %zero = arith.constant 0.000000e+00 : f16
  %one = arith.constant 1.000000e+00 : f16

  scf.for %row = %c0 to %c32 step %c1 {
    scf.for %col = %c0 to %c64 step %c1 {
      %value = memref.load %input[%row, %col] : memref<32x64xf16>
      %plus = arith.addf %value, %one : f16
      %sum = scf.for %red = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (f16) {
        %next = arith.addf %plus, %acc {reduction_axes = [2 : index], reduction_type = "x"} : f16
        scf.yield %next : f16
      } {reduction_x = 8 : i64}
      %cast = arith.extf %sum : f16 to f32
      memref.store %cast, %output[%col, %row] : memref<64x32xf32>
    } {vector = 64 : i64}
  }
  return
}
