// RUN: akg-opt %s --npuvector-vectorize --mlir-disable-threading --allow-unregistered-dialect | FileCheck %s

// ============================================================================
// 1. 1-D Elementwise static and dynamic
// ============================================================================

// CHECK-LABEL: func.func @test_elementwise_static
func.func @test_elementwise_static(
    %X: memref<1024xf32>,
    %Y: memref<1024xf32>,
    %Z: memref<1024xf32>,
    %alpha: f32) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[MAX:.*]] = arith.constant 4096 : index
  // CHECK: %[[X_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[MAX]]], %{{.*}} : memref<1024xf32>, !npuvector<1024xf32>
  // CHECK: %[[Y_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[MAX]]], %{{.*}} : memref<1024xf32>, !npuvector<1024xf32>
  // CHECK: %[[ALPHA_VEC:.*]] = npuvector.broadcast %{{.*}}[%[[MAX]]] : f32 to !npuvector<1024xf32>
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
  } {vector=4096}
  return
}

// -----

// CHECK-LABEL: func.func @test_elementwise_dynamic
func.func @test_elementwise_dynamic(
    %input: memref<?xf32>,
    %output: memref<?xf32>,
    %scale: f32,
    %bias: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %N = memref.dim %input, %c0 : memref<?xf32>

  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
  // CHECK: %[[MAX:.*]] = arith.constant 4096 : index
  // CHECK: %[[TRIP:.*]] = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[VEC_SIZE:.*]] = arith.minsi %[[TRIP]], %[[MAX]] : index
  // CHECK: scf.for {{.*}} = %{{.*}} to %[[DIM]] step %{{.*}} {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VEC_SIZE]]] [%[[MAX]]], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   %[[SCALE_VEC:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
  // CHECK:   %[[SCALED_VEC:.*]] = arith.mulf %[[V_VEC]], %[[SCALE_VEC]] : !npuvector<?xf32>
  // CHECK:   %[[BIAS_VEC:.*]] = npuvector.broadcast %{{.*}}[%[[VEC_SIZE]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
  // CHECK:   %[[RESULT_VEC:.*]] = arith.addf %[[SCALED_VEC]], %[[BIAS_VEC]] : !npuvector<?xf32>
  // CHECK:   npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<?xf32>, memref<?xf32>

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
func.func @test_reduction_static(%input: memref<1024xf32>, %output: memref<f32>) {
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
func.func @test_reduction_dynamic(%input: memref<?xf32>, %output: memref<f32>) {
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

// ============================================================================
// 3. 1-D tail handling static and dynamic
// Static: 300 elements, vf=128 -> 256+44 tail; dynamic tail see test_reduction_dynamic (section 2)
// ============================================================================

// CHECK-LABEL: func.func @test_reduction_static_with_tail
func.func @test_reduction_static_with_tail(%input: memref<300xf32>, %output: memref<f32>) {
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
    %input: memref<32x64xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index

  // CHECK: memref.load
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
    %input: memref<32x64xf32>) {
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
    %M: index, %N: index) {
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
    %output: memref<64xf32>) {
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
    %M: index, %N: index) {
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
    %a: memref<32x64xf32>, %b: memref<32x64xf32>, %c: memref<32x64xf32>) {
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
    %a: memref<32x64xf32>, %b: memref<32x64xf32>, %c: memref<32x64xf32>) {
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
    %M: index, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: npuvector.transfer_read
  // CHECK: npuvector.transfer_read
  // CHECK: arith.addf
  // CHECK: npuvector.transfer_write
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

// ============================================================================
// 6. Multi-dim with Transpose static and dynamic
// ============================================================================

// Store-only transpose: load a[i,j], b[i,j], store out[j,i]

// CHECK-LABEL: func.func @test_multidim_transpose_store_only
func.func @test_multidim_transpose_store_only(
    %a: memref<32x64xf32>, %b: memref<32x64xf32>, %out: memref<64x32xf32>) {
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
    %a1: memref<64x32xf32>, %b1: memref<64x32xf32>, %d: memref<64x32xf32>) {
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
    %a: memref<8x16x32xf32>, %out: memref<8x16x32xf32>) {
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
    %a: memref<32x64xf32>, %out: memref<64x32xf32>, %threshold: index) {
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
    %a: memref<?x?xf32>, %out: memref<?x?xf32>, %M: index, %N: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: npuvector.transfer_read
  // CHECK: npuvector.transpose
  // CHECK: npuvector.transfer_write

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
    %a: memref<?x?xf32>, %out: memref<?x?xf32>, %M: index, %N: index, %thresh: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2.0 : f32

  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
  // CHECK: npuvector.transfer_read
  // CHECK: scf.if
  // CHECK: npuvector.transpose
  // CHECK: npuvector.transfer_write

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
    %a: memref<32x64xf32>, %out: memref<64x32xf32>) {
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
    %a: memref<8x16x32xf32>, %out: memref<32x16x8xf32>) {
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
    %src: memref<4x8x16x32xf32>, %dst: memref<4x16x32x8xf32>) {
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
    %a: memref<16x32xf32>, %out: memref<32x16xf32>, %tile: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index

  // CHECK: scf.for {{.*}} to {{.*}} step {{.*}} {
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
func.func @test_extf_static(%input: memref<128xbf16>, %output: memref<128xf32>) {
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
func.func @test_truncf_static(%input: memref<128xf32>, %output: memref<128xbf16>) {
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
func.func @test_extsi_static(%input: memref<128xi8>, %output: memref<128xi32>) {
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
func.func @test_extui_static(%input: memref<128xi8>, %output: memref<128xi32>) {
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
func.func @test_trunci_static(%input: memref<128xi32>, %output: memref<128xi8>) {
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
func.func @test_sitofp_static(%input: memref<128xi32>, %output: memref<128xf32>) {
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
func.func @test_uitofp_static(%input: memref<128xi32>, %output: memref<128xf32>) {
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
func.func @test_fptosi_static(%input: memref<128xf32>, %output: memref<128xi32>) {
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
func.func @test_fptoui_static(%input: memref<128xf32>, %output: memref<128xi32>) {
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
func.func @test_bitcast_static(%input: memref<128xbf16>, %output: memref<128xi16>) {
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
func.func @test_mixed_type_conversion(%input: memref<128xbf16>, %output: memref<128xbf16>) {
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
func.func @test_extf_dynamic(%input: memref<?xbf16>, %output: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %N = memref.dim %input, %c0 : memref<?xbf16>

  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xbf16>
  // CHECK: %[[MAX:.*]] = arith.constant 4096 : index
  // CHECK: %[[TRIP:.*]] = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[VEC_SIZE:.*]] = arith.minsi %[[TRIP]], %[[MAX]] : index
  // CHECK: scf.for {{.*}} = %{{.*}} to %[[DIM]] step %{{.*}} {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[VEC_SIZE]]] [%[[MAX]]], %{{.*}} : memref<?xbf16>, !npuvector<?xbf16>
  // CHECK:   %[[RESULT_VEC:.*]] = npuvector.extf %[[V_VEC]] : !npuvector<?xbf16> to !npuvector<?xf32>
  // CHECK:   npuvector.transfer_write %[[RESULT_VEC]], %{{.*}}[%{{.*}}] : !npuvector<?xf32>, memref<?xf32>

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
func.func @test_cmpi_static(%input: memref<128xi32>, %output: memref<128xi1>) {
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
func.func @test_cmpf_static(%input: memref<128xf32>, %output: memref<128xi1>) {
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
    %output: memref<128xf32>) {
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
    %threshold: index) {
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
