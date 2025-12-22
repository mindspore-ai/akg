// RUN: akg-opt %s --npuvector-vectorize --mlir-disable-threading --allow-unregistered-dialect | FileCheck %s

// ============================================================================
// Static Elementwise
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
  
  // CHECK: %[[ALPHA_VEC:.*]] = npuvector.broadcast %{{.*}} : f32 to !npuvector<1024xf32>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
  // CHECK:   %[[X_VEC:.*]] = npuvector.transfer_read %{{.*}} : memref<1024xf32>, !npuvector<1024xf32>
  // CHECK:   %[[Y_VEC:.*]] = npuvector.transfer_read %{{.*}} : memref<1024xf32>, !npuvector<1024xf32>
  // CHECK:   %[[AX_VEC:.*]] = arith.mulf %[[ALPHA_VEC]], %[[X_VEC]] : !npuvector<1024xf32>
  // CHECK:   %[[RESULT_VEC:.*]] = arith.addf %[[AX_VEC]], %[[Y_VEC]] : !npuvector<1024xf32>
  // CHECK:   npuvector.transfer_write %[[RESULT_VEC]], %{{.*}} : !npuvector<1024xf32>, memref<1024xf32>
  // CHECK: }
  
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

// ============================================================================
// Dynamic Elementwise
// ============================================================================

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
  // CHECK: %[[SCALE_VEC:.*]] = npuvector.broadcast %{{.*}}[%[[DIM]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
  // CHECK: %[[BIAS_VEC:.*]] = npuvector.broadcast %{{.*}}[%[[DIM]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
  // CHECK: scf.for %{{.*}} = %{{.*}} to %[[DIM]] step %[[DIM]] {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}][%[[DIM]]][%[[MAX]]], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   %[[SCALED_VEC:.*]] = arith.mulf %[[V_VEC]], %[[SCALE_VEC]] : !npuvector<?xf32>
  // CHECK:   %[[RESULT_VEC:.*]] = arith.addf %[[SCALED_VEC]], %[[BIAS_VEC]] : !npuvector<?xf32>
  // CHECK:   npuvector.transfer_write %[[RESULT_VEC]], %{{.*}} : !npuvector<?xf32>, memref<?xf32>
  // CHECK: }
  
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
// Static Reduction
// ============================================================================

// CHECK-LABEL: func.func @test_reduction_static
func.func @test_reduction_static(%input: memref<1024xf32>, %output: memref<f32>) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0.0 : f32
  
  // CHECK: %[[NEUTRAL:.*]] = arith.constant dense<0.000000e+00> : !npuvector<1024xf32>
  // CHECK: %[[SUM_VEC:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC_VEC:.*]] = %[[NEUTRAL]]) -> (!npuvector<1024xf32>) {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}} : memref<1024xf32>, !npuvector<1024xf32>
  // CHECK:   %[[NEW_ACC_VEC:.*]] = arith.addf %[[ACC_VEC]], %[[V_VEC]] : !npuvector<1024xf32>
  // CHECK:   scf.yield %[[NEW_ACC_VEC]] : !npuvector<1024xf32>
  // CHECK: }
  // CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %[[SUM_VEC]] : !npuvector<1024xf32> into f32
  // CHECK: memref.store %[[SUM]], %{{.*}}[] : memref<f32>
  
  %sum = scf.for %i = %c0 to %c1024 step %c1 iter_args(%acc = %init) -> f32 {
    %v = memref.load %input[%i] : memref<1024xf32>
    %new_acc = arith.addf %acc, %v : f32
    scf.yield %new_acc : f32
  } {reduction_loop=4096}
  
  memref.store %sum, %output[] : memref<f32>
  return
}

// -----

// ============================================================================
// Dynamic Reduction
// ============================================================================

// CHECK-LABEL: func.func @test_reduction_dynamic
func.func @test_reduction_dynamic(%input: memref<?xf32>, %output: memref<f32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %N = memref.dim %input, %c0 : memref<?xf32>
  %init = arith.constant 0.0 : f32
  
  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[MAX:.*]] = arith.constant 4096 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[NEUTRAL:.*]] = npuvector.broadcast %{{.*}}[%[[DIM]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
  // CHECK: %[[SUM_VEC:.*]] = scf.for %{{.*}} = %{{.*}} to %[[DIM]] step %[[DIM]] iter_args(%[[ACC_VEC:.*]] = %[[NEUTRAL]]) -> (!npuvector<?xf32>) {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}][%[[DIM]]][%[[MAX]]], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK:   %[[NEW_ACC_VEC:.*]] = arith.addf %[[ACC_VEC]], %[[V_VEC]] : !npuvector<?xf32>
  // CHECK:   scf.yield %[[NEW_ACC_VEC]] : !npuvector<?xf32>
  // CHECK: }
  // CHECK: %[[SUM:.*]] = npuvector.reduction <add>, %[[SUM_VEC]] : !npuvector<?xf32> into f32
  // CHECK: memref.store %[[SUM]], %{{.*}}[] : memref<f32>
  
  %sum = scf.for %i = %c0 to %N step %c1 iter_args(%acc = %init) -> f32 {
    %v = memref.load %input[%i] : memref<?xf32>
    %new_acc = arith.addf %acc, %v : f32
    scf.yield %new_acc : f32
  } {reduction_loop=4096}
  
  memref.store %sum, %output[] : memref<f32>
  return
}