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
  
  // CHECK: %[[X_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<1024xf32>, !npuvector<1024xf32>
  // CHECK: %[[Y_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<1024xf32>, !npuvector<1024xf32>
  // CHECK: %[[ALPHA_VEC:.*]] = npuvector.broadcast %{{.*}} : f32 to !npuvector<1024xf32>
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
  // CHECK: %[[TRIP:.*]] = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[MAX:.*]] = arith.constant 4096 : index
  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[TRIP]]] [%[[MAX]]], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
  // CHECK: %[[SCALE_VEC:.*]] = npuvector.broadcast %{{.*}}[%[[TRIP]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
  // CHECK: %[[SCALED_VEC:.*]] = arith.mulf %[[V_VEC]], %[[SCALE_VEC]] : !npuvector<?xf32>
  // CHECK: %[[BIAS_VEC:.*]] = npuvector.broadcast %{{.*}}[%[[TRIP]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
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
  // CHECK: %[[TRIP:.*]] = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[MAX:.*]] = arith.constant 4096 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[NEUTRAL:.*]] = npuvector.broadcast %{{.*}}[%[[TRIP]]] [%[[MAX]]] : f32 to !npuvector<?xf32>
  // CHECK: %[[SUM_VEC:.*]] = scf.for %{{.*}} = %{{.*}} to %[[DIM]] step %[[TRIP]] iter_args(%[[ACC_VEC:.*]] = %[[NEUTRAL]]) -> (!npuvector<?xf32>) {
  // CHECK:   %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[TRIP]]] [%[[MAX]]], %{{.*}} : memref<?xf32>, !npuvector<?xf32>
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

// -----

// ============================================================================
// Type Conversion: Float Extension
// ============================================================================

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

// ============================================================================
// Type Conversion: Float Truncation
// ============================================================================

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

// ============================================================================
// Type Conversion: Integer Extension (Signed)
// ============================================================================

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

// ============================================================================
// Type Conversion: Integer Extension (Unsigned)
// ============================================================================

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

// ============================================================================
// Type Conversion: Integer Truncation
// ============================================================================

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

// ============================================================================
// Type Conversion: Integer to Float (Signed)
// ============================================================================

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

// ============================================================================
// Type Conversion: Integer to Float (Unsigned)
// ============================================================================

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

// ============================================================================
// Type Conversion: Float to Integer (Signed)
// ============================================================================

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

// ============================================================================
// Type Conversion: Float to Integer (Unsigned)
// ============================================================================

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

// ============================================================================
// Type Conversion: Mixed Operations
// ============================================================================

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

// ============================================================================
// Type Conversion: Dynamic Shape
// ============================================================================

// CHECK-LABEL: func.func @test_extf_dynamic
func.func @test_extf_dynamic(%input: memref<?xbf16>, %output: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %N = memref.dim %input, %c0 : memref<?xbf16>

  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xbf16>
  // CHECK: %[[TRIP:.*]] = arith.subi %[[DIM]], %{{.*}} : index
  // CHECK: %[[MAX:.*]] = arith.constant 4096 : index
  // CHECK: %[[V_VEC:.*]] = npuvector.transfer_read %{{.*}}[%{{.*}}] [%[[TRIP]]] [%[[MAX]]], %{{.*}} : memref<?xbf16>, !npuvector<?xbf16>
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

// ============================================================================
// Type Conversion: Bitcast
// ============================================================================

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

// ============================================================================
// CmpIOp
// ============================================================================

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

// ============================================================================
// CmpFOp
// ============================================================================

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

// ============================================================================
// SelectOp
// ============================================================================

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

// ============================================================================
// SCF.If: Conditional operation with side effects (store)
// ============================================================================

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
    %cond = arith.cmpi ult, %i, %threshold : index
    scf.if %cond {
      %scaled = arith.mulf %val, %c2 : f32
      memref.store %scaled, %output[%i] : memref<128xf32>
    }
  } {vector=128}
  return
}