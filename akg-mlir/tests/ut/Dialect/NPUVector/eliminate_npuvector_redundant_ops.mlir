// RUN: akg-opt %s --eliminate-npuvector-redundant-ops --mlir-disable-threading | FileCheck %s

// CHECK-LABEL: func.func @deduplicate_identical_reads
func.func @deduplicate_identical_reads(%arg0: memref<128xf32>, %out: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f32

  // CHECK: %[[READ:.*]] = npuvector.transfer_read %arg0[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK-NOT: npuvector.transfer_read
  // CHECK: %[[SUM:.*]] = arith.addf %[[READ]], %[[READ]] : !npuvector<128xf32>
  // CHECK: npuvector.transfer_write %[[SUM]], %{{.*}}[%{{.*}}] : !npuvector<128xf32>, memref<128xf32>
  %read0 = npuvector.transfer_read %arg0[%c0], %pad : memref<128xf32>, !npuvector<128xf32>
  %read1 = npuvector.transfer_read %arg0[%c0], %pad : memref<128xf32>, !npuvector<128xf32>
  %sum = arith.addf %read0, %read1 : !npuvector<128xf32>
  npuvector.transfer_write %sum, %out[%c0] : !npuvector<128xf32>, memref<128xf32>
  return
}

// -----

// CHECK-LABEL: func.func @forward_write_to_later_read_and_drop_local_buffer
// CHECK-NOT: memref.alloc
func.func @forward_write_to_later_read_and_drop_local_buffer(
    %arg0: memref<128xf32>,
    %arg1: memref<128xf32>,
    %out: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f32

  // CHECK: %[[LHS:.*]] = npuvector.transfer_read %arg0[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK: %[[RHS:.*]] = npuvector.transfer_read %arg1[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK: %[[SUM:.*]] = arith.addf %[[LHS]], %[[RHS]] : !npuvector<128xf32>
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: npuvector.transfer_read
  // CHECK: %[[OUT:.*]] = arith.addf %[[SUM]], %[[SUM]] : !npuvector<128xf32>
  // CHECK: npuvector.transfer_write %[[OUT]], %{{.*}}[%{{.*}}] : !npuvector<128xf32>, memref<128xf32>
  %lhs = npuvector.transfer_read %arg0[%c0], %pad : memref<128xf32>, !npuvector<128xf32>
  %rhs = npuvector.transfer_read %arg1[%c0], %pad : memref<128xf32>, !npuvector<128xf32>
  %sum = arith.addf %lhs, %rhs : !npuvector<128xf32>
  %scratch = memref.alloc() : memref<128xf32>
  npuvector.transfer_write %sum, %scratch[%c0] : !npuvector<128xf32>, memref<128xf32>
  %reload = npuvector.transfer_read %scratch[%c0], %pad : memref<128xf32>, !npuvector<128xf32>
  %outv = arith.addf %sum, %reload : !npuvector<128xf32>
  npuvector.transfer_write %outv, %out[%c0] : !npuvector<128xf32>, memref<128xf32>
  return
}

// -----

// CHECK-LABEL: func.func @forward_latest_write_not_stale_read
func.func @forward_latest_write_not_stale_read(
    %arg0: memref<128xf32>,
    %arg1: memref<128xf32>,
    %out: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.000000e+00 : f32

  // CHECK: %[[OLD:.*]] = npuvector.transfer_read %arg0[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK: %[[RHS:.*]] = npuvector.transfer_read %arg1[%{{.*}}], %{{.*}} : memref<128xf32>, !npuvector<128xf32>
  // CHECK: %[[NEW:.*]] = arith.addf %[[OLD]], %[[RHS]] : !npuvector<128xf32>
  // CHECK: npuvector.transfer_write %[[NEW]], %arg0[%{{.*}}] : !npuvector<128xf32>, memref<128xf32>
  // CHECK-NOT: npuvector.transfer_read
  // CHECK: %[[OUT:.*]] = arith.addf %[[NEW]], %[[NEW]] : !npuvector<128xf32>
  // CHECK: npuvector.transfer_write %[[OUT]], %{{.*}}[%{{.*}}] : !npuvector<128xf32>, memref<128xf32>
  %old = npuvector.transfer_read %arg0[%c0], %pad : memref<128xf32>, !npuvector<128xf32>
  %rhs = npuvector.transfer_read %arg1[%c0], %pad : memref<128xf32>, !npuvector<128xf32>
  %new = arith.addf %old, %rhs : !npuvector<128xf32>
  npuvector.transfer_write %new, %arg0[%c0] : !npuvector<128xf32>, memref<128xf32>
  %reload = npuvector.transfer_read %arg0[%c0], %pad : memref<128xf32>, !npuvector<128xf32>
  %outv = arith.addf %new, %reload : !npuvector<128xf32>
  npuvector.transfer_write %outv, %out[%c0] : !npuvector<128xf32>, memref<128xf32>
  return
}

// -----

// CHECK-LABEL: func.func @do_not_forward_across_unknown_memory_write
func.func @do_not_forward_across_unknown_memory_write(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %out: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.000000e+00 : f32
  %pad = arith.constant 0.000000e+00 : f32

  // CHECK: npuvector.transfer_read %arg0
  // CHECK: memref.store
  // CHECK: npuvector.transfer_read %arg0
  %read0 = npuvector.transfer_read %arg0[%c0], %pad : memref<128xf32>, !npuvector<128xf32>
  memref.store %one, %arg1[%c0] : memref<128xf32>
  %read1 = npuvector.transfer_read %arg0[%c0], %pad : memref<128xf32>, !npuvector<128xf32>
  %sum = arith.addf %read0, %read1 : !npuvector<128xf32>
  npuvector.transfer_write %sum, %out[%c0] : !npuvector<128xf32>, memref<128xf32>
  return
}
