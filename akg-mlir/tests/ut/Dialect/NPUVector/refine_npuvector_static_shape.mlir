// RUN: akg-opt %s --refine-npuvector-static-shape --mlir-disable-threading --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @refine_const_dynamic_size_chain
func.func @refine_const_dynamic_size_chain(%input: memref<32xf32>, %out: memref<32xf32>) {
  // CHECK: %[[READ:.*]] = npuvector.transfer_read {{.*}} : memref<32xf32>, !npuvector<16xf32>
  // CHECK: %[[BCAST:.*]] = npuvector.broadcast {{.*}} : f32 to !npuvector<16xf32>
  // CHECK: %[[DIV:.*]] = arith.divf %[[READ]], %[[BCAST]] : !npuvector<16xf32>
  // CHECK: %[[SQRT:.*]] = math.sqrt %[[DIV]] : !npuvector<16xf32>
  // CHECK: npuvector.transfer_write %[[SQRT]], {{.*}} : !npuvector<16xf32>, memref<32xf32>
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.000000e+00 : f32
  %one = arith.constant 1.000000e+00 : f32
  %read = npuvector.transfer_read %input[%c0] [%c16] [%c16], %zero : memref<32xf32>, !npuvector<?xf32>
  %bcast = npuvector.broadcast %one[%c16] [%c16] : f32 to !npuvector<?xf32>
  %div = arith.divf %read, %bcast : !npuvector<?xf32>
  %sqrt = math.sqrt %div : !npuvector<?xf32>
  npuvector.transfer_write %sqrt, %out[%c0] : !npuvector<?xf32>, memref<32xf32>
  return
}

// CHECK-LABEL: func.func @refine_partial_dynamic_extf_transpose
func.func @refine_partial_dynamic_extf_transpose(%input: memref<32x40x64xbf16>, %out: memref<8x16x40xf32>, %n: index) {
  // CHECK: %[[READ:.*]] = npuvector.transfer_read {{.*}} : memref<32x40x64xbf16>, !npuvector<16x?x8xbf16>
  // CHECK: %[[EXT:.*]] = npuvector.extf %[[READ]] : !npuvector<16x?x8xbf16> to !npuvector<16x?x8xf32>
  // CHECK: %[[TRANS:.*]] = npuvector.transpose %[[EXT]], [2, 0, 1] : !npuvector<16x?x8xf32> to !npuvector<8x16x?xf32>
  // CHECK: npuvector.transfer_write %[[TRANS]], {{.*}} : !npuvector<8x16x?xf32>, memref<8x16x40xf32>
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c40 = arith.constant 40 : index
  %pad = arith.constant 0.000000e+00 : bf16
  %read = npuvector.transfer_read %input[%c0, %c0, %c0] [%c16, %n, %c8] [%c16, %c40, %c8], %pad : memref<32x40x64xbf16>, !npuvector<?x?x?xbf16>
  %ext = npuvector.extf %read : !npuvector<?x?x?xbf16> to !npuvector<?x?x?xf32>
  %trans = npuvector.transpose %ext, [2, 0, 1] : !npuvector<?x?x?xf32> to !npuvector<?x?x?xf32>
  npuvector.transfer_write %trans, %out[%c0, %c0, %c0] : !npuvector<?x?x?xf32>, memref<8x16x40xf32>
  return
}
