// RUN: akg-opt %s --elim-scf-iter-args | FileCheck %s


// CHECK-LABEL:  module {
// CHECK-NEXT:  func.func @test_reduction_x_two_results(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<f32>, %arg3: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1024 = arith.constant 1024 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %c128 = arith.constant 128 : index
// CHECK-NEXT:    %c128_0 = arith.constant 128 : index
// CHECK-NEXT:    %alloc = memref.alloc() : memref<128xf32>
// CHECK-NEXT:    %alloc_1 = memref.alloc() : memref<128xf32>
// CHECK-NEXT:    %cst_2 = arith.constant dense<0.000000e+00> : !npuvector<128xf32>
// CHECK-NEXT:    %cst_3 = arith.constant dense<0.000000e+00> : !npuvector<128xf32>
// CHECK-NEXT:    %c0_4 = arith.constant 0 : index
// CHECK-NEXT:    npuvector.transfer_write %cst_2, %alloc[%c0_4] : !npuvector<128xf32>, memref<128xf32>
// CHECK-NEXT:    npuvector.transfer_write %cst_3, %alloc_1[%c0_4] : !npuvector<128xf32>, memref<128xf32>
// CHECK-NEXT:    scf.for %arg4 = %c0 to %c1024 step %c128_0 {
// CHECK-NEXT:      %c0_11 = arith.constant 0 : index
// CHECK-NEXT:      %c128_12 = arith.constant 128 : index
// CHECK-NEXT:      %cst_13 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %4 = npuvector.transfer_read %alloc[%c0_11] [%c128_12] [%c128], %cst_13 : memref<128xf32>, !npuvector<128xf32>
// CHECK-NEXT:      %cst_14 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %5 = npuvector.transfer_read %alloc_1[%c0_11] [%c128_12] [%c128], %cst_14 : memref<128xf32>, !npuvector<128xf32>
// CHECK-NEXT:      %cst_15 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %6 = npuvector.transfer_read %arg0[%arg4] [%c128_12] [%c128], %cst_15 : memref<1024xf32>, !npuvector<128xf32>
// CHECK-NEXT:      %cst_16 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %c128_17 = arith.constant 128 : index
// CHECK-NEXT:      %7 = npuvector.transfer_read %arg1[%arg4] [%c128_17] [%c128], %cst_16 : memref<1024xf32>, !npuvector<128xf32>
// CHECK-NEXT:      %8 = arith.mulf %6, %7 : !npuvector<128xf32>
// CHECK-NEXT:      %9 = arith.mulf %6, %6 : !npuvector<128xf32>
// CHECK-NEXT:      %10 = arith.addf %4, %8 {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<128xf32>
// CHECK-NEXT:      %11 = arith.addf %5, %9 {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<128xf32>
// CHECK-NEXT:      npuvector.transfer_write %10, %alloc[%c0_11] : !npuvector<128xf32>, memref<128xf32>
// CHECK-NEXT:      npuvector.transfer_write %11, %alloc_1[%c0_11] : !npuvector<128xf32>, memref<128xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:    %c128_5 = arith.constant 128 : index
// CHECK-NEXT:    %c128_6 = arith.constant 128 : index
// CHECK-NEXT:    %cst_7 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %0 = npuvector.transfer_read %alloc[%c0_4] [%c128_5] [%c128_6], %cst_7 : memref<128xf32>, !npuvector<128xf32>
// CHECK-NEXT:    %c128_8 = arith.constant 128 : index
// CHECK-NEXT:    %c128_9 = arith.constant 128 : index
// CHECK-NEXT:    %cst_10 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %1 = npuvector.transfer_read %alloc_1[%c0_4] [%c128_8] [%c128_9], %cst_10 : memref<128xf32>, !npuvector<128xf32>
// CHECK-NEXT:    %2 = npuvector.reduction <add>, %0 : !npuvector<128xf32> into f32
// CHECK-NEXT:    %3 = npuvector.reduction <add>, %1 : !npuvector<128xf32> into f32
// CHECK-NEXT:    memref.store %2, %arg2[] : memref<f32>
// CHECK-NEXT:    memref.store %3, %arg3[] : memref<f32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:}


func.func @test_reduction_x_two_results(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<f32>, %arg3: memref<f32>) attributes {hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c128 = arith.constant 128 : index
    %c128_0 = arith.constant 128 : index
    %cst_1 = arith.constant dense<0.000000e+00> : !npuvector<128xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : !npuvector<128xf32>
    %0:2 = scf.for %arg4 = %c0 to %c1024 step %c128_0 iter_args(%arg5 = %cst_1, %arg6 = %cst_2) -> (!npuvector<128xf32>, !npuvector<128xf32>) {
        %cst_3 = arith.constant 0.000000e+00 : f32
        %c128_4 = arith.constant 128 : index
        %3 = npuvector.transfer_read %arg0[%arg4] [%c128_4] [%c128], %cst_3 : memref<1024xf32>, !npuvector<128xf32>
        %cst_5 = arith.constant 0.000000e+00 : f32
        %c128_6 = arith.constant 128 : index
        %4 = npuvector.transfer_read %arg1[%arg4] [%c128_6] [%c128], %cst_5 : memref<1024xf32>, !npuvector<128xf32>
        %5 = arith.mulf %3, %4 : !npuvector<128xf32>
        %6 = arith.mulf %3, %3 : !npuvector<128xf32>
        %7 = arith.addf %arg5, %5 {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<128xf32>
        %8 = arith.addf %arg6, %6 {reduction_axes = [0 : index], reduction_type = "x"} : !npuvector<128xf32>
        scf.yield %7, %8 : !npuvector<128xf32>, !npuvector<128xf32>
    }
    %1 = npuvector.reduction <add>, %0#0 : !npuvector<128xf32> into f32
    %2 = npuvector.reduction <add>, %0#1 : !npuvector<128xf32> into f32
    memref.store %1, %arg2[] : memref<f32>
    memref.store %2, %arg3[] : memref<f32>
    return
}
