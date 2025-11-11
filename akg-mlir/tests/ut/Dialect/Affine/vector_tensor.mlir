// RUN: akg-opt %s --vector-transfer-tensorize | FileCheck %s

// CHECK-LABEL: func.func @Fused_Add_Small(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>) -> tensor<1xf32> {
// CHECK-NEXT:   %0 = bufferization.to_memref %arg0 : memref<1xf32>
// CHECK-NEXT:   %1 = bufferization.to_memref %arg1 : memref<1xf32>
// CHECK-NEXT:   %2 = bufferization.to_memref %arg2 : memref<1xf32>
// CHECK-NEXT:   %3 = bufferization.to_memref %arg3 : memref<1xf32>
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
// CHECK-NEXT:   %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
// CHECK-NEXT:   %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
// CHECK-NEXT:   %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
// CHECK-NEXT:   affine.for %arg4 = 0 to 1 {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %subview = memref.subview %0[0] [1] [1] : memref<1xf32> to memref<1xf32>
// CHECK-NEXT:     %5 = bufferization.to_tensor %subview restrict writable : memref<1xf32>
// CHECK-NEXT:     %cst_3 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %subview_4 = memref.subview %1[0] [1] [1] : memref<1xf32> to memref<1xf32>
// CHECK-NEXT:     %6 = bufferization.to_tensor %subview_4 restrict writable : memref<1xf32>
// CHECK-NEXT:     %7 = arith.addf %5, %6 : tensor<1xf32>
// CHECK-NEXT:     %subview_5 = memref.subview %alloc[0] [1] [1] : memref<1xf32> to memref<1xf32>
// CHECK-NEXT:     %8 = bufferization.to_memref %7 : memref<1xf32>
// CHECK-NEXT:     memref.copy %8, %subview_5 : memref<1xf32> to memref<1xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %4 = bufferization.to_tensor %alloc_2 : memref<1xf32>
// CHECK-NEXT:   return %4 : tensor<1xf32>
// CHECK-NEXT: }


func.func @Fused_Add_Small(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>) -> tensor<1xf32> {
    %0 = bufferization.to_memref %arg0 : memref<1xf32>
    %1 = bufferization.to_memref %arg1 : memref<1xf32>
    %2 = bufferization.to_memref %arg2 : memref<1xf32>
    %3 = bufferization.to_memref %arg3 : memref<1xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1xf32>
    affine.for %arg4 = 0 to 1 {
        %cst = arith.constant 0.000000e+00 : f32
        %5 = vector.transfer_read %0[%arg4], %cst : memref<1xf32>, vector<1xf32>
        %cst_3 = arith.constant 0.000000e+00 : f32
        %6 = vector.transfer_read %1[%arg4], %cst_3 : memref<1xf32>, vector<1xf32>
        %7 = arith.addf %5, %6 : vector<1xf32>
        vector.transfer_write %7, %alloc[%arg4] : vector<1xf32>, memref<1xf32>
    }
    %4 = bufferization.to_tensor %alloc_2 : memref<1xf32>
    return %4 : tensor<1xf32>
}
