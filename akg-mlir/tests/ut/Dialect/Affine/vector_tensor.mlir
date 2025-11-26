// RUN: akg-opt %s --vector-transfer-tensorize -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @Fused_BiasAdd_10033593016906428850(%arg0: tensor<28x3072xbf16>, %arg1: tensor<3072xbf16>) -> tensor<28x3072xbf16> {
// CHECK-NEXT:     %0 = tensor.empty() : tensor<28x3072xbf16>
// CHECK-NEXT:     %1 = tensor.empty() : tensor<28x3072xbf16>
// CHECK-NEXT:     %2:2 = affine.for %arg2 = 0 to 28 iter_args(%arg3 = %0, %arg4 = %1) -> (tensor<28x3072xbf16>, tensor<28x3072xbf16>) {
// CHECK-NEXT:       %3:2 = affine.for %arg5 = 0 to 3072 step 3072 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<28x3072xbf16>, tensor<28x3072xbf16>) {
// CHECK-NEXT:         %extracted_slice = tensor.extract_slice %arg1[%arg5] [3072] [1] : tensor<3072xbf16> to tensor<3072xbf16>
// CHECK-NEXT:         %inserted_slice = tensor.insert_slice %extracted_slice into %arg6[%arg2, %arg5] [1, 3072] [1, 1] : tensor<3072xbf16> into tensor<28x3072xbf16>
// CHECK-NEXT:         %extracted_slice_0 = tensor.extract_slice %arg0[%arg2, %arg5] [1, 3072] [1, 1] : tensor<28x3072xbf16> to tensor<3072xbf16>
// CHECK-NEXT:         %extracted_slice_1 = tensor.extract_slice %0[%arg2, %arg5] [1, 3072] [1, 1] : tensor<28x3072xbf16> to tensor<3072xbf16>
// CHECK-NEXT:         %4 = arith.addf %extracted_slice_0, %extracted_slice_1 : tensor<3072xbf16>
// CHECK-NEXT:         %inserted_slice_2 = tensor.insert_slice %4 into %arg7[%arg2, %arg5] [1, 3072] [1, 1] : tensor<3072xbf16> into tensor<28x3072xbf16>
// CHECK-NEXT:         affine.yield %inserted_slice, %inserted_slice_2 : tensor<28x3072xbf16>, tensor<28x3072xbf16>
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.yield %3#0, %3#1 : tensor<28x3072xbf16>, tensor<28x3072xbf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return %2#1 : tensor<28x3072xbf16>
// CHECK-NEXT:   }
// CHECK-NEXT: }

module {
    func.func @Fused_BiasAdd_10033593016906428850(%arg0: tensor<28x3072xbf16>, %arg1: tensor<3072xbf16>) -> tensor<28x3072xbf16> {
    %0 = bufferization.to_memref %arg0 : memref<28x3072xbf16>
    %1 = bufferization.to_memref %arg1 : memref<3072xbf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<28x3072xbf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<28x3072xbf16>
    affine.for %arg2 = 0 to 28 {
        affine.for %arg3 = 0 to 3072 step 3072 {
        %cst = arith.constant 0.000000e+00 : bf16
        %3 = vector.transfer_read %1[%arg3], %cst : memref<3072xbf16>, vector<3072xbf16>
        vector.transfer_write %3, %alloc[%arg2, %arg3] : vector<3072xbf16>, memref<28x3072xbf16>
        %cst_1 = arith.constant 0.000000e+00 : bf16
        %4 = vector.transfer_read %0[%arg2, %arg3], %cst_1 : memref<28x3072xbf16>, vector<3072xbf16>
        %cst_2 = arith.constant 0.000000e+00 : bf16
        %5 = vector.transfer_read %alloc[%arg2, %arg3], %cst_2 : memref<28x3072xbf16>, vector<3072xbf16>
        %6 = arith.addf %4, %5 : vector<3072xbf16>
        vector.transfer_write %6, %alloc_0[%arg2, %arg3] : vector<3072xbf16>, memref<28x3072xbf16>
        }
    }
    %2 = bufferization.to_tensor %alloc_0 : memref<28x3072xbf16>
    return %2 : tensor<28x3072xbf16>
    }
}