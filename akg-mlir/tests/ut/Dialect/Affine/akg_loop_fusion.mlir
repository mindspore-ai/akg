// RUN: akg-opt %s -akg-loop-fusion | FileCheck %s

// CHECK-LABEL: func.func @Fused_Mul_ReduceSum(%arg0: tensor<2x3072xbf16>) -> tensor<2x1xbf16> attributes {OperatorType = "Reduce", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : bf16
// CHECK-NEXT:   %0 = bufferization.to_memref %arg0 : memref<2x3072xbf16, strided<[?, ?], offset: ?>>
// CHECK-NEXT:   %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2xbf16>
// CHECK-NEXT:   affine.for %arg1 = 0 to 2 {
// CHECK-NEXT:     affine.store %cst, %alloc_0[%arg1] : memref<2xbf16>
// CHECK-NEXT:     affine.for %arg2 = 0 to 3072 {
// CHECK-NEXT:       %2 = affine.load %0[%arg1, %arg2] : memref<2x3072xbf16, strided<[?, ?], offset: ?>>
// CHECK-NEXT:       %3 = arith.mulf %2, %2 : bf16
// CHECK-NEXT:       %4 = affine.load %alloc_0[%arg1] : memref<2xbf16>
// CHECK-NEXT:       %5 = arith.addf %3, %4 : bf16
// CHECK-NEXT:       affine.store %5, %alloc_0[%arg1] : memref<2xbf16>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %expand_shape = memref.expand_shape %alloc_0 [[0, 1]] output_shape [2, 1] : memref<2xbf16> into memref<2x1xbf16>
// CHECK-NEXT:   %1 = bufferization.to_tensor %expand_shape : memref<2x1xbf16>
// CHECK-NEXT:   return %1 : tensor<2x1xbf16>
// CHECK-NEXT: }


func.func @Fused_Mul_ReduceSum_split(%arg0: tensor<2x3072xbf16>) -> tensor<2x1xbf16> attributes {OperatorType = "Reduce", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %cst = arith.constant 0.000000e+00 : bf16
  %0 = bufferization.to_memref %arg0 : memref<2x3072xbf16, strided<[?, ?], offset: ?>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x3072xbf16>
  affine.for %arg1 = 0 to 2 {
    affine.for %arg2 = 0 to 3072 {
      %2 = affine.load %0[%arg1, %arg2] : memref<2x3072xbf16, strided<[?, ?], offset: ?>>
      %3 = arith.mulf %2, %2 : bf16
      affine.store %3, %alloc[%arg1, %arg2] : memref<2x3072xbf16>
    }
  }
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2xbf16>
  affine.for %arg1 = 0 to 2 {
    affine.store %cst, %alloc_0[%arg1] : memref<2xbf16>
  }
  affine.for %arg1 = 0 to 2 {
    affine.for %arg2 = 0 to 3072 {
      %2 = affine.load %alloc[%arg1, %arg2] : memref<2x3072xbf16>
      %3 = affine.load %alloc_0[%arg1] : memref<2xbf16>
      %4 = arith.addf %2, %3 : bf16
      affine.store %4, %alloc_0[%arg1] : memref<2xbf16>
    }
  }
  %expand_shape = memref.expand_shape %alloc_0 [[0, 1]] output_shape [2, 1] : memref<2xbf16> into memref<2x1xbf16>
  %1 = bufferization.to_tensor %expand_shape : memref<2x1xbf16>
  return %1 : tensor<2x1xbf16>
}