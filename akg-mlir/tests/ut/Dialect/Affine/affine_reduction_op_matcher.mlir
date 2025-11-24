// RUN: akg-opt %s -affine-reduction-annotation -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @Fused_Cast_ReduceSum_split_1645489274500693274(%arg0: memref<1x5100x3072xbf16>) -> memref<1x1x3072xf32> attributes {OperatorType = "Reduce", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5100x3072xf32>
// CHECK-NEXT:   affine.for %arg1 = 0 to 1 {
// CHECK-NEXT:     affine.for %arg2 = 0 to 5100 {
// CHECK-NEXT:       affine.for %arg3 = 0 to 3072 {
// CHECK-NEXT:         %0 = affine.load %arg0[%c0, %arg2, %arg3] : memref<1x5100x3072xbf16>
// CHECK-NEXT:         %1 = arith.extf %0 : bf16 to f32
// CHECK-NEXT:         affine.store %1, %alloc[%arg1, %arg2, %arg3] : memref<1x5100x3072xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x3072xf32>
// CHECK-NEXT:   affine.for %arg1 = 0 to 1 {
// CHECK-NEXT:     affine.for %arg2 = 0 to 3072 {
// CHECK-NEXT:       affine.store %cst, %alloc_0[%arg1, %arg2] : memref<1x3072xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.for %arg1 = 0 to 1 {
// CHECK-NEXT:     affine.for %arg2 = 0 to 5100 {
// CHECK-NEXT:       affine.for %arg3 = 0 to 3072 {
// CHECK-NEXT:         %0 = affine.load %alloc[%arg1, %arg2, %arg3] : memref<1x5100x3072xf32>
// CHECK-NEXT:         %1 = affine.load %alloc_0[%arg1, %arg3] : memref<1x3072xf32>
// CHECK-NEXT:         %2 = arith.addf %0, %1 {reduction_axes = [1 : index], reduction_type = "y"} : f32
// CHECK-NEXT:         affine.store %2, %alloc_0[%arg1, %arg3] : memref<1x3072xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:    } {reduceLoop}
// CHECK-NEXT:   }
// CHECK-NEXT:   %expand_shape = memref.expand_shape %alloc_0 {{\[\[0\], \[1, 2\]\]}} output_shape [1, 1, 3072] : memref<1x3072xf32> into memref<1x1x3072xf32>
// CHECK-NEXT:   return %expand_shape : memref<1x1x3072xf32>
// CHECK-NEXT: }

func.func @Fused_Cast_ReduceSum_split_1645489274500693274(%arg0: memref<1x5100x3072xbf16>) -> memref<1x1x3072xf32> attributes {OperatorType = "Reduce", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5100x3072xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 5100 {
      affine.for %arg3 = 0 to 3072 {
        %0 = affine.load %arg0[%c0, %arg2, %arg3] : memref<1x5100x3072xbf16>
        %1 = arith.extf %0 : bf16 to f32
        affine.store %1, %alloc[%arg1, %arg2, %arg3] : memref<1x5100x3072xf32>
      }
    }
  }
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x3072xf32>
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 3072 {
      affine.store %cst, %alloc_0[%arg1, %arg2] : memref<1x3072xf32>
    }
  }
  affine.for %arg1 = 0 to 1 {
    affine.for %arg2 = 0 to 5100 {
      affine.for %arg3 = 0 to 3072 {
        %0 = affine.load %alloc[%arg1, %arg2, %arg3] : memref<1x5100x3072xf32>
        %1 = affine.load %alloc_0[%arg1, %arg3] : memref<1x3072xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc_0[%arg1, %arg3] : memref<1x3072xf32>
      }
    }
  }
  %expand_shape = memref.expand_shape %alloc_0 [[0], [1, 2]] output_shape [1, 1, 3072] : memref<1x3072xf32> into memref<1x1x3072xf32>
  return %expand_shape : memref<1x1x3072xf32>
}
