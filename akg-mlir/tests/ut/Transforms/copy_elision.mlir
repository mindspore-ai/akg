// RUN: akg-opt --copy-elision -allow-unregistered-dialect -split-input-file %s | FileCheck %s --check-prefix=CHECK1
// RUN: akg-opt --copy-elision -allow-unregistered-dialect -split-input-file %s | FileCheck %s --check-prefix=CHECK2
// RUN: akg-opt --copy-elision -allow-unregistered-dialect -split-input-file %s | FileCheck %s --check-prefix=CHECK3
// RUN: akg-opt --copy-elision -allow-unregistered-dialect -split-input-file %s | FileCheck %s --check-prefix=CHECK4
// RUN: akg-opt --copy-elision -allow-unregistered-dialect -split-input-file %s | FileCheck %s --check-prefix=CHECK5
// RUN: akg-opt --copy-elision -allow-unregistered-dialect -split-input-file %s | FileCheck %s --check-prefix=CHECK6

// Test CopyElisionPass with multiple scenarios
//===----------------------------------------------------------------------===//

// -----

// Test 1: Basic copy elision with same layout
// CHECK1-LABEL: func.func @test_basic_copy
// CHECK1-NOT: memref.copy
// CHECK1: memref.alloc
// CHECK1-NOT: memref.alloc
// CHECK1: return

func.func @test_basic_copy() -> memref<10x20xf32> {
  %c0 = arith.constant 0.0 : f32
  %alloc_src = memref.alloc() : memref<10x20xf32>
  %alloc_dst = memref.alloc() : memref<10x20xf32>
  
  // Initialize source
  affine.for %i = 0 to 10 {
    affine.for %j = 0 to 20 {
      affine.store %c0, %alloc_src[%i, %j] : memref<10x20xf32>
    }
  }
  
  // Copy operation
  memref.copy %alloc_src, %alloc_dst : memref<10x20xf32> to memref<10x20xf32>
  
  return %alloc_dst : memref<10x20xf32>
}

// -----

// Test 2: Different layouts (stride vs no stride) - should insert cast
// CHECK2-LABEL: func.func @test_stride_layout
// CHECK2: memref.cast
// CHECK2-NOT: memref.copy
// CHECK2: return

func.func @test_stride_layout(%arg0: tensor<5x10xf32>) -> memref<5x10xf32> {
  %0 = bufferization.to_memref %arg0 : memref<5x10xf32, strided<[10, 1], offset: ?>>
  %alloc_dst = memref.alloc() : memref<5x10xf32>
  
  // Copy from strided memref to non-strided memref
  memref.copy %0, %alloc_dst : memref<5x10xf32, strided<[10, 1], offset: ?>> to memref<5x10xf32>
  
  return %alloc_dst : memref<5x10xf32>
}

// -----

// Test 3: Copy elision with collapse_shape operation
// CHECK3-LABEL: func.func @test_with_collapse
// CHECK3: memref.cast
// CHECK3: memref.collapse_shape
// CHECK3-NOT: memref.copy
// CHECK3: return

func.func @test_with_collapse(%arg0: tensor<1x1x100xf32>) -> memref<100xf32> {
  %0 = bufferization.to_memref %arg0 : memref<1x1x100xf32, strided<[100, 100, 1], offset: ?>>
  %alloc_temp = memref.alloc() : memref<1x1x100xf32>
  
  // Copy operation
  memref.copy %0, %alloc_temp : memref<1x1x100xf32, strided<[100, 100, 1], offset: ?>> to memref<1x1x100xf32>
  
  // collapse_shape operation
  %collapsed = memref.collapse_shape %alloc_temp [[0, 1, 2]] : memref<1x1x100xf32> into memref<100xf32>
  
  return %collapsed : memref<100xf32>
}

// -----

// Test 4: Multiple copy operations
// CHECK4-LABEL: func.func @test_multiple_copies
// CHECK4: memref.cast
// CHECK4-NOT: memref.copy
// CHECK4: return

func.func @test_multiple_copies(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> (memref<8x8xf32>, memref<8x8xf32>) {
  %0 = bufferization.to_memref %arg0 : memref<8x8xf32, strided<[8, 1], offset: ?>>
  %1 = bufferization.to_memref %arg1 : memref<8x8xf32, strided<[8, 1], offset: ?>>
  
  %alloc1 = memref.alloc() : memref<8x8xf32>
  %alloc2 = memref.alloc() : memref<8x8xf32>
  %alloc3 = memref.alloc() : memref<8x8xf32>
  
  // First copy - should be eliminated
  memref.copy %0, %alloc1 : memref<8x8xf32, strided<[8, 1], offset: ?>> to memref<8x8xf32>
  
  // Some computation using alloc1
  %c1 = arith.constant 1.0 : f32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %val = affine.load %alloc1[%i, %j] : memref<8x8xf32>
      %new_val = arith.addf %val, %c1 : f32
      affine.store %new_val, %alloc1[%i, %j] : memref<8x8xf32>
    }
  }
  
  // Second copy
  memref.copy %alloc1, %alloc2 : memref<8x8xf32> to memref<8x8xf32>
  
  // Third copy - from strided to non-strided
  memref.copy %1, %alloc3 : memref<8x8xf32, strided<[8, 1], offset: ?>> to memref<8x8xf32>
  
  return %alloc2, %alloc3 : memref<8x8xf32>, memref<8x8xf32>
}


// -----

// Test 5: Complex scenario
// CHECK5-LABEL: func.func @test_complex_scenario
// CHECK5: memref.cast
// CHECK5: memref.collapse_shape
// CHECK5-NOT: memref.copy
// CHECK5: return

func.func @test_complex_scenario(
  %arg0: tensor<1x5100x3072xbf16>, %arg1: tensor<1x5100x3072xbf16>, %arg2: tensor<1x5100x3072xbf16>, %arg3: tensor<1x1x3072xbf16>,
  %arg4: tensor<1x5100x3072xbf16>) -> (tensor<1x5100x3072xbf16>, tensor<1x1x3072xf32>, tensor<1x5100x3072xf32>) attributes {OperatorType = "Reduce",
  compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = bufferization.to_memref %arg3 : memref<1x1x3072xbf16, strided<[?, ?, ?], offset: ?>>
  %1 = bufferization.to_memref %arg4 : memref<1x5100x3072xbf16, strided<[?, ?, ?], offset: ?>>
  %2 = bufferization.to_memref %arg2 : memref<1x5100x3072xbf16, strided<[?, ?, ?], offset: ?>>
  %3 = bufferization.to_memref %arg1 : memref<1x5100x3072xbf16, strided<[?, ?, ?], offset: ?>>
  %4 = bufferization.to_memref %arg0 : memref<1x5100x3072xbf16, strided<[?, ?, ?], offset: ?>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x5100x3072xbf16>
  affine.for %arg5 = 0 to 5100 {
    affine.for %arg6 = 0 to 3072 {
      %8 = affine.load %4[0, %arg5, %arg6] : memref<1x5100x3072xbf16, strided<[?, ?, ?], offset: ?>>
      %9 = affine.load %3[0, %arg5, %arg6] : memref<1x5100x3072xbf16, strided<[?, ?, ?], offset: ?>>
      %10 = arith.addf %8, %9 : bf16
      affine.store %10, %alloc[0, %arg5, %arg6] : memref<1x5100x3072xbf16>
    }
  }
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x5100x3072xbf16>
  affine.for %arg5 = 0 to 5100 {
    affine.for %arg6 = 0 to 3072 {
      %8 = affine.load %alloc[0, %arg5, %arg6] : memref<1x5100x3072xbf16>
      %9 = affine.load %2[0, %arg5, %arg6] : memref<1x5100x3072xbf16, strided<[?, ?, ?], offset: ?>>
      %10 = arith.addf %8, %9 : bf16
      affine.store %10, %alloc_0[0, %arg5, %arg6] : memref<1x5100x3072xbf16>
    }
  }
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x1x3072xbf16>
  memref.copy %0, %alloc_1 : memref<1x1x3072xbf16, strided<[?, ?, ?], offset: ?>> to memref<1x1x3072xbf16>
  %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1, 2]] : memref<1x1x3072xbf16> into memref<3072xbf16>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x5100x3072xbf16>
  affine.for %arg5 = 0 to 5100 {
    affine.for %arg6 = 0 to 3072 {
      %8 = affine.load %collapse_shape[%arg6] : memref<3072xbf16>
      affine.store %8, %alloc_2[0, %arg5, %arg6] : memref<1x5100x3072xbf16>
    }
  }
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x5100x3072xbf16>
  affine.for %arg5 = 0 to 5100 {
    affine.for %arg6 = 0 to 3072 {
      %8 = affine.load %alloc_2[0, %arg5, %arg6] : memref<1x5100x3072xbf16>
      %9 = affine.load %alloc_0[0, %arg5, %arg6] : memref<1x5100x3072xbf16>
      %10 = arith.mulf %8, %9 : bf16
      affine.store %10, %alloc_3[0, %arg5, %arg6] : memref<1x5100x3072xbf16>
    }
  }
  %5 = bufferization.to_tensor %alloc_3 : memref<1x5100x3072xbf16>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x5100x3072xf32>
  affine.for %arg5 = 0 to 5100 {
    affine.for %arg6 = 0 to 3072 {
      %8 = affine.load %alloc_0[0, %arg5, %arg6] : memref<1x5100x3072xbf16>
      %9 = arith.extf %8 : bf16 to f32
      affine.store %9, %alloc_4[0, %arg5, %arg6] : memref<1x5100x3072xf32>
    }
  }
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x3072xf32>
  affine.for %arg5 = 0 to 3072 {
    affine.store %cst, %alloc_5[0, %arg5] : memref<1x3072xf32>
  }
  affine.for %arg5 = 0 to 5100 {
    affine.for %arg6 = 0 to 3072 {
      %8 = affine.load %alloc_4[0, %arg5, %arg6] : memref<1x5100x3072xf32>
      %9 = affine.load %alloc_5[0, %arg6] : memref<1x3072xf32>
      %10 = arith.addf %8, %9 {reduction_axes = [1 : index], reduction_type = "y"} : f32
      affine.store %10, %alloc_5[0, %arg6] : memref<1x3072xf32>
    }
  } {reduceLoop}
  %expand_shape = memref.expand_shape %alloc_5 [[0], [1, 2]] output_shape [1, 1, 3072] : memref<1x3072xf32> into memref<1x1x3072xf32>
  %6 = bufferization.to_tensor %expand_shape : memref<1x1x3072xf32>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<1x5100x3072xbf16>
  affine.for %arg5 = 0 to 5100 {
    affine.for %arg6 = 0 to 3072 {
      %8 = affine.load %1[0, %arg5, %arg6] : memref<1x5100x3072xbf16, strided<[?, ?, ?], offset: ?>>
      %9 = affine.load %alloc_0[0, %arg5, %arg6] : memref<1x5100x3072xbf16>
      %10 = arith.mulf %8, %9 : bf16
      affine.store %10, %alloc_6[0, %arg5, %arg6] : memref<1x5100x3072xbf16>
    }
  }
  %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<1x5100x3072xf32>
  affine.for %arg5 = 0 to 5100 {
    affine.for %arg6 = 0 to 3072 {
      %8 = affine.load %alloc_6[0, %arg5, %arg6] : memref<1x5100x3072xbf16>
      %9 = arith.extf %8 : bf16 to f32
      affine.store %9, %alloc_7[0, %arg5, %arg6] : memref<1x5100x3072xf32>
    }
  }
  %7 = bufferization.to_tensor %alloc_7 : memref<1x5100x3072xf32>
  return %5, %6, %7 : tensor<1x5100x3072xbf16>, tensor<1x1x3072xf32>, tensor<1x5100x3072xf32>
} 

// -----

// Test 6: Nested loops with copy elision
// CHECK6-LABEL: func.func @test_nested_loop
// CHECK6-NOT: memref.copy
// CHECK6: return

func.func @test_nested_loop() -> memref<16x16xf32> {
  %alloc_src = memref.alloc() : memref<16x16xf32>
  %alloc_dst = memref.alloc() : memref<16x16xf32>
  %c1 = arith.constant 1.0 : f32
  
  // Initialize source
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 16 {
      affine.store %c1, %alloc_src[%i, %j] : memref<16x16xf32>
    }
  }
  
  // Copy in nested loop context
  memref.copy %alloc_src, %alloc_dst : memref<16x16xf32> to memref<16x16xf32>
  
  // Use destination after copy
  affine.for %i = 0 to 16 {
    affine.for %j = 0 to 16 {
      %val = affine.load %alloc_dst[%i, %j] : memref<16x16xf32>
      %new_val = arith.mulf %val, %c1 : f32
      affine.store %new_val, %alloc_dst[%i, %j] : memref<16x16xf32>
    }
  }
  
  return %alloc_dst : memref<16x16xf32>
}
