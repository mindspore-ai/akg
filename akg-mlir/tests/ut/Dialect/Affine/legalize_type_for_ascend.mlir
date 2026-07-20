// RUN: akg-opt %s --legalize-type-for-ascend | FileCheck %s
// RUN: akg-opt %s --legalize-type-for-ascend="enable-i64-to-i32=true" | FileCheck %s --check-prefix=I64

// CHECK-LABEL: func.func @test_load_bf16(%arg0: memref<4xbf16>) -> memref<4xbf16> {
// CHECK:         %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xbf16>
// CHECK:         affine.for %arg1 = 0 to 4 {
// CHECK:           %0 = affine.load %arg0[%arg1] : memref<4xbf16>
// CHECK:           %1 = arith.extf %0 : bf16 to f32
// CHECK:           %2 = arith.addf %1, %1 : f32
// CHECK:           %3 = arith.truncf %2 : f32 to bf16
// CHECK:           affine.store %3, %alloc[%arg1] : memref<4xbf16>
// CHECK:         }
// CHECK:         return %alloc : memref<4xbf16>
// CHECK:       }
// I64-LABEL: func.func @test_load_bf16(%arg0: memref<4xbf16>) -> memref<4xbf16> {
// I64:         %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xbf16>
// I64:         affine.for %arg1 = 0 to 4 {
// I64:           %0 = affine.load %arg0[%arg1] : memref<4xbf16>
// I64:           %1 = arith.extf %0 : bf16 to f32
// I64:           %2 = arith.addf %1, %1 : f32
// I64:           %3 = arith.truncf %2 : f32 to bf16
// I64:           affine.store %3, %alloc[%arg1] : memref<4xbf16>
// I64:         }
// I64:         return %alloc : memref<4xbf16>
// I64:       }
func.func @test_load_bf16(%arg0: memref<4xbf16>) -> memref<4xbf16> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xbf16>
  affine.for %arg1 = 0 to 4 {
    %0 = affine.load %arg0[%arg1] : memref<4xbf16>
    %1 = arith.addf %0, %0 : bf16
    affine.store %1, %alloc[%arg1] : memref<4xbf16>
  }
  return %alloc : memref<4xbf16>
}

// CHECK-LABEL: func.func @test_bf16_f32(%arg0: memref<4xbf16>) -> memref<4xf32> {
// CHECK:         %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
// CHECK:         affine.for %arg1 = 0 to 4 {
// CHECK:           %0 = affine.load %arg0[%arg1] : memref<4xbf16>
// CHECK:           %1 = arith.extf %0 : bf16 to f32
// CHECK:           %2 = arith.addf %1, %1 : f32
// CHECK:           affine.store %2, %alloc[%arg1] : memref<4xf32>
// CHECK:         }
// CHECK:         return %alloc : memref<4xf32>
// CHECK:       }
// I64-LABEL: func.func @test_bf16_f32(%arg0: memref<4xbf16>) -> memref<4xf32> {
// I64:         %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
// I64:         affine.for %arg1 = 0 to 4 {
// I64:           %0 = affine.load %arg0[%arg1] : memref<4xbf16>
// I64:           %1 = arith.extf %0 : bf16 to f32
// I64:           %2 = arith.addf %1, %1 : f32
// I64:           affine.store %2, %alloc[%arg1] : memref<4xf32>
// I64:         }
// I64:         return %alloc : memref<4xf32>
// I64:       }
func.func @test_bf16_f32(%arg0: memref<4xbf16>) -> memref<4xf32> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
  affine.for %arg1 = 0 to 4 {
    %0 = affine.load %arg0[%arg1] : memref<4xbf16>
    %1 = arith.addf %0, %0 : bf16
    %2 = arith.extf %1 : bf16 to f32
    affine.store %2, %alloc[%arg1] : memref<4xf32>
  }
  return %alloc : memref<4xf32>
}

// CHECK-LABEL: func.func @test_f32_bf16(%arg0: memref<4xf32>) -> memref<4xbf16> {
// CHECK:         %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xbf16>
// CHECK:         affine.for %arg1 = 0 to 4 {
// CHECK:           %0 = affine.load %arg0[%arg1] : memref<4xf32>
// CHECK:           %1 = arith.addf %0, %0 : f32
// CHECK:           %2 = arith.truncf %1 : f32 to bf16
// CHECK:           affine.store %2, %alloc[%arg1] : memref<4xbf16>
// CHECK:         }
// CHECK:         return %alloc : memref<4xbf16>
// CHECK:       }
// I64-LABEL: func.func @test_f32_bf16(%arg0: memref<4xf32>) -> memref<4xbf16> {
// I64:         %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xbf16>
// I64:         affine.for %arg1 = 0 to 4 {
// I64:           %0 = affine.load %arg0[%arg1] : memref<4xf32>
// I64:           %1 = arith.addf %0, %0 : f32
// I64:           %2 = arith.truncf %1 : f32 to bf16
// I64:           affine.store %2, %alloc[%arg1] : memref<4xbf16>
// I64:         }
// I64:         return %alloc : memref<4xbf16>
// I64:       }
func.func @test_f32_bf16(%arg0: memref<4xf32>) -> memref<4xbf16> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xbf16>
  affine.for %arg1 = 0 to 4 {
    %0 = affine.load %arg0[%arg1] : memref<4xf32>
    %1 = arith.addf %0, %0 : f32
    %2 = arith.truncf %1 : f32 to bf16
    affine.store %2, %alloc[%arg1] : memref<4xbf16>
  }
  return %alloc : memref<4xbf16>
}

// I64-LABEL: func.func @test_load_i64_add(%arg0: memref<4xi64>) -> memref<4xi64> {
// I64:         %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
// I64:         affine.for %arg1 = 0 to 4 {
// I64:           %0 = affine.load %arg0[%arg1] : memref<4xi64>
// I64:           %1 = arith.trunci %0 : i64 to i32
// I64:           %2 = arith.addi %1, %1 : i32
// I64:           %3 = arith.extsi %2 : i32 to i64
// I64:           affine.store %3, %alloc[%arg1] : memref<4xi64>
// I64:         }
// I64:         return %alloc : memref<4xi64>
// I64:       }
func.func @test_load_i64_add(%arg0: memref<4xi64>) -> memref<4xi64> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
  affine.for %arg1 = 0 to 4 {
    %0 = affine.load %arg0[%arg1] : memref<4xi64>
    %1 = arith.addi %0, %0 : i64
    affine.store %1, %alloc[%arg1] : memref<4xi64>
  }
  return %alloc : memref<4xi64>
}

// I64-LABEL: func.func @test_i64_constant(%arg0: memref<4xi64>) -> memref<4xi64> {
// I64:         %c1_i32 = arith.constant 1 : i32
// I64:         %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
// I64:         affine.for %arg1 = 0 to 4 {
// I64:           %0 = affine.load %arg0[%arg1] : memref<4xi64>
// I64:           %1 = arith.trunci %0 : i64 to i32
// I64:           %2 = arith.addi %1, %c1_i32 : i32
// I64:           %3 = arith.extsi %2 : i32 to i64
// I64:           affine.store %3, %alloc[%arg1] : memref<4xi64>
// I64:         }
// I64:         return %alloc : memref<4xi64>
// I64:       }
func.func @test_i64_constant(%arg0: memref<4xi64>) -> memref<4xi64> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
  affine.for %arg1 = 0 to 4 {
    %0 = affine.load %arg0[%arg1] : memref<4xi64>
    %c1 = arith.constant 1 : i64
    %1 = arith.addi %0, %c1 : i64
    affine.store %1, %alloc[%arg1] : memref<4xi64>
  }
  return %alloc : memref<4xi64>
}

// I64-LABEL: func.func @test_i64_for_iter_arg(%arg0: memref<4xi64>, %arg1: memref<1xi64>) -> memref<1xi64> {
// I64:         %c0 = arith.constant 0 : index
// I64:         %c0_i32 = arith.constant 0 : i32
// I64:         affine.for %arg2 = 0 to 4 iter_args(%arg3 = %c0_i32) -> (i32) {
// I64:           %2 = affine.load %arg0[%arg2] : memref<4xi64>
// I64:           %3 = arith.trunci %2 : i64 to i32
// I64:           %4 = arith.addi %arg3, %3 : i32
// I64:           affine.yield %4 : i32
// I64:         }
// I64:         %1 = arith.extsi %0 : i32 to i64
// I64:         affine.store %1, %arg1[%c0] : memref<1xi64>
// I64:         return %arg1 : memref<1xi64>
// I64:       }
func.func @test_i64_for_iter_arg(%arg0: memref<4xi64>, %arg1: memref<1xi64>) -> memref<1xi64> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = affine.for %arg2 = 0 to 4 iter_args(%arg3 = %c0_i64) -> i64 {
    %1 = affine.load %arg0[%arg2] : memref<4xi64>
    %2 = arith.addi %arg3, %1 : i64
    affine.yield %2 : i64
  }
  affine.store %0, %arg1[%c0] : memref<1xi64>
  return %arg1 : memref<1xi64>
}

// I64-LABEL: func.func @test_i8_to_i64_with_compute(%arg0: memref<4xi8>, %arg1: memref<4xi64>) -> memref<4xi64> {
// I64:         %c10_i32 = arith.constant 10 : i32
// I64:         %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
// I64:         affine.for %arg2 = 0 to 4 {
// I64:           %0 = affine.load %arg0[%arg2] : memref<4xi8>
// I64:           %1 = arith.sitofp %0 : i8 to f16
// I64:           %2 = arith.fptosi %1 : f16 to i32
// I64:           %3 = affine.load %arg1[%arg2] : memref<4xi64>
// I64:           %4 = arith.trunci %3 : i64 to i32
// I64:           %5 = arith.addi %2, %4 : i32
// I64:           %6 = arith.muli %5, %c10_i32 : i32
// I64:           %7 = arith.extsi %6 : i32 to i64
// I64:           affine.store %7, %alloc[%arg2] : memref<4xi64>
// I64:         }
// I64:         return %alloc : memref<4xi64>
// I64:       }
func.func @test_i8_to_i64_with_compute(%arg0: memref<4xi8>, %arg1: memref<4xi64>) -> memref<4xi64> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xi64>
  affine.for %arg2 = 0 to 4 {
    %0 = affine.load %arg0[%arg2] : memref<4xi8>
    %1 = arith.extsi %0 : i8 to i64
    %2 = affine.load %arg1[%arg2] : memref<4xi64>
    %3 = arith.addi %1, %2 : i64
    %c10 = arith.constant 10 : i64
    %4 = arith.muli %3, %c10 : i64
    affine.store %4, %alloc[%arg2] : memref<4xi64>
  }
  return %alloc : memref<4xi64>
}
