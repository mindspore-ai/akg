// RUN: akg-opt %s --bf16-to-f32 | FileCheck %s

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

func.func @test_load_bf16(%arg0: memref<4xbf16>) -> memref<4xbf16> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xbf16>
  affine.for %arg1 = 0 to 4 {
    %0 = affine.load %arg0[%arg1] : memref<4xbf16>
    %1 = arith.addf %0, %0 : bf16
    affine.store %1, %alloc[%arg1] : memref<4xbf16>
  }
  return %alloc : memref<4xbf16>
}

// CHECK-LABEL:   func.func @test_bf16_f32(%arg0: memref<4xbf16>) -> memref<4xf32> {
// CHECK:           %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xf32>
// CHECK:           affine.for %arg1 = 0 to 4 {
// CHECK:             %0 = affine.load %arg0[%arg1] : memref<4xbf16>
// CHECK:             %1 = arith.extf %0 : bf16 to f32
// CHECK:             %2 = arith.addf %1, %1 : f32
// CHECK:             affine.store %2, %alloc[%arg1] : memref<4xf32>
// CHECK:           }
// CHECK:           return %alloc : memref<4xf32>
// CHECK:         }
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

// CHECK-LABEL:   func.func @test_f32_bf16(%arg0: memref<4xf32>) -> memref<4xbf16> {
// CHECK:           %alloc = memref.alloc() {alignment = 64 : i64} : memref<4xbf16>
// CHECK:           affine.for %arg1 = 0 to 4 {
// CHECK:             %0 = affine.load %arg0[%arg1] : memref<4xf32>
// CHECK:             %1 = arith.addf %0, %0 : f32
// CHECK:             %2 = arith.truncf %1 : f32 to bf16
// CHECK:             affine.store %2, %alloc[%arg1] : memref<4xbf16>
// CHECK:           }
// CHECK:           return %alloc : memref<4xbf16>
// CHECK:         }
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
