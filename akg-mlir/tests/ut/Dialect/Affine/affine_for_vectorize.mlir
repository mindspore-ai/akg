// RUN: akg-opt %s --affine-for-vectorize | FileCheck %s

// Dynamic

// CHECK-LABEL:  func.func @vector_add_2d(%arg0: index, %arg1: index) -> f32 {
// CHECK-NEXT:     %alloc = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
// CHECK-NEXT:     %alloc_0 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
// CHECK-NEXT:     %alloc_1 = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
// CHECK-NEXT:     %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:     %cst_2 = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:     affine.for %arg2 = 0 to %arg0 {
// CHECK-NEXT:       affine.for %arg3 = 0 to %arg1 step 512 {
// CHECK-NEXT:         %cst_3 = arith.constant dense<1.000000e+00> : vector<512xf32>
// CHECK-NEXT:         vector.transfer_write %cst_3, %alloc[%arg2, %arg3] : vector<512xf32>, memref<?x?xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg2 = 0 to %arg0 {
// CHECK-NEXT:       affine.for %arg3 = 0 to %arg1 step 512 {
// CHECK-NEXT:         %cst_3 = arith.constant dense<2.000000e+00> : vector<512xf32>
// CHECK-NEXT:         vector.transfer_write %cst_3, %alloc_0[%arg2, %arg3] : vector<512xf32>, memref<?x?xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.for %arg2 = 0 to %arg0 {
// CHECK-NEXT:       affine.for %arg3 = 0 to %arg1 step 512 {
// CHECK-NEXT:         %cst_3 = arith.constant dense<2.000000e+00> : vector<512xf32>
// CHECK-NEXT:         %cst_4 = arith.constant dense<1.000000e+00> : vector<512xf32>
// CHECK-NEXT:         %cst_5 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:         %1 = vector.transfer_read %alloc[%arg2, %arg3], %cst_5 : memref<?x?xf32>, vector<512xf32>
// CHECK-NEXT:         %cst_6 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:         %2 = vector.transfer_read %alloc_0[%arg2, %arg3], %cst_6 : memref<?x?xf32>, vector<512xf32>
// CHECK-NEXT:         %3 = arith.addf %1, %2 : vector<512xf32>
// CHECK-NEXT:         %4 = arith.addf %3, %cst_4 : vector<512xf32>
// CHECK-NEXT:         %5 = arith.addf %3, %cst_3 : vector<512xf32>
// CHECK-NEXT:         %6 = arith.addf %5, %4 : vector<512xf32>
// CHECK-NEXT:         vector.transfer_write %6, %alloc_1[%arg2, %arg3] : vector<512xf32>, memref<?x?xf32>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     %c7 = arith.constant 7 : index
// CHECK-NEXT:     %c42 = arith.constant 42 : index
// CHECK-NEXT:     %0 = affine.load %alloc_1[%c7, %c42] : memref<?x?xf32>
// CHECK-NEXT:     return %0 : f32
// CHECK-NEXT:   }


func.func @vector_add_2d(%M : index, %N : index) -> f32 {
  %A = memref.alloc (%M, %N) : memref<?x?xf32, 0>
  %B = memref.alloc (%M, %N) : memref<?x?xf32, 0>
  %C = memref.alloc (%M, %N) : memref<?x?xf32, 0>
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  affine.for %i0 = 0 to %M {
    affine.for %i1 = 0 to %N {
      affine.store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
    }
  }
  affine.for %i2 = 0 to %M {
    affine.for %i3 = 0 to %N {
      affine.store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
    }
  }
  affine.for %i4 = 0 to %M {
    affine.for %i5 = 0 to %N {
      %a5 = affine.load %A[%i4, %i5] : memref<?x?xf32, 0>
      %b5 = affine.load %B[%i4, %i5] : memref<?x?xf32, 0>
      %s5 = arith.addf %a5, %b5 : f32
      %s6 = arith.addf %s5, %f1 : f32
      %s7 = arith.addf %s5, %f2 : f32
      %s8 = arith.addf %s7, %s6 : f32
      affine.store %s8, %C[%i4, %i5] : memref<?x?xf32, 0>
    }
  }
  %c7 = arith.constant 7 : index
  %c42 = arith.constant 42 : index
  %res = affine.load %C[%c7, %c42] : memref<?x?xf32, 0>
  return %res : f32
}


// --------------------------------------


// reduction

// CHECK-LABEL:  func.func @vecdim_reduction_2d(%arg0: memref<256x512x1024xf32>, %arg1: memref<256xf32>) {
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     affine.for %arg2 = 0 to 256 {
// CHECK-NEXT:       %0 = affine.for %arg3 = 0 to 512 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:         %cst_0 = arith.constant dense<0.000000e+00> : vector<1024xf32>
// CHECK-NEXT:         %1 = affine.for %arg5 = 0 to 1024 step 1024 iter_args(%arg6 = %cst_0) -> (vector<1024xf32>) {
// CHECK-NEXT:           %cst_1 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:           %4 = vector.transfer_read %arg0[%arg2, %arg3, %arg5], %cst_1 : memref<256x512x1024xf32>, vector<1024xf32>
// CHECK-NEXT:           %5 = arith.addf %arg6, %4 : vector<1024xf32>
// CHECK-NEXT:           affine.yield %5 : vector<1024xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %2 = vector.reduction <add>, %1 : vector<1024xf32> into f32
// CHECK-NEXT:         %3 = arith.addf %arg4, %2 : f32
// CHECK-NEXT:         affine.yield %3 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.store %0, %arg1[%arg2] : memref<256xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


func.func @vecdim_reduction_2d(%in: memref<256x512x1024xf32>, %out: memref<256xf32>) {
 %cst = arith.constant 0.000000e+00 : f32
 affine.for %i = 0 to 256 {
   %sum_j = affine.for %j = 0 to 512 iter_args(%red_iter_j = %cst) -> (f32) {
     %sum_k = affine.for %k = 0 to 1024 iter_args(%red_iter_k = %cst) -> (f32) {
       %ld = affine.load %in[%i, %j, %k] : memref<256x512x1024xf32>
       %add = arith.addf %red_iter_k, %ld : f32
       affine.yield %add : f32
     }
     %add = arith.addf %red_iter_j, %sum_k : f32
     affine.yield %add : f32
   }
   affine.store %sum_j, %out[%i] : memref<256xf32>
 }
 return
}