// RUN: akg-opt --npu-auto-tiling -split-input-file %s | FileCheck %s

// Non-innermost permutation only swaps the middle axes, so transpose marking must stay off.
// CHECK-LABEL: func.func @transpose_middle_axes_no_mark(
// CHECK-NOT: } {transpose
// CHECK: } {map_for_to_forall}
// CHECK: return %arg1 : memref<4x512x12x64xf32>
module {
  func.func @transpose_middle_axes_no_mark(%arg0: memref<48x512x64xf32>, %arg1: memref<4x512x12x64xf32>) -> memref<4x512x12x64xf32> attributes {OperatorType = "Transpose"} {
    %c64 = arith.constant 64 : index
    %c512 = arith.constant 512 : index
    %c12 = arith.constant 12 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %expand_shape = memref.expand_shape %arg0 [[0, 1], [2], [3]] output_shape [4, 12, 512, 64] : memref<48x512x64xf32> into memref<4x12x512x64xf32>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c12 step %c1 {
        scf.for %arg4 = %c0 to %c512 step %c1 {
          scf.for %arg5 = %c0 to %c64 step %c1 {
            %0 = memref.load %expand_shape[%arg2, %arg3, %arg4, %arg5] : memref<4x12x512x64xf32>
            memref.store %0, %arg1[%arg2, %arg4, %arg3, %arg5] : memref<4x512x12x64xf32>
          }
        }
      }
    }
    return %arg1 : memref<4x512x12x64xf32>
  }
}

// -----

// Full 2D transpose changes the innermost axis, so transpose marking must be kept.
// CHECK-LABEL: func.func @transpose_2d_mark(
// CHECK: } {transpose
// CHECK: } {map_for_to_forall}
// CHECK: return %arg1 : memref<64x32xf32>
module {
  func.func @transpose_2d_mark(%arg0: memref<32x64xf32>, %arg1: memref<64x32xf32>) -> memref<64x32xf32> attributes {OperatorType = "Transpose"} {
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c32 step %c1 {
      scf.for %arg3 = %c0 to %c64 step %c1 {
        %0 = memref.load %arg0[%arg2, %arg3] : memref<32x64xf32>
        memref.store %0, %arg1[%arg3, %arg2] : memref<64x32xf32>
      }
    }
    return %arg1 : memref<64x32xf32>
  }
}

// -----

// Stable prefix + last-two-axis swap still changes the band innermost loop, so transpose must be marked.
// CHECK-LABEL: func.func @transpose_last_two_axes_mark(
// CHECK: } {transpose
// CHECK: } {map_for_to_forall}
// CHECK: return %arg1 : memref<4x12x64x32xf32>
module {
  func.func @transpose_last_two_axes_mark(%arg0: memref<4x12x32x64xf32>, %arg1: memref<4x12x64x32xf32>) -> memref<4x12x64x32xf32> attributes {OperatorType = "Transpose"} {
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c12 = arith.constant 12 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c12 step %c1 {
        scf.for %arg4 = %c0 to %c32 step %c1 {
          scf.for %arg5 = %c0 to %c64 step %c1 {
            %0 = memref.load %arg0[%arg2, %arg3, %arg4, %arg5] : memref<4x12x32x64xf32>
            memref.store %0, %arg1[%arg2, %arg3, %arg5, %arg4] : memref<4x12x64x32xf32>
          }
        }
      }
    }
    return %arg1 : memref<4x12x64x32xf32>
  }
}

// -----

// Valueless non-mapforall broadcast tags should be cleaned in post-processing.
// CHECK-LABEL: func.func @clear_valueless_broadcast_attr(
// CHECK-NOT: } {.*broadcast
// CHECK: } {map_for_to_forall}
// CHECK: return %arg1 : memref<4x12x64xf32>
module {
  func.func @clear_valueless_broadcast_attr(%arg0: memref<4x12x64xf32>, %arg1: memref<4x12x64xf32>) -> memref<4x12x64xf32> attributes {OperatorType = "Elementwise"} {
    %c64 = arith.constant 64 : index
    %c12 = arith.constant 12 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg2 = %c0 to %c4 step %c1 {
      scf.for %arg3 = %c0 to %c12 step %c1 {
        scf.for %arg4 = %c0 to %c64 step %c1 {
          %0 = memref.load %arg0[%arg2, %arg3, %arg4] : memref<4x12x64xf32>
          memref.store %0, %arg1[%arg2, %arg3, %arg4] : memref<4x12x64xf32>
        }
      } {broadcast}
    }
    return %arg1 : memref<4x12x64xf32>
  }
}
