// RUN: akg-opt %s --split-input-file --unify-shape | FileCheck %s


// CHECK-LABEL:   func.func @collapse_arg(
// CHECK-SAME:                            %[[VAL_0:.*]]: memref<28224x4xf32>,
// CHECK-SAME:                            %[[VAL_1:.*]]: memref<28224x4xf32>) attributes {OperatorType = "Reshape", mindspore_kernel} {
// CHECK-NEXT:      memref.copy %[[VAL_0]], %[[VAL_1]] : memref<28224x4xf32> to memref<28224x4xf32>
// CHECK-NEXT:      return
func.func @collapse_arg(%arg0: memref<144x196x4xf32>, %arg1: memref<28224x4xf32>) attributes {OperatorType = "Reshape", mindspore_kernel} {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<144x196x4xf32> into memref<28224x4xf32>
  memref.copy %collapse_shape, %arg1 : memref<28224x4xf32> to memref<28224x4xf32>
  return
}

// -----

// CHECK-LABEL:   func.func @expand_arg(
// CHECK-SAME:                          %[[VAL_0:.*]]: memref<2016x14x2x2xf32>,
// CHECK-SAME:                          %[[VAL_1:.*]]: memref<2016x14x2x2xf32>) attributes {OperatorType = "Reshape", mindspore_kernel} {
// CHECK-NEXT:      memref.copy %[[VAL_0]], %[[VAL_1]] : memref<2016x14x2x2xf32> to memref<2016x14x2x2xf32>
// CHECK-NEXT:      return
func.func @expand_arg(%arg0: memref<28224x4xf32>, %arg1: memref<2016x14x2x2xf32>) attributes {OperatorType = "Reshape", mindspore_kernel} {
  %expand_shape = memref.expand_shape %arg0 [[0, 1], [2, 3]] : memref<28224x4xf32> into memref<2016x14x2x2xf32>
  memref.copy %expand_shape, %arg1 : memref<2016x14x2x2xf32> to memref<2016x14x2x2xf32>
  return
}

// -----

// CHECK-LABEL:   func.func @collaps_expand_arg(
// CHECK-SAME:                                  %[[VAL_0:.*]]: memref<2016x14x2x2xf32>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: memref<2016x14x2x2xf32>) attributes {OperatorType = "Reshape", mindspore_kernel} {
// CHECK-NEXT:      memref.copy %[[VAL_0]], %[[VAL_1]] : memref<2016x14x2x2xf32> to memref<2016x14x2x2xf32>
// CHECK-NEXT:      return
func.func @collaps_expand_arg(%arg0: memref<144x196x4xf32>, %arg1: memref<2016x14x2x2xf32>) attributes {OperatorType = "Reshape", mindspore_kernel} {
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<144x196x4xf32> into memref<28224x4xf32>
  %expand_shape = memref.expand_shape %collapse_shape [[0, 1], [2, 3]] : memref<28224x4xf32> into memref<2016x14x2x2xf32>
  memref.copy %expand_shape, %arg1 : memref<2016x14x2x2xf32> to memref<2016x14x2x2xf32>
  return
}

// -----

// CHECK-LABEL:   func.func @expand_copy(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<784xf32>, %[[VAL_1:.*]]: memref<784x144xf32>,
// CHECK-SAME:                           %[[VAL_2:.*]]: memref<144xf32>, %[[VAL_3:.*]]: memref<144xf32>,
// CHECK-SAME:                           %[[VAL_4:.*]]: memref<784x144xf32>) attributes {OperatorType = "Reshape", enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
// CHECK-NEXT:      %[[VAL_5:.*]] = arith.constant 9.99999974E-6 : f32
// CHECK-NEXT:      %[[VAL_6:.*]] = arith.constant 0.0069444445 : f32
// CHECK-NEXT:      %[[VAL_7:.*]] = memref.alloc() {alignment = 64 : i64} : memref<784x144xf32>
// CHECK-NEXT:      affine.for %[[VAL_8:.*]] = 0 to 784 {
// CHECK-NEXT:        affine.for %[[VAL_9:.*]] = 0 to 1 {
// CHECK-NEXT:          affine.for %[[VAL_10:.*]] = 0 to 144 {
// CHECK-NEXT:            %[[VAL_11:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_8]], %[[VAL_10]]] : memref<784x144xf32>
// CHECK-NEXT:            %[[VAL_12:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_8]]] : memref<784xf32>
// CHECK-NEXT:            %[[VAL_13:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_10]]] : memref<144xf32>
// CHECK-NEXT:            %[[VAL_14:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_10]]] : memref<144xf32>
// CHECK-NEXT:            %[[VAL_15:.*]] = arith.mulf %[[VAL_12]], %[[VAL_6]] : f32
// CHECK-NEXT:            %[[VAL_16:.*]] = arith.addf %[[VAL_15]], %[[VAL_5]] : f32
// CHECK-NEXT:            %[[VAL_17:.*]] = math.rsqrt %[[VAL_16]] : f32
// CHECK-NEXT:            %[[VAL_18:.*]] = arith.mulf %[[VAL_11]], %[[VAL_17]] : f32
// CHECK-NEXT:            %[[VAL_19:.*]] = arith.mulf %[[VAL_18]], %[[VAL_13]] : f32
// CHECK-NEXT:            %[[VAL_20:.*]] = arith.addf %[[VAL_19]], %[[VAL_14]] : f32
// CHECK-NEXT:            affine.store %[[VAL_20]], %[[VAL_7]]{{\[}}%[[VAL_8]], %[[VAL_10]]] : memref<784x144xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.copy %[[VAL_7]], %[[VAL_4]] : memref<784x144xf32> to memref<784x144xf32>
// CHECK-NEXT:      return
func.func @expand_copy(%arg0: memref<784xf32>, %arg1: memref<784x144xf32>, %arg2: memref<144xf32>, %arg3: memref<144xf32>, %arg4: memref<4x196x144xf32>) attributes {OperatorType = "Reshape", enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
  %cst = arith.constant 9.99999974E-6 : f32
  %cst_0 = arith.constant 0.0069444445 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<784x144xf32>
  affine.for %arg5 = 0 to 784 {
    affine.for %arg6 = 0 to 1 {
      affine.for %arg7 = 0 to 144 {
        %0 = affine.load %arg1[%arg5, %arg7] : memref<784x144xf32>
        %1 = affine.load %arg0[%arg5] : memref<784xf32>
        %2 = affine.load %arg3[%arg7] : memref<144xf32>
        %3 = affine.load %arg2[%arg7] : memref<144xf32>
        %4 = arith.mulf %1, %cst_0 : f32
        %5 = arith.addf %4, %cst : f32
        %6 = math.rsqrt %5 : f32
        %7 = arith.mulf %0, %6 : f32
        %8 = arith.mulf %7, %2 : f32
        %9 = arith.addf %8, %3 : f32
        affine.store %9, %alloc[%arg5, %arg7] : memref<784x144xf32>
      }
    }
  }
  %expand_shape = memref.expand_shape %alloc [[0, 1], [2]] : memref<784x144xf32> into memref<4x196x144xf32>
  memref.copy %expand_shape, %arg4 : memref<4x196x144xf32> to memref<4x196x144xf32>
  return
}

// -----

// CHECK-LABEL:   func.func @collapse_copy(
// CHECK-SAME:                             %[[VAL_0:.*]]: memref<128x3xf32>, %[[VAL_1:.*]]: memref<128x3xf32>, %[[VAL_2:.*]]: memref<128x3xf32>, %[[VAL_3:.*]]: memref<128x3xf32>, %[[VAL_4:.*]]: memref<128x3xf32>, %[[VAL_5:.*]]: memref<128x3xf32>, %[[VAL_6:.*]]: memref<f32>,
// CHECK-SAME:                             %[[VAL_7:.*]]: memref<384xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
// CHECK-NEXT:      %[[VAL_8:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:      %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[VAL_10:.*]] = memref.alloc() {alignment = 64 : i64} : memref<384xf32>
// CHECK-NEXT:      affine.for %[[VAL_11:.*]] = 0 to 128 {
// CHECK-NEXT:        affine.for %[[VAL_12:.*]] = 0 to 3 {
// CHECK-NEXT:          affine.for %[[VAL_13:.*]] = 0 to 1 {
// CHECK-NEXT:            %[[VAL_14:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<128x3xf32>
// CHECK-NEXT:            %[[VAL_15:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<128x3xf32>
// CHECK-NEXT:            %[[VAL_16:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<128x3xf32>
// CHECK-NEXT:            %[[VAL_17:.*]] = affine.load %[[VAL_2]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<128x3xf32>
// CHECK-NEXT:            %[[VAL_18:.*]] = affine.load %[[VAL_4]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<128x3xf32>
// CHECK-NEXT:            %[[VAL_19:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<128x3xf32>
// CHECK-NEXT:            %[[VAL_20:.*]] = arith.mulf %[[VAL_18]], %[[VAL_19]] : f32
// CHECK-NEXT:            %[[VAL_21:.*]] = arith.negf %[[VAL_20]] : f32
// CHECK-NEXT:            %[[VAL_22:.*]] = arith.divf %[[VAL_8]], %[[VAL_16]] : f32
// CHECK-NEXT:            %[[VAL_23:.*]] = arith.mulf %[[VAL_22]], %[[VAL_17]] : f32
// CHECK-NEXT:            %[[VAL_24:.*]] = arith.addf %[[VAL_23]], %[[VAL_21]] : f32
// CHECK-NEXT:            %[[VAL_25:.*]] = arith.divf %[[VAL_8]], %[[VAL_15]] : f32
// CHECK-NEXT:            %[[VAL_26:.*]] = arith.mulf %[[VAL_25]], %[[VAL_24]] : f32
// CHECK-NEXT:            %[[VAL_27:.*]] = arith.mulf %[[VAL_14]], %[[VAL_26]] : f32
// CHECK-NEXT:            affine.store %[[VAL_27]], %[[VAL_10]]{{\[}}%[[VAL_11]] * 3 + %[[VAL_12]]] : memref<384xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[VAL_28:.*]] = memref.alloc() {alignment = 64 : i64} : memref<f32>
// CHECK-NEXT:      affine.store %[[VAL_9]], %[[VAL_28]][] : memref<f32>
// CHECK-NEXT:      affine.for %[[VAL_29:.*]] = 0 to 384 {
// CHECK-NEXT:        %[[VAL_30:.*]] = affine.load %[[VAL_10]]{{\[}}%[[VAL_29]]] : memref<384xf32>
// CHECK-NEXT:        %[[VAL_31:.*]] = affine.load %[[VAL_28]][] : memref<f32>
// CHECK-NEXT:        %[[VAL_32:.*]] = arith.addf %[[VAL_30]], %[[VAL_31]] {reduction_axes = [0 : index], reduction_type = "all"} : f32
// CHECK-NEXT:        affine.store %[[VAL_32]], %[[VAL_28]][] : memref<f32>
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.copy %[[VAL_28]], %[[VAL_6]] : memref<f32> to memref<f32>
// CHECK-NEXT:      memref.copy %[[VAL_10]], %[[VAL_7]] : memref<384xf32> to memref<384xf32>
// CHECK-NEXT:      return
func.func @collapse_copy(%arg0: memref<128x3xf32>, %arg1: memref<128x3xf32>, %arg2: memref<128x3xf32>, %arg3: memref<128x3xf32>, %arg4: memref<128x3xf32>, %arg5: memref<128x3xf32>, %arg6: memref<f32>, %arg7: memref<128x3xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x3xf32>
  affine.for %arg8 = 0 to 128 {
    affine.for %arg9 = 0 to 3 {
      affine.for %arg10 = 0 to 1 {
        %0 = affine.load %arg0[%arg8, %arg9] : memref<128x3xf32>
        %1 = affine.load %arg1[%arg8, %arg9] : memref<128x3xf32>
        %2 = affine.load %arg3[%arg8, %arg9] : memref<128x3xf32>
        %3 = affine.load %arg2[%arg8, %arg9] : memref<128x3xf32>
        %4 = affine.load %arg4[%arg8, %arg9] : memref<128x3xf32>
        %5 = affine.load %arg5[%arg8, %arg9] : memref<128x3xf32>
        %6 = arith.mulf %4, %5 : f32
        %7 = arith.negf %6 : f32
        %8 = arith.divf %cst, %2 : f32
        %9 = arith.mulf %8, %3 : f32
        %10 = arith.addf %9, %7 : f32
        %11 = arith.divf %cst, %1 : f32
        %12 = arith.mulf %11, %10 : f32
        %13 = arith.mulf %0, %12 : f32
        affine.store %13, %alloc[%arg8, %arg9] : memref<128x3xf32>
      }
    }
  }
  %collapse_shape = memref.collapse_shape %alloc [[0, 1]] : memref<128x3xf32> into memref<384xf32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<f32>
  affine.store %cst_0, %alloc_1[] : memref<f32>
  affine.for %arg8 = 0 to 384 {
    %0 = affine.load %collapse_shape[%arg8] : memref<384xf32>
    %1 = affine.load %alloc_1[] : memref<f32>
    %2 = arith.addf %0, %1 {reduction_axes = [0 : index], reduction_type = "all"} : f32
    affine.store %2, %alloc_1[] : memref<f32>
  }
  memref.copy %alloc_1, %arg6 : memref<f32> to memref<f32>
  memref.copy %alloc, %arg7 : memref<128x3xf32> to memref<128x3xf32>
  return
}

// -----

// CHECK-LABEL:   func.func @expand_empty(
// CHECK-SAME:                            %[[VAL_0:.*]]: memref<f32>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<f32>
// CHECK:           affine.store %[[VAL_1]], %[[VAL_2]][] : memref<f32>
// CHECK:           memref.copy %[[VAL_2]], %[[VAL_0]] : memref<f32> to memref<f32>
// CHECK:           return
// CHECK:         }
func.func @expand_empty(%arg0: memref<1xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
  affine.store %cst, %alloc[] : memref<f32>
  %expand_shape = memref.expand_shape %alloc [] : memref<f32> into memref<1xf32>
  memref.copy %expand_shape, %arg0 : memref<1xf32> to memref<1xf32>
  return
}
