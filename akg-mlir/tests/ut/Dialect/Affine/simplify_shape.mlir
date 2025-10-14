// RUN: akg-opt %s --split-input-file --simplify-shape | FileCheck %s

// CHECK-LABEL:   func.func @expand_with_1_alone(
// CHECK-SAME:                                   %[[VAL_0:.*]]: memref<384xf32>,
// CHECK-SAME:                                   %[[VAL_1:.*]]: memref<f32>,
// CHECK-SAME:                                   %[[VAL_2:.*]]: memref<128x3xf32>) attributes {OperatorType = "Reshape", enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.expand_shape %[[VAL_0]] {{\[\[}}0, 1]] : memref<384xf32> into memref<128x3xf32>
// CHECK-NEXT:      %[[VAL_4:.*]] = memref.alloc() {alignment = 64 : i64} : memref<128x3xf32>
// CHECK-NEXT:      affine.for %[[VAL_5:.*]] = 0 to 128 {
// CHECK-NEXT:        affine.for %[[VAL_6:.*]] = 0 to 3 {
// CHECK-NEXT:          affine.for %[[VAL_7:.*]] = 0 to 1 {
// CHECK-NEXT:            %[[VAL_8:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_5]], %[[VAL_6]]] : memref<128x3xf32>
// CHECK-NEXT:            %[[VAL_9:.*]] = affine.load %[[VAL_1]][] : memref<f32>
// CHECK-NEXT:            %[[VAL_10:.*]] = arith.addf %[[VAL_8]], %[[VAL_9]] : f32
// CHECK-NEXT:            affine.store %[[VAL_10]], %[[VAL_4]]{{\[}}%[[VAL_5]], %[[VAL_6]]] : memref<128x3xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.copy %[[VAL_4]], %[[VAL_2]] : memref<128x3xf32> to memref<128x3xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @expand_with_1_alone(%arg0: memref<384x1xf32>, %arg1: memref<1xf32>, %arg2: memref<128x3x1xf32>) attributes {OperatorType = "Reshape", enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
  %expand_shape = memref.expand_shape %arg0 [[0, 1], [2]] : memref<384x1xf32> into memref<128x3x1xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x3x1xf32>
  affine.for %arg3 = 0 to 128 {
    affine.for %arg4 = 0 to 3 {
      affine.for %arg5 = 0 to 1 {
        %0 = affine.load %expand_shape[%arg3, %arg4, %arg5] : memref<128x3x1xf32>
        %1 = affine.load %arg1[%arg5] : memref<1xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc[%arg3, %arg4, %arg5] : memref<128x3x1xf32>
      }
    }
  }
  memref.copy %alloc, %arg2 : memref<128x3x1xf32> to memref<128x3x1xf32>
  return
}

// -----

// CHECK-LABEL: module {
// CHECK-NEXT:    memref.global "private" constant @__constant_1xf32 : memref<f32> = dense<0.000000e+00>
// CHECK-NEXT:    func.func @global(%[[VAL_0:.*]]: memref<f32>) attributes {OperatorType = "Default", enable_atomic_add = false, mindspore_kernel, process = "cuda"} {
// CHECK-NEXT:      %[[VAL_1:.*]] = memref.get_global @__constant_1xf32 : memref<f32>
// CHECK-NEXT:      memref.copy %[[VAL_1]], %[[VAL_0]] : memref<f32> to memref<f32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
memref.global "private" constant @__constant_1xf32 : memref<1xf32> = dense<0.000000e+00>
func.func @global(%arg0: memref<1xf32>) attributes {OperatorType = "Default", enable_atomic_add = false, mindspore_kernel, process = "cuda"} {
  %0 = memref.get_global @__constant_1xf32 : memref<1xf32>
  memref.copy %0, %arg0 : memref<1xf32> to memref<1xf32>
  return
}


// -----

// CHECK-LABEL:   func.func @expand1(
// CHECK-SAME:                       %[[VAL_0:.*]]: memref<3136x196xf32>,
// CHECK-SAME:                       %[[VAL_1:.*]]: memref<3136xf32>) attributes {OperatorType = "Reduce", mindspore_kernel} {
// CHECK-NEXT:      %[[VAL_2:.*]] = arith.constant -3.40282347E+38 : f32
// CHECK-NEXT:      %[[VAL_3:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3136xf32>
// CHECK-NEXT:      affine.for %[[VAL_4:.*]] = 0 to 3136 {
// CHECK-NEXT:        affine.store %[[VAL_2]], %[[VAL_3]]{{\[}}%[[VAL_4]]] : memref<3136xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %[[VAL_5:.*]] = 0 to 3136 {
// CHECK-NEXT:        affine.for %[[VAL_6:.*]] = 0 to 196 {
// CHECK-NEXT:          %[[VAL_7:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_5]], %[[VAL_6]]] : memref<3136x196xf32>
// CHECK-NEXT:          %[[VAL_8:.*]] = affine.load %[[VAL_3]]{{\[}}%[[VAL_5]]] : memref<3136xf32>
// CHECK-NEXT:          %[[VAL_9:.*]] = arith.maxf %[[VAL_7]], %[[VAL_8]] {reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK-NEXT:          affine.store %[[VAL_9]], %[[VAL_3]]{{\[}}%[[VAL_5]]] : memref<3136xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.copy %[[VAL_3]], %[[VAL_1]] : memref<3136xf32> to memref<3136xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @expand1(%arg0: memref<3136x196xf32>, %arg1: memref<3136x1xf32>) attributes {OperatorType = "Reduce", mindspore_kernel} {
  %cst = arith.constant -3.40282347E+38 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<3136xf32>
  affine.for %arg2 = 0 to 3136 {
    affine.store %cst, %alloc[%arg2] : memref<3136xf32>
  }
  affine.for %arg2 = 0 to 3136 {
    affine.for %arg3 = 0 to 196 {
      %0 = affine.load %arg0[%arg2, %arg3] : memref<3136x196xf32>
      %1 = affine.load %alloc[%arg2] : memref<3136xf32>
      %2 = arith.maxf %0, %1 {reduction_axes = [1 : index], reduction_type = "x"} : f32
      affine.store %2, %alloc[%arg2] : memref<3136xf32>
    }
  }
  %expand_shape = memref.expand_shape %alloc [[0, 1]] : memref<3136xf32> into memref<3136x1xf32>
  memref.copy %expand_shape, %arg1 : memref<3136x1xf32> to memref<3136x1xf32>
  return
}

// -----

// CHECK-LABEL:   func.func @expand2(
// CHECK-SAME:                       %[[VAL_0:.*]]: memref<784x144xf32>, %[[VAL_1:.*]]: memref<784x144xf32>, 
// CHECK-SAME:                       %[[VAL_2:.*]]: memref<784xf32>, %[[VAL_3:.*]]: memref<784x144xf32>) attributes {OperatorType = "Reduce", mindspore_kernel} {
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[VAL_5:.*]] = memref.alloc() {alignment = 64 : i64} : memref<784x144xf32>
// CHECK-NEXT:      affine.for %[[VAL_6:.*]] = 0 to 784 {
// CHECK-NEXT:        affine.for %[[VAL_7:.*]] = 0 to 144 {
// CHECK-NEXT:          %[[VAL_8:.*]] = affine.load %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : memref<784x144xf32>
// CHECK-NEXT:          %[[VAL_9:.*]] = affine.load %[[VAL_1]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : memref<784x144xf32>
// CHECK-NEXT:          %[[VAL_10:.*]] = arith.addf %[[VAL_8]], %[[VAL_9]] : f32
// CHECK-NEXT:          affine.store %[[VAL_10]], %[[VAL_5]]{{\[}}%[[VAL_6]], %[[VAL_7]]] : memref<784x144xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[VAL_11:.*]] = memref.alloc() {alignment = 64 : i64} : memref<784xf32>
// CHECK-NEXT:      affine.for %[[VAL_12:.*]] = 0 to 784 {
// CHECK-NEXT:        affine.store %[[VAL_4]], %[[VAL_11]]{{\[}}%[[VAL_12]]] : memref<784xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      affine.for %[[VAL_13:.*]] = 0 to 784 {
// CHECK-NEXT:        affine.for %[[VAL_14:.*]] = 0 to 144 {
// CHECK-NEXT:          %[[VAL_15:.*]] = affine.load %[[VAL_5]]{{\[}}%[[VAL_13]], %[[VAL_14]]] : memref<784x144xf32>
// CHECK-NEXT:          %[[VAL_16:.*]] = affine.load %[[VAL_11]]{{\[}}%[[VAL_13]]] : memref<784xf32>
// CHECK-NEXT:          %[[VAL_17:.*]] = arith.addf %[[VAL_15]], %[[VAL_16]] {reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK-NEXT:          affine.store %[[VAL_17]], %[[VAL_11]]{{\[}}%[[VAL_13]]] : memref<784xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.copy %[[VAL_11]], %[[VAL_2]] : memref<784xf32> to memref<784xf32>
// CHECK-NEXT:      memref.copy %[[VAL_5]], %[[VAL_3]] : memref<784x144xf32> to memref<784x144xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }
func.func @expand2(%arg0: memref<784x144xf32>, %arg1: memref<784x144xf32>, %arg2: memref<784x1xf32>, %arg3: memref<784x144xf32>) attributes {OperatorType = "Reduce", mindspore_kernel} {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<784x144xf32>
  affine.for %arg4 = 0 to 784 {
    affine.for %arg5 = 0 to 144 {
      %0 = affine.load %arg0[%arg4, %arg5] : memref<784x144xf32>
      %1 = affine.load %arg1[%arg4, %arg5] : memref<784x144xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %alloc[%arg4, %arg5] : memref<784x144xf32>
    }
  }
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<784xf32>
  affine.for %arg4 = 0 to 784 {
    affine.store %cst, %alloc_0[%arg4] : memref<784xf32>
  }
  affine.for %arg4 = 0 to 784 {
    affine.for %arg5 = 0 to 144 {
      %0 = affine.load %alloc[%arg4, %arg5] : memref<784x144xf32>
      %1 = affine.load %alloc_0[%arg4] : memref<784xf32>
      %2 = arith.addf %0, %1 {reduction_axes = [1 : index], reduction_type = "x"} : f32
      affine.store %2, %alloc_0[%arg4] : memref<784xf32>
    }
  }
  %expand_shape = memref.expand_shape %alloc_0 [[0, 1]] : memref<784xf32> into memref<784x1xf32>
  memref.copy %expand_shape, %arg2 : memref<784x1xf32> to memref<784x1xf32>
  memref.copy %alloc, %arg3 : memref<784x144xf32> to memref<784x144xf32>
  return
}

