// RUN: akg-opt %s -split-input-file --fold-dimension | FileCheck %s

// CHECK-LABEL: func.func @Fused_Activation_12627997470141352698(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<401408xf32>) -> tensor<401408xf32>
// CHECK:       "tosa.const"
// CHECK-SAME:    tensor<1xf32>
// CHECK:       "tosa.negate"
// CHECK-SAME:    tensor<401408xf32>
// CHECK:       "tosa.add"
// CHECK-SAME:    (tensor<401408xf32>, tensor<1xf32>)
// CHECK:       "tosa.mul"
// CHECK:       "tosa.mul"
// CHECK-SAME:    (tensor<401408xf32>, tensor<401408xf32>)
// CHECK:       return
// CHECK-SAME:    tensor<401408xf32>

module {
  func.func @Fused_Activation_12627997470141352698(%arg0: tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32> attributes {OperatorType = "Broadcast", enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
    %0 = "tosa.const"() {value = dense<1.000000e+00> : tensor<1x1x1x1xf32>} : () -> tensor<1x1x1x1xf32>
    %1 = "tosa.negate"(%arg0) : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %2 = "tosa.exp"(%1) : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %3 = "tosa.add"(%2, %0) : (tensor<1x128x56x56xf32>, tensor<1x1x1x1xf32>) -> tensor<1x128x56x56xf32>
    %4 = "tosa.reciprocal"(%3) {ori_op = "RealDiv"} : (tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    %5 = "tosa.mul"(%4, %0) {shift = 0 : i32} : (tensor<1x128x56x56xf32>, tensor<1x1x1x1xf32>) -> tensor<1x128x56x56xf32>
    %6 = "tosa.mul"(%arg0, %5) {shift = 0 : i32} : (tensor<1x128x56x56xf32>, tensor<1x128x56x56xf32>) -> tensor<1x128x56x56xf32>
    return %6 : tensor<1x128x56x56xf32>
  }
}

// -----

// CHECK-LABEL:   func.func @Fused_Mul_Mul_Mul_split_13610515452872716857(
// CHECK-NOT:       tensor<1x17349x3xf32>
// CHECK:         "tosa.mul"
// CHECK-SAME:      (tensor<17349x3xf32>, tensor<17349x3xf32>) -> tensor<17349x3xf32>
// CHECK:         "tosa.mul"
// CHECK-SAME:      (tensor<17349x3xf32>, tensor<17349x3xf32>) -> tensor<17349x3xf32>
// CHECK:         "tosa.mul"
// CHECK-SAME:      (tensor<17349x1xf32>, tensor<17349x3xf32>) -> tensor<17349x3xf32>
// CHECK:         return
// CHECK-SAME:      tensor<17349x3xf32>, tensor<17349x3xf32>

module {

  func.func @Fused_Mul_Mul_Mul_split_13610515452872716857(%arg0: tensor<1x17349x3xf32>, %arg1: tensor<1x17349x1xf32>, %arg2: tensor<1x17349x3xf32>, %arg3: tensor<1x17349x3xf32>) -> (tensor<1x17349x3xf32>, tensor<1x17349x3xf32>) attributes {OperatorType = "Broadcast", enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
    %0 = "tosa.mul"(%arg2, %arg3) {shift = 0 : i32} : (tensor<1x17349x3xf32>, tensor<1x17349x3xf32>) -> tensor<1x17349x3xf32>
    %1 = "tosa.mul"(%arg0, %0) {shift = 0 : i32} : (tensor<1x17349x3xf32>, tensor<1x17349x3xf32>) -> tensor<1x17349x3xf32>
    %2 = "tosa.mul"(%arg1, %0) {shift = 0 : i32} : (tensor<1x17349x1xf32>, tensor<1x17349x3xf32>) -> tensor<1x17349x3xf32>
    return %1, %2 : tensor<1x17349x3xf32>, tensor<1x17349x3xf32>
  }
}

// -----