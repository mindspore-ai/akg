// RUN: akg-opt %s -akg-tosa-operator-identify | FileCheck %s
 
// CHECK-LABEL: func.func @operator_identify(%arg0: tensor<2xf32>) -> tensor<1xf32> attributes {OperatorType = "Reduce", mindspore_kernel} {
// CHECK-NEXT:   %0 = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:   %1 = "tosa.reduce_sum"(%0) {axis = 0 : i64} : (tensor<2xf32>) -> tensor<1xf32>
// CHECK-NEXT:   return %1 : tensor<1xf32>
// CHECK-NEXT: }

 
func.func @operator_identify(%arg0: tensor<2xf32>) -> tensor<1xf32> attributes {mindspore_kernel} { 
	%0 = "tosa.mul"(%arg0, %arg0) {shift = 0 : i32} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32> 
	%1 = "tosa.reduce_sum"(%0) {axis = 0 : i64} : (tensor<2xf32>) -> tensor<1xf32> 
	return %1 : tensor<1xf32> 
}