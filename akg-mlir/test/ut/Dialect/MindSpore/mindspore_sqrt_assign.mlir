// RUN: akg-opt %s -convert-mindspore-to-tosa -convert-mindspore-to-linalg | FileCheck %s
module {
func.func @sqrt(%arg0: tensor<4x4x?xf32>) -> tensor<4x4x?xf32> {
	// CHECK: math.sqrt
	%0 = "mindspore.sqrt"(%arg0) : (tensor<4x4x?xf32>) -> tensor<4x4x?xf32>
	return %0 : tensor<4x4x?xf32>
}

func.func @assign(%arg0: tensor<16x?x?xf32>, %arg1: tensor<16x?x?xf32>) -> tensor<16x?x?xf32> {
	// CHECK: linalg.copy
	%0 = "mindspore.assign"(%arg0, %arg1) : (tensor<16x?x?xf32>, tensor<16x?x?xf32>) -> tensor<16x?x?xf32>
	return %0 : tensor<16x?x?xf32>
}

}

