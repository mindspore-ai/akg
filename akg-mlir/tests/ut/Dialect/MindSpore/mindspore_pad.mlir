// RUN: akg-opt %s -convert-mindspore-to-tosa | FileCheck %s
module {
func.func @test_padOp(%arg0 : tensor<1x2xf32>) -> tensor<1x5xf32>  {
	// CHECK: tosa.const
	// CHECK: tosa.const
	// CHECK: tosa.pad
	%0 = "mindspore.pad"(%arg0) {padding = array<i64: 1, 2>}: (tensor<1x2xf32>) -> (tensor<1x5xf32>)
	return %0 : tensor<1x5xf32>
}

}


