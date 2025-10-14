// RUN: akg-opt %s -convert-mindspore-to-tosa | FileCheck %s --check-prefix=CHECK1
// RUN: akg-opt %s -convert-mindspore-to-tosa -convert-mindspore-to-linalg | FileCheck %s --check-prefix=CHECK2
module {
func.func @test_acos(%arg0: tensor<5x6x7xf32>) -> tensor<5x6x7xf32> {
	// CHECK1: mindspore.acos
	// CHECK2: tensor.empty
	// CHECK2: linalg.generic
	// CHECK2: mathExt.acos
	// CHECK2: linalg.yield
	%0 = "mindspore.acos"(%arg0): (tensor<5x6x7xf32>) -> tensor<5x6x7xf32>
	return %0 : tensor<5x6x7xf32>
}

func.func @test_asin(%arg0: tensor<5x6x7xf32>) -> tensor<5x6x7xf32> {
	// CHECK1: mindspore.asin
	// CHECK2: tensor.empty
	// CHECK2: linalg.generic
	// CHECK2: mathExt.asin
	// CHECK2: linalg.yield
	%0 = "mindspore.asin"(%arg0): (tensor<5x6x7xf32>) -> tensor<5x6x7xf32>
	return %0 : tensor<5x6x7xf32>
}

func.func @test_isnan(%arg0: tensor<5x6x7xf32>) -> tensor<5x6x7xi1> {
	// CHECK1: mindspore.isnan
	// CHECK2: tensor.empty
	// CHECK2: linalg.generic
	// CHECK2: mathExt.isnan
	// CHECK2: linalg.yield
	%0 = "mindspore.isnan"(%arg0): (tensor<5x6x7xf32>) -> tensor<5x6x7xi1>
	return %0 : tensor<5x6x7xi1>
}

func.func @test_isinf(%arg0: tensor<5x6x7xf32>) -> tensor<5x6x7xi1> {
	// CHECK1: mindspore.isinf
	// CHECK2: tensor.empty
	// CHECK2: linalg.generic
	// CHECK2: mathExt.isinf
	// CHECK2: linalg.yield
	%0 = "mindspore.isinf"(%arg0): (tensor<5x6x7xf32>) -> tensor<5x6x7xi1>
	return %0 : tensor<5x6x7xi1>
}

}
