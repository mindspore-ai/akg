// RUN: akg-opt %s  -math-ext-lower | FileCheck %s --check-prefix=CHECK
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @test_isnan(%arg0: tensor<5x6x7xf32>) -> tensor<5x6x7xi1> {
    %0 = tensor.empty() : tensor<5x6x7xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<5x6x7xf32>) outs(%0 : tensor<5x6x7xi1>) {
    ^bb0(%in: f32, %out: i1):
	    // CHECK: arith.constant
	    // CHECK: arith.bitcast
	    // CHECK: arith.andi
      // CHECK: arith.subi
      // CHECK: arith.shrui
      // CHECK: arith.trunci
      %2 = mathExt.isnan %in : f32
      linalg.yield %2 : i1
    } -> tensor<5x6x7xi1>
    return %1 : tensor<5x6x7xi1>
  }
  func.func @test_isinf(%arg0: tensor<5x6x7xf32>) -> tensor<5x6x7xi1> {
    %0 = tensor.empty() : tensor<5x6x7xi1>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<5x6x7xf32>) outs(%0 : tensor<5x6x7xi1>) {
    ^bb0(%in: f32, %out: i1):
	    // CHECK: arith.constant
	    // CHECK: arith.constant
	    // CHECK: arith.bitcast
	    // CHECK: arith.bitcast
	    // CHECK: arith.cmpf ueq
	    // CHECK: arith.cmpf ueq
	    // CHECK: arith.ori
      %2 = mathExt.isinf %in : f32
      linalg.yield %2 : i1
    } -> tensor<5x6x7xi1>
    return %1 : tensor<5x6x7xi1>
  }
}
