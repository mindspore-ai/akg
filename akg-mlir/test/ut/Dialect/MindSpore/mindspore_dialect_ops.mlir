// RUN: akg-opt %s -convert-mindspore-to-tosa | FileCheck %s --check-prefix=CHECK1

// RUN: akg-opt %s -convert-mindspore-to-tosa -convert-mindspore-to-linalg | FileCheck %s --check-prefix=CHECK2

// RUN: akg-opt %s -convert-mindspore-to-tosa -convert-mindspore-to-linalg -mindspore-finalizing-lower | FileCheck %s --check-prefix=CHECK3
module {
  func.func @test_elements(%arg0: tensor<16x?x?xf32>, %arg1: tensor<16x?x?xf32>) -> tensor<16x?x1xf32> {
    // CHECK1: tosa.add
    %0 = "mindspore.add"(%arg0, %arg1) : (tensor<16x?x?xf32>, tensor<16x?x?xf32>) -> tensor<16x?x?xf32>
    // CHECK1: tosa.mul
    %1 = "mindspore.square"(%arg0) : (tensor<16x?x?xf32>) -> tensor<16x?x?xf32>
    // CHECK1: tosa.sub
    %2 = "mindspore.sub"(%1, %0) : (tensor<16x?x?xf32>, tensor<16x?x?xf32>) -> tensor<16x?x?xf32>
    // CHECK1: tosa.mul
    %3 = "mindspore.mul"(%2, %1) : (tensor<16x?x?xf32>, tensor<16x?x?xf32>) -> tensor<16x?x?xf32>
    // CHECK1: mindspore.div
    // CHECK2: tensor.dim
    // CHECK2: tensor.empty
    // CHECK2: linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
    // CHECK2: arith.divf
    %4 = "mindspore.div"(%3, %2) : (tensor<16x?x?xf32>, tensor<16x?x?xf32>) -> tensor<16x?x?xf32>
    // CHECK1: tosa.pow
    %5 = "mindspore.pow"(%4, %3) : (tensor<16x?x?xf32>, tensor<16x?x?xf32>) -> tensor<16x?x?xf32>
    // CHECK1: tosa.exp
    %6 = "mindspore.exp"(%5) : (tensor<16x?x?xf32>) -> tensor<16x?x?xf32>
    // CHECK1: mindspore.assign
    // CHECK2: linalg.copy
    %7 = "mindspore.assign"(%6, %5) : (tensor<16x?x?xf32>, tensor<16x?x?xf32>) -> tensor<16x?x?xf32>
    // CHECK1: mindspore.reduce_max
    // CHECK2: linalg.generic
    %8 = "mindspore.reduce_max"(%7) {axis = array<i64: 2>, keepdims = true} : (tensor<16x?x?xf32>) -> tensor<16x?x1xf32>
    // CHECK1: mindspore.sin
    // CHECK2: tensor.dim
    // CHECK2: tensor.empty
    // CHECK2: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
    // CHECK2: math.sin %in
    %9 = "mindspore.sin"(%8) : (tensor<16x?x1xf32>) -> tensor<16x?x1xf32>
    // CHECK1: mindspore.cos
    // CHECK2: tensor.dim
    // CHECK2: tensor.empty
    // CHECK2: linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
    // CHECK2: math.cos %in
    %10 = "mindspore.cos"(%9) : (tensor<16x?x1xf32>) -> tensor<16x?x1xf32>
    return %10 : tensor<16x?x1xf32>
  }

  func.func @all_reduce(%arg0: tensor<1x4954x3xf32>) -> tensor<1xf32> attributes {enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
    // CHECK1: mindspore.reduce_sum
    // CHECK2: linalg.generic
    %0 = "mindspore.reduce_sum"(%arg0) {axis = array<i64: 0, 1, 2>, keepdims = false, ms_attr = {axis = array<i64: 0, 1, 2>, keep_dims = false}, ori_op = "ReduceSum"} : (tensor<1x4954x3xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }

  func.func @tile(%arg0: tensor<16x1xf32>) -> tensor<16x2xf32> {
    // CHECK1: mindspore.tile
    // CHECK2: linalg.generic
    %0 = "mindspore.tile"(%arg0) {multiples = array<i64: 1, 2>} : (tensor<16x1xf32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
  }

  func.func @const_cast() -> tensor<1xf32> {
    // CHECK1: tosa.const
    %0 = "mindspore.const"() {value = dense<1.250000e-01> : tensor<1xf16>} : () -> tensor<1xf16>
    // CHECK1: tosa.cast
    %1 = "mindspore.cast"(%0) : (tensor<1xf16>) -> tensor<1xf32>
    return %1 : tensor<1xf32>
  }

  func.func @concat(%arg0: tensor<2x?xf32>) -> tensor<8x?xf32> {
    // CHECK1: "tosa.concat"(%arg0, %arg0) {axis = 0 : i64}
    // CHECK1: "tosa.concat"(%0, %0) {axis = 0 : i64}
    %0 = "mindspore.concat"(%arg0, %arg0) {axis = 0 : i64} : (tensor<2x?xf32>, tensor<2x?xf32>)  -> (tensor<4x?xf32>)
    %1 = "mindspore.concat"(%0, %0) : (tensor<4x?xf32>, tensor<4x?xf32>)  -> (tensor<8x?xf32>)
    return %1 : tensor<8x?xf32>
  }

  func.func @strided_slice1(%arg0: tensor<5x6x7xf32>) -> tensor<2x2x2xf32> {
    // CHECK3:  tensor.extract_slice %arg0[1, 3, 2] [2, 2, 2] [1, 1, 2] : tensor<5x6x7xf32> to tensor<2x2x2xf32>
    %0 = "mindspore.strided_slice"(%arg0) {start = array<i64: 1, 3, 2>, end = array<i64: 3, 5, 5>, strides = array<i64: 1, 1, 2>} : (tensor<5x6x7xf32>) -> tensor<2x2x2xf32>
    return %0 : tensor<2x2x2xf32>
  }

  func.func @strided_slice2(%arg0: tensor<5x6x7xf32>) -> tensor<2x2x2xf32> {
    // CHECK3:  tensor.extract_slice %arg0[1, 3, 2] [2, 2, 2] [1, 1, 2] : tensor<5x6x7xf32> to tensor<2x2x2xf32>
    %0 = "mindspore.strided_slice"(%arg0) {start = array<i64: 1, 3, 2>, end = array<i64: 3, 5, 6>, strides = array<i64: 1, 1, 2>} : (tensor<5x6x7xf32>) -> tensor<2x2x2xf32>
    return %0 : tensor<2x2x2xf32>
  }

  func.func @addn(%arg0: tensor<5x?x7xf32>, %arg1: tensor<5x?x7xf32>, %arg2: tensor<5x?x7xf32>) -> tensor<5x?x7xf32> {
    // CHECK1: mindspore.addn
    // CHECK2: linalg.generic
    // CHECK2: arith.addf
    // CHECK3: linalg.generic
    // CHECK3: arith.addf
    %0 = "mindspore.addn"(%arg0, %arg1, %arg2) : (tensor<5x?x7xf32>, tensor<5x?x7xf32>, tensor<5x?x7xf32>) -> tensor<5x?x7xf32>
    return %0 : tensor<5x?x7xf32>
  }
}
