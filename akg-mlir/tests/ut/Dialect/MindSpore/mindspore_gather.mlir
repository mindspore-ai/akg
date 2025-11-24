//RUN: akg-opt %s -convert-mindspore-to-linalg | FileCheck %s
//CHECK: map = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2)>
//CHECK: map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
//CHECK: tensor.empty() : tensor<4x8x9x6x7xf32>
//CHECK: linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<8x9xi32>) outs(%0 : tensor<4x8x9x6x7xf32>) {
//CHECK: linalg.index 0 : index
//CHECK: arith.index_cast %in : i32 to index
//CHECK: linalg.index 3 : index
//CHECK: linalg.index 4 : index
//CHECK: tensor.extract %arg0[%2, %3, %4, %5] : tensor<4x5x6x7xf32>
//CHECK: linalg.yield %extracted : f32

module {
  func.func @gather_float(%arg0: tensor<4x5x6x7xf32>, %arg1: tensor<8x9xi32>) -> tensor<4x8x9x6x7xf32> {
    %0 = "mindspore.gather"(%arg0, %arg1) {axis = 1 : i64}  : (tensor<4x5x6x7xf32>, tensor<8x9xi32>)  -> (tensor<4x8x9x6x7xf32>)
    return %0 : tensor<4x8x9x6x7xf32>
  }
}


