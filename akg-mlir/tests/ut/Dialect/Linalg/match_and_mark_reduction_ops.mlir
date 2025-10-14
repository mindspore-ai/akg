// RUN: akg-opt %s -split-input-file --match-and-mark-reduction-ops | FileCheck %s

//CHECK-LABEL: #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> ()>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @fused_reduction(%arg0: tensor<16x1x1x128xf16>, %arg1: tensor<16x12x128x128xf16>) -> tensor<16x12x128x1xf16> attributes {OperatorType = "Reduce", mindspore_kernel} {
    %0 = "tosa.const"() {value = dense<1.250000e-01> : tensor<1x1x1x1xf16>} : () -> tensor<1x1x1x1xf16>
    %1 = tensor.empty() : tensor<16x12x128x128xf16>
    %2 = "tosa.const"() {value = dense<1.250000e-01> : tensor<f16>} : () -> tensor<f16>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1, %2 : tensor<16x12x128x128xf16>, tensor<f16>) outs(%1 : tensor<16x12x128x128xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %9 = arith.mulf %in, %in_0 : f16
      linalg.yield %9 : f16
    } -> tensor<16x12x128x128xf16>
    %4 = tensor.empty() : tensor<16x12x128x128xf16>
    %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2], [3]] : tensor<16x1x1x128xf16> into tensor<16x128xf16>
    %5 = linalg.generic {indexing_maps = [#map2, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%collapsed, %3 : tensor<16x128xf16>, tensor<16x12x128x128xf16>) outs(%4 : tensor<16x12x128x128xf16>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %9 = arith.addf %in, %in_0 : f16
      linalg.yield %9 : f16
    } -> tensor<16x12x128x128xf16>
    %6 = tensor.empty() : tensor<16x12x128xf16>
    %cst = arith.constant -6.550400e+04 : f16
    %7 = linalg.fill ins(%cst : f16) outs(%6 : tensor<16x12x128xf16>) -> tensor<16x12x128xf16>
    %8 = linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%5 : tensor<16x12x128x128xf16>) outs(%7 : tensor<16x12x128xf16>) {
    ^bb0(%in: f16, %out: f16):
      %9 = arith.maxf %in, %out : f16
      linalg.yield %9 : f16
    } -> tensor<16x12x128xf16>
    %expanded = tensor.expand_shape %8 [[0], [1], [2, 3]] : tensor<16x12x128xf16> into tensor<16x12x128x1xf16>
    return %expanded : tensor<16x12x128x1xf16>
  }
  func.func @reduce_x(%arg0: tensor<4096x7680xf32>) -> tensor<4096x1xf32> attributes {OperatorType = "Reduce", feature = "neon", mindspore_kernel, process = "cpu"} {
    %0 = tensor.empty() : tensor<4096xf32>
    %cst = arith.constant -3.40282347E+38 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096xf32>) -> tensor<4096xf32>
    %2 = linalg.generic {indexing_maps = [#map4, #map5], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<4096x7680xf32>) outs(%1 : tensor<4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.maxf %in, %out : f32
      linalg.yield %3 : f32
    } -> tensor<4096xf32>
    %expanded = tensor.expand_shape %2 [[0, 1]] : tensor<4096xf32> into tensor<4096x1xf32>
    return %expanded : tensor<4096x1xf32>
  }
}
