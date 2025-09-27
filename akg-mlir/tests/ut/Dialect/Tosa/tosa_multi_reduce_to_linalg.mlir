// RUN: akg-opt %s --tosa-multi-reduce-to-linalg | FileCheck %s

// CHECK-LABEL:  #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:   #map1 = affine_map<(d0, d1) -> ()>
// CHECK-NEXT:   #map2 = affine_map<(d0) -> (d0)>
// CHECK-NEXT:   #map3 = affine_map<(d0) -> ()>
// CHECK-NEXT:   #map4 = affine_map<(d0, d1) -> (d1)>
// CHECK-NEXT:   #map5 = affine_map<(d0, d1) -> (d0)>
// CHECK-NEXT:   #map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT:   #map7 = affine_map<(d0, d1, d2, d3) -> (d2)>
// CHECK-NEXT:   #map8 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-NEXT:   module {
// CHECK-NEXT:     func.func @two_reduce_groups(%arg0: tensor<8x2xf32>, %arg1: tensor<7xf32>) -> (tensor<1x1xf32>, tensor<1xf32>) attributes {mindspore_kernel, process = "cuda"} {
// CHECK-NEXT:       %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %0 = tensor.empty() : tensor<f32>
// CHECK-NEXT:       %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<f32>) -> tensor<f32>
// CHECK-NEXT:       %2 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction"]} ins(%arg0 : tensor<8x2xf32>) outs(%1 : tensor<f32>) {
// CHECK-NEXT:       ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:         %6 = arith.addf %in, %out : f32
// CHECK-NEXT:         linalg.yield %6 : f32
// CHECK-NEXT:       } -> tensor<f32>
// CHECK-NEXT:       %expanded = tensor.expand_shape %2 [] : tensor<f32> into tensor<1x1xf32>
// CHECK-NEXT:       %3 = tensor.empty() : tensor<f32>
// CHECK-NEXT:       %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
// CHECK-NEXT:       %5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["reduction"]} ins(%arg1 : tensor<7xf32>) outs(%4 : tensor<f32>) {
// CHECK-NEXT:       ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:         %6 = arith.addf %in, %out : f32
// CHECK-NEXT:         linalg.yield %6 : f32
// CHECK-NEXT:       } -> tensor<f32>
// CHECK-NEXT:       %expanded_0 = tensor.expand_shape %5 [] : tensor<f32> into tensor<1xf32>
// CHECK-NEXT:       return %expanded, %expanded_0 : tensor<1x1xf32>, tensor<1xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func @can_not_merge(%arg0: tensor<8x2xf32>) -> tensor<1x1xf32> attributes {mindspore_kernel, process = "cuda"} {
// CHECK-NEXT:       %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %0 = tensor.empty() : tensor<2xf32>
// CHECK-NEXT:       %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2xf32>) -> tensor<2xf32>
// CHECK-NEXT:       %2 = linalg.generic {indexing_maps = [#map, #map4], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<8x2xf32>) outs(%1 : tensor<2xf32>) {
// CHECK-NEXT:       ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:         %7 = arith.addf %in, %out : f32
// CHECK-NEXT:         linalg.yield %7 : f32
// CHECK-NEXT:       } -> tensor<2xf32>
// CHECK-NEXT:       %expanded = tensor.expand_shape %2 {{\[}}[0, 1]{{\]}} : tensor<2xf32> into tensor<1x2xf32>
// CHECK-NEXT:       %3 = "tosa.add"(%expanded, %expanded) : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
// CHECK-NEXT:       %4 = tensor.empty() : tensor<1xf32>
// CHECK-NEXT:       %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<1xf32>) -> tensor<1xf32>
// CHECK-NEXT:       %6 = linalg.generic {indexing_maps = [#map, #map5], iterator_types = ["parallel", "reduction"]} ins(%3 : tensor<1x2xf32>) outs(%5 : tensor<1xf32>) {
// CHECK-NEXT:       ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:         %7 = arith.addf %in, %out : f32
// CHECK-NEXT:         linalg.yield %7 : f32
// CHECK-NEXT:       } -> tensor<1xf32>
// CHECK-NEXT:       %expanded_0 = tensor.expand_shape %6 {{\[}}[0, 1]{{\]}} : tensor<1xf32> into tensor<1x1xf32>
// CHECK-NEXT:       return %expanded_0 : tensor<1x1xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func @reduce_4d(%arg0: tensor<2x3x4x5xf32>) -> tensor<1x1x4x1xf32> attributes {mindspore_kernel, process = "cuda"} {
// CHECK-NEXT:       %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %0 = tensor.empty() : tensor<4xf32>
// CHECK-NEXT:       %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT:       %2 = linalg.generic {indexing_maps = [#map6, #map7], iterator_types = ["reduction", "reduction", "parallel", "reduction"]} ins(%arg0 : tensor<2x3x4x5xf32>) outs(%1 : tensor<4xf32>) {
// CHECK-NEXT:       ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:         %3 = arith.addf %in, %out : f32
// CHECK-NEXT:         linalg.yield %3 : f32
// CHECK-NEXT:       } -> tensor<4xf32>
// CHECK-NEXT:       %expanded = tensor.expand_shape %2 {{\[}}[0, 1, 2, 3]{{\]}} : tensor<4xf32> into tensor<1x1x4x1xf32>
// CHECK-NEXT:       return %expanded : tensor<1x1x4x1xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func @reduce_4d_inter(%arg0: tensor<2x3x4x5xf32>) -> tensor<1x3x1x5xf32> attributes {mindspore_kernel, process = "cuda"} {
// CHECK-NEXT:       %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %0 = tensor.empty() : tensor<3x5xf32>
// CHECK-NEXT:       %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<3x5xf32>) -> tensor<3x5xf32>
// CHECK-NEXT:       %2 = linalg.generic {indexing_maps = [#map6, #map8], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%arg0 : tensor<2x3x4x5xf32>) outs(%1 : tensor<3x5xf32>) {
// CHECK-NEXT:       ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:         %3 = arith.addf %in, %out : f32
// CHECK-NEXT:         linalg.yield %3 : f32
// CHECK-NEXT:       } -> tensor<3x5xf32>
// CHECK-NEXT:       %expanded = tensor.expand_shape %2 {{\[}}[0, 1], [2, 3]{{\]}} : tensor<3x5xf32> into tensor<1x3x1x5xf32>
// CHECK-NEXT:       return %expanded : tensor<1x3x1x5xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }

module {
  func.func @two_reduce_groups(%arg1: tensor<8x2xf32>, %arg2: tensor<7xf32>) -> (tensor<1x1xf32>, tensor<1xf32>) attributes {mindspore_kernel, process = "cuda"} {
    %7 = "tosa.reduce_sum"(%arg1) {axis = 0 : i64, ori_op = "ReduceSum"} : (tensor<8x2xf32>) -> tensor<1x2xf32>
    %8 = "tosa.reduce_sum"(%7) {axis = 1 : i64, ori_op = "ReduceSum"} : (tensor<1x2xf32>) -> tensor<1x1xf32>
    %9 = "tosa.reduce_sum"(%arg2) {axis = 0 : i64, ori_op = "ReduceSum"} : (tensor<7xf32>) -> tensor<1xf32>
    return %8, %9 : tensor<1x1xf32>, tensor<1xf32>
  }
  func.func @can_not_merge(%arg1: tensor<8x2xf32>) -> (tensor<1x1xf32>) attributes {mindspore_kernel, process = "cuda"} {
    %10 = "tosa.reduce_sum"(%arg1) {axis = 0 : i64, ori_op = "ReduceSum"} : (tensor<8x2xf32>) -> tensor<1x2xf32>
    %11 = "tosa.add"(%10, %10) : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x2xf32>
    %12 = "tosa.reduce_sum"(%11) {axis = 1 : i64, ori_op = "ReduceSum"} : (tensor<1x2xf32>) -> tensor<1x1xf32>
    return %12: tensor<1x1xf32>
  }
  func.func @reduce_4d(%arg1: tensor<2x3x4x5xf32>) -> (tensor<1x1x4x1xf32>) attributes {mindspore_kernel, process = "cuda"} {
    %7 = "tosa.reduce_sum"(%arg1) {axis = 0 : i64, ori_op = "ReduceSum"} : (tensor<2x3x4x5xf32>) -> tensor<1x3x4x5xf32>
    %8 = "tosa.reduce_sum"(%7) {axis = 1 : i64, ori_op = "ReduceSum"} : (tensor<1x3x4x5xf32>) -> tensor<1x1x4x5xf32>
    %9 = "tosa.reduce_sum"(%8) {axis = 3 : i64, ori_op = "ReduceSum"} : (tensor<1x1x4x5xf32>) -> tensor<1x1x4x1xf32>
    return %9 : tensor<1x1x4x1xf32>
  }
  func.func @reduce_4d_inter(%arg1: tensor<2x3x4x5xf32>) -> (tensor<1x3x1x5xf32>) attributes {mindspore_kernel, process = "cuda"} {
    %7 = "tosa.reduce_sum"(%arg1) {axis = 0 : i64, ori_op = "ReduceSum"} : (tensor<2x3x4x5xf32>) -> tensor<1x3x4x5xf32>
    %8 = "tosa.reduce_sum"(%7) {axis = 2 : i64, ori_op = "ReduceSum"} : (tensor<1x3x4x5xf32>) -> tensor<1x3x1x5xf32>
    return %8 : tensor<1x3x1x5xf32>
  }    
 
}
