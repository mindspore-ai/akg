// RUN: akg-opt %s -split-input-file -linalg-copy-bufferize | FileCheck %s

// CHECK-LABEL: func.func @one_linalg_copy
// CHECK-SAME: -> tensor<21128x768xf32>
// CHECK: %expanded_0 = tensor.expand_shape
// CHECK-NEXT: bufferization.to_memref
// CHECK-NEXT: bufferization.to_memref
// CHECK-NEXT: memref.copy
// CHECK: return
// CHECK-SAME : tensor<21128x768xf32>

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @one_linalg_copy(%arg0: tensor<1xf32>, %arg1: tensor<21128x768xf32>, %arg2: tensor<21128x768xf16>, %arg3: tensor<1x1xf32>) -> (tensor<1x1xf32>, tensor<21128x768xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cuda"} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<21128x768xf32>
    %collapsed = tensor.collapse_shape %arg0 [] : tensor<1xf32> into tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg2, %collapsed : tensor<21128x768xf32>, tensor<21128x768xf16>, tensor<f32>) outs(%0 : tensor<21128x768xf32>) {
    ^bb0(%in: f32, %in_1: f16, %in_2: f32, %out: f32):
      %9 = arith.extf %in_1 : f16 to f32
      %10 = arith.addf %in, %9 : f32
      %11 = arith.mulf %10, %in_2 : f32
      linalg.yield %11 : f32
    } -> tensor<21128x768xf32>
    %2 = tensor.empty() : tensor<768xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<768xf32>) -> tensor<768xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["reduction", "parallel"]} ins(%1 : tensor<21128x768xf32>) outs(%3 : tensor<768xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.mulf %in, %in : f32
      %10 = arith.addf %9, %out : f32
      linalg.yield %10 : f32
    } -> tensor<768xf32>
    %expanded = tensor.expand_shape %4 [[0, 1]] : tensor<768xf32> into tensor<1x768xf32>
    %5 = tensor.empty() : tensor<1xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1xf32>) -> tensor<1xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["parallel", "reduction"]} ins(%expanded : tensor<1x768xf32>) outs(%6 : tensor<1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %9 = arith.addf %in, %out : f32
      linalg.yield %9 : f32
    } -> tensor<1xf32>
    %expanded_0 = tensor.expand_shape %7 [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
    %8 = linalg.copy ins(%expanded_0 : tensor<1x1xf32>) outs(%arg3 : tensor<1x1xf32>) -> tensor<1x1xf32>
    return %8, %1 : tensor<1x1xf32>, tensor<21128x768xf32>
  }
}

// -----

// CHECK-LABEL: func.func @no_linalg_copy
// CHECK-SAME: -> tensor<640x1x1xf32>
// CHECK-NEXT: return %[[ARG0:.*]] : tensor<640x1x1xf32>

func.func @no_linalg_copy(%arg0: tensor<640x1x1xf32>) -> tensor<640x1x1xf32> attributes {OperatorType = "Default", enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
  return %arg0 : tensor<640x1x1xf32>
}

// -----
