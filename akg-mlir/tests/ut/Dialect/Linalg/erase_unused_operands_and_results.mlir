// RUN: akg-opt %s -split-input-file -erase-unused-operands-and-results | FileCheck %s

// Drop an output whose SSA result has no users (pass-through on the removed init is eligible).

#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @erase_unused_drop_second_output
// CHECK: %[[G:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<4xf32>) outs(%{{.*}} : tensor<4xf32>)
// CHECK: linalg.yield %{{.*}} : f32
// CHECK: return %[[G]] : tensor<4xf32>
func.func @erase_unused_drop_second_output(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %cst = arith.constant 0.0 : f32
  %e0 = tensor.empty() : tensor<4xf32>
  %e1 = tensor.empty() : tensor<4xf32>
  %0:2 = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]}
    ins(%arg0 : tensor<4xf32>)
    outs(%e0, %e1 : tensor<4xf32>, tensor<4xf32>) {
  ^bb0(%in: f32, %o0: f32, %o1: f32):
    %1 = arith.addf %in, %cst : f32
    linalg.yield %1, %o1 : f32, f32
  } -> (tensor<4xf32>, tensor<4xf32>)
  return %0#0 : tensor<4xf32>
}

// -----

// Do not drop an output that is unused at return but still participates in the reduction payload.

#map_r = affine_map<(d0) -> (d0)>
#map_acc = affine_map<(d0) -> ()>
// CHECK-LABEL: func @erase_unused_keep_reduction_aux
// CHECK: %[[G:.*]]:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction"]}
// CHECK-SAME: ins(%arg0 : tensor<4xf32>) outs(%{{.*}}, %{{.*}} : tensor<f32>, tensor<i32>)
// CHECK: return %[[G]]#1
func.func @erase_unused_keep_reduction_aux(%arg0: tensor<4xf32>) -> tensor<i32> {
  %init0 = tensor.empty() : tensor<f32>
  %init1 = tensor.empty() : tensor<i32>
  %0:2 = linalg.generic {
    indexing_maps = [#map_r, #map_acc, #map_acc],
    iterator_types = ["reduction"]}
    ins(%arg0 : tensor<4xf32>)
    outs(%init0, %init1 : tensor<f32>, tensor<i32>) {
  ^bb0(%b0: f32, %b1: f32, %b2: i32):
    %cmp = arith.cmpf oge, %b0, %b1 : f32
    %maxv = arith.select %cmp, %b0, %b1 : f32
    %idx = arith.constant 0 : i32
    %sel = arith.select %cmp, %idx, %b2 : i32
    linalg.yield %maxv, %sel : f32, i32
  } -> (tensor<f32>, tensor<i32>)
  return %0#1 : tensor<i32>
}

// -----

// Both results used: keep two outputs.

#map2 = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @erase_unused_keep_both_outputs
// CHECK: %[[G:.*]]:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
// CHECK-SAME: outs(%{{.*}}, %{{.*}} : tensor<4xf32>, tensor<4xf32>)
// CHECK: return %[[G]]#0, %[[G]]#1
func.func @erase_unused_keep_both_outputs(%arg0: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %cst = arith.constant 0.0 : f32
  %e0 = tensor.empty() : tensor<4xf32>
  %e1 = tensor.empty() : tensor<4xf32>
  %0:2 = linalg.generic {
    indexing_maps = [#map2, #map2, #map2],
    iterator_types = ["parallel"]}
    ins(%arg0 : tensor<4xf32>)
    outs(%e0, %e1 : tensor<4xf32>, tensor<4xf32>) {
  ^bb0(%in: f32, %o0: f32, %o1: f32):
    %1 = arith.addf %in, %cst : f32
    linalg.yield %1, %1 : f32, f32
  } -> (tensor<4xf32>, tensor<4xf32>)
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}
