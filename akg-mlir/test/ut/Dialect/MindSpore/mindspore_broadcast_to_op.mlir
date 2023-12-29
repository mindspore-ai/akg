// RUN: akg-opt %s -convert-mindspore-to-tosa -convert-mindspore-to-linalg | FileCheck %s

module {
  // outDimSize = 1
  func.func @Fused_dyn_tile_00_dynamic(%arg0: tensor<1x?xf32>, %arg1: tensor<2xindex>) -> tensor<1x?xf32> {
    %0 = "mindspore.broadcast_to"(%arg0, %arg1) : (tensor<1x?xf32>, tensor<2xindex>)  -> (tensor<1x?xf32>)
    return %0 : tensor<1x?xf32>
  }
// CHECK: tensor.empty(%1, %dim) : tensor<1x?x?xf32>
// CHECK: linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?xf32>) outs(%2 : tensor<1x?x?xf32>) {
// CHECK: tensor.collapse_shape

  // outDimSize > 1
  // inputDimSize == ShapedType::kDynamic
  func.func @Fused_dyn_tile_10_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<2xindex>) -> tensor<32x?xf32> {
    %0 = "mindspore.broadcast_to"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<2xindex>)  -> (tensor<32x?xf32>)
    return %0 : tensor<32x?xf32>
  }
// CHECK: tensor.empty(%0, %dim, %2, %dim_0) : tensor<?x?x?x?xf32>
// CHECK: linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%3 : tensor<?x?x?x?xf32>) {
// CHECK: tensor.collapse_shape

  // outDimSize > 1
  // inputDimSize == 1
  func.func @Fused_dyn_tile_11_dynamic(%arg0: tensor<1x?xf32>, %arg1: tensor<2xindex>) -> tensor<32x?xf32> {
    %0 = "mindspore.broadcast_to"(%arg0, %arg1) : (tensor<1x?xf32>, tensor<2xindex>)  -> (tensor<32x?xf32>)
    return %0 : tensor<32x?xf32>
  }
// CHECK: tensor.empty(%1, %dim) : tensor<32x1x?x?xf32>
// CHECK: linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?xf32>) outs(%2 : tensor<32x1x?x?xf32>) {
// CHECK: tensor.collapse_shape

  // outDimSize > 1
  // inputDimSize == outDimSize
  func.func @Fused_dyn_tile_12_dynamic(%arg0: tensor<32x?xf32>, %arg1: tensor<2xindex>) -> tensor<32x?xf32> {
    %0 = "mindspore.broadcast_to"(%arg0, %arg1) : (tensor<32x?xf32>, tensor<2xindex>)  -> (tensor<32x?xf32>)
    return %0 : tensor<32x?xf32>
  }
// CHECK: tensor.empty(%1, %dim) : tensor<32x?x?xf32>
// CHECK: linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<32x?xf32>) outs(%2 : tensor<32x?x?xf32>) {
// CHECK: tensor.collapse_shape 

  // outDimSize > 1
  // inputDimSize < outDimSize
  func.func @Fused_dyn_tile_13_dynamic(%arg0: tensor<2x?xf32>, %arg1: tensor<2xindex>) -> tensor<32x?xf32> {
    %0 = "mindspore.broadcast_to"(%arg0, %arg1) : (tensor<2x?xf32>, tensor<2xindex>)  -> (tensor<32x?xf32>)
    return %0 : tensor<32x?xf32>
  }
// CHECK: tensor.empty(%1, %dim) : tensor<16x2x?x?xf32>
// CHECK: linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x?xf32>) outs(%2 : tensor<16x2x?x?xf32>) {
// CHECK: tensor.collapse_shape
    
  // output type is dynamic
  // inputDimSize == 1
  func.func @Fused_dyn_tile_20_dynamic(%arg0: tensor<1x?xf32>, %arg1: tensor<2xindex>) -> tensor<?x?xf32> {
    %0 = "mindspore.broadcast_to"(%arg0, %arg1) : (tensor<1x?xf32>, tensor<2xindex>)  -> (tensor<?x?xf32>)
    return %0 : tensor<?x?xf32>
  }
// CHECK: tensor.empty(%0, %2, %dim) : tensor<?x1x?x?xf32>
// CHECK: linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x?xf32>) outs(%3 : tensor<?x1x?x?xf32>) {
// CHECK: tensor.collapse_shape

  // output type is dynamic
  // inputDimSize > 1
  func.func @Fused_dyn_tile_21_dynamic(%arg0: tensor<2x?xf32>, %arg1: tensor<2xindex>) -> tensor<?x?xf32> {
    %0 = "mindspore.broadcast_to"(%arg0, %arg1) : (tensor<2x?xf32>, tensor<2xindex>)  -> (tensor<?x?xf32>)
    return %0 : tensor<?x?xf32>
  }
// CHECK: tensor.empty(%1, %3, %dim) : tensor<?x2x?x?xf32>
// CHECK: linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x?xf32>) outs(%4 : tensor<?x2x?x?xf32>) {
// CHECK:  tensor.collapse_shape
  
  // output type is dynamic
  // output type is dynamic
  func.func @Fused_dyn_tile_22_dynamic(%arg0: tensor<?x?xf32>, %arg1: tensor<2xindex>) -> tensor<?x?xf32> {
    %0 = "mindspore.broadcast_to"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<2xindex>)  -> (tensor<?x?xf32>)
    return %0 : tensor<?x?xf32>
  }
// CHECK: tensor.empty(%1, %dim, %3, %dim_1) : tensor<?x?x?x?xf32>
// CHECK: linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%4 : tensor<?x?x?x?xf32>) {
// CHECK: tensor.collapse_shape
}