// RUN: akg-opt %s -split-input-file --linalg-fuse-elementwise-ext | FileCheck %s

// CHECK-LABEL: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT: #map1 = affine_map<(d0) -> (d0)>
// CHECK-NEXT: #map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: #map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func @Fused_Add_Cast_fusion_16268886576649085536(%arg0: tensor<?x1024xf16>, %arg1: tensor<?x1024xf16>) -> (tensor<?x1024xf16>, tensor<?x1024xf32>)
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %dim = tensor.dim %arg0, %c0 : tensor<?x1024xf16>
// CHECK-NEXT:     %0 = tensor.empty(%dim) : tensor<?x1024xf16>
// CHECK-NEXT:     %1 = tensor.empty(%dim) : tensor<?x1024xf32>
// CHECK-NEXT:     %2:2 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x1024xf16>, tensor<?x1024xf16>) outs(%0, %1 : tensor<?x1024xf16>, tensor<?x1024xf32>) {
// CHECK-NEXT:     ^bb0(%in: f16, %in_0: f16, %out: f16, %out_1: f32):
// CHECK-NEXT:       %3 = arith.addf %in, %in_0 : f16
// CHECK-NEXT:       %4 = arith.extf %3 : f16 to f32
// CHECK-NEXT:       linalg.yield %3, %4 : f16, f32
// CHECK-NEXT:     } -> (tensor<?x1024xf16>, tensor<?x1024xf32>)
// CHECK-NEXT:     return %2#0, %2#1 : tensor<?x1024xf16>, tensor<?x1024xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @Fused_Cast_Neg_Mul_fusion_3304020172462508230(%arg0: tensor<?xi64>, %arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %dim = tensor.dim %arg0, %c0 : tensor<?xi64>
// CHECK-NEXT:     %0 = tensor.empty(%dim) : tensor<?xf32>
// CHECK-NEXT:     %1 = tensor.empty(%dim) : tensor<?xf32>
// CHECK-NEXT:     %2 = tensor.empty(%dim) : tensor<?xf32>
// CHECK-NEXT:     %3:3 = linalg.generic {indexing_maps = [#map1, #map1, #map1, #map1, #map1], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<?xi64>, tensor<?xf32>) outs(%1, %0, %2 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) {
// CHECK-NEXT:     ^bb0(%in: i64, %in_0: f32, %out: f32, %out_1: f32, %out_2: f32):
// CHECK-NEXT:       %4 = arith.negf %in_0 : f32
// CHECK-NEXT:       %5 = arith.sitofp %in : i64 to f32
// CHECK-NEXT:       %6 = arith.mulf %5, %4 : f32
// CHECK-NEXT:       linalg.yield %4, %5, %6 : f32, f32, f32
// CHECK-NEXT:     } -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>)
// CHECK-NEXT:     return %3#1, %3#0, %3#2 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   func.func @Fused_Log_Sub_Mul_split_8402738215490280482(%arg0: tensor<?x1xf32>, %arg1: tensor<?x36560xf32>, %arg2: tensor<?x36560xf32>) -> (tensor<?x36560xf32>, tensor<?x36560xf32>)
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %dim = tensor.dim %arg0, %c0 : tensor<?x1xf32>
// CHECK-NEXT:     %0 = tensor.empty(%dim) : tensor<?x36560xf32>
// CHECK-NEXT:     %expanded = tensor.expand_shape
// CHECK-NEXT:     %dim_0 = tensor.dim %0, %c0 : tensor<?x36560xf32>
// CHECK-NEXT:     %1 = tensor.empty(%dim_0) : tensor<?x1x36560xf32>
// CHECK-NEXT:     %2 = tensor.empty(%dim) : tensor<?x36560xf32>
// CHECK-NEXT:     %expanded_1 = tensor.expand_shape
// CHECK-NEXT:     %expanded_2 = tensor.expand_shape
// CHECK-NEXT:     %3:2 = linalg.generic {indexing_maps = [#map2, #map3, #map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %arg0, %expanded_1 : tensor<?x1x36560xf32>, tensor<?x1xf32>, tensor<?x1x36560xf32>) outs(%1, %expanded_2 : tensor<?x1x36560xf32>, tensor<?x1x36560xf32>) {
// CHECK-NEXT:     ^bb0(%in: f32, %in_4: f32, %in_5: f32, %out: f32, %out_6: f32):
// CHECK-NEXT:       %4 = math.log %in_4 : f32
// CHECK-NEXT:       %5 = arith.subf %in, %4 : f32
// CHECK-NEXT:       %6 = arith.mulf %5, %in_5 : f32
// CHECK-NEXT:       linalg.yield %5, %6 : f32, f32
// CHECK-NEXT:     } -> (tensor<?x1x36560xf32>, tensor<?x1x36560xf32>)
// CHECK-NEXT:     %collapsed = tensor.collapse_shape %3#1
// CHECK-NEXT:     %collapsed_3 = tensor.collapse_shape %3#0
// CHECK-NEXT:     return %collapsed_3, %collapsed : tensor<?x36560xf32>, tensor<?x36560xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
func.func @Fused_Add_Cast_fusion_16268886576649085536(%arg0: tensor<?x1024xf16>, %arg1: tensor<?x1024xf16>) -> (tensor<?x1024xf16>, tensor<?x1024xf32>) attributes {OperatorType = "Elementwise", enable_atomic_add = false, mindspore_kernel, need_fix = [[1 : i32, 0 : i32], [1 : i32, 0 : i32]], process = "cuda"} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x1024xf16>
  %0 = tensor.empty(%dim) : tensor<?x1024xf16>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x1024xf16>, tensor<?x1024xf16>) outs(%0 : tensor<?x1024xf16>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %4 = arith.addf %in, %in_0 : f16
    linalg.yield %4 : f16
  } -> tensor<?x1024xf16>
  %2 = tensor.empty(%dim) : tensor<?x1024xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<?x1024xf16>) outs(%2 : tensor<?x1024xf32>) {
  ^bb0(%in: f16, %out: f32):
    %4 = arith.extf %in : f16 to f32
    linalg.yield %4 : f32
  } -> tensor<?x1024xf32>
  return %1, %3 : tensor<?x1024xf16>, tensor<?x1024xf32>
}

func.func @Fused_Cast_Neg_Mul_fusion_3304020172462508230(%arg0: tensor<?xi64>, %arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) attributes {OperatorType = "Elementwise", enable_atomic_add = false, mindspore_kernel, need_fix = [[0 : i32], [0 : i32]], process = "cuda"} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?xi64>
  %0 = tensor.empty(%dim) : tensor<?xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0 : tensor<?xi64>) outs(%0 : tensor<?xf32>) {
  ^bb0(%in: i64, %out: f32):
    %6 = arith.sitofp %in : i64 to f32
    linalg.yield %6 : f32
  } -> tensor<?xf32>
  %2 = tensor.empty(%dim) : tensor<?xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg1 : tensor<?xf32>) outs(%2 : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.negf %in : f32
    linalg.yield %6 : f32
  } -> tensor<?xf32>
  %4 = tensor.empty(%dim) : tensor<?xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1, %3 : tensor<?xf32>, tensor<?xf32>) outs(%4 : tensor<?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.mulf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<?xf32>
  return %1, %3, %5 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>
}

func.func @Fused_Log_Sub_Mul_split_8402738215490280482(%arg0: tensor<?x1xf32>, %arg1: tensor<?x36560xf32>, %arg2: tensor<?x36560xf32>) -> (tensor<?x36560xf32>, tensor<?x36560xf32>) attributes {OperatorType = "Broadcast", enable_atomic_add = false, mindspore_kernel, need_fix = [[0 : i32, 0 : i32], [1 : i32, 0 : i32], [1 : i32, 0 : i32]], process = "cuda"} {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x1xf32>
  %0 = tensor.empty(%dim) : tensor<?x36560xf32>
  %expanded = tensor.expand_shape %arg1 [[0, 1], [2]] : tensor<?x36560xf32> into tensor<?x1x36560xf32>
  %dim_0 = tensor.dim %0, %c0 : tensor<?x36560xf32>
  %1 = tensor.empty(%dim_0) : tensor<?x1x36560xf32>
  %2 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded, %arg0 : tensor<?x1x36560xf32>, tensor<?x1xf32>) outs(%1 : tensor<?x1x36560xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %5 = math.log %in_1 : f32
    %6 = arith.subf %in, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<?x1x36560xf32>
  %collapsed = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<?x1x36560xf32> into tensor<?x36560xf32>
  %3 = tensor.empty(%dim) : tensor<?x36560xf32>
  %4 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%collapsed, %arg2 : tensor<?x36560xf32>, tensor<?x36560xf32>) outs(%3 : tensor<?x36560xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %5 = arith.mulf %in, %in_1 : f32
    linalg.yield %5 : f32
  } -> tensor<?x36560xf32>
  return %collapsed, %4 : tensor<?x36560xf32>, tensor<?x36560xf32>
}
}