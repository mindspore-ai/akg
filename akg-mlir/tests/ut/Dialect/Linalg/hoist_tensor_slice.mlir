// RUN: akg-opt %s --hoist-tensor-slice | FileCheck %s

// CHECK-LABEL: #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, 0, d2, d3, d4, 0)>
// CHECK-NEXT:#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-NEXT:#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, 0)>
// CHECK-NEXT:#map3 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
// CHECK-NEXT:#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT:#map5 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
// CHECK-NEXT:#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT:#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>
// CHECK-NEXT:#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-NEXT:#map9 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, 0, d3, d4, d5)>
// CHECK-NEXT:#map10 = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, 0, d2, d3, d4, d5)>
// CHECK-NEXT:#map11 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, 0, d4, d5)>
// CHECK-NEXT:#map12 = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
// CHECK-NEXT:#map13 = affine_map<(d0, d1, d2) -> (0, d1, 0)>
// CHECK-NEXT:#map14 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT:#map15 = affine_map<(d0, d1) -> (d0)>
// CHECK-NEXT: #map16 = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-NEXT:  module {
// CHECK-NEXT:  func.func @akg_fused_mul_70(%arg0: tensor<1x1x256x256x128x1xf32>, %arg1: tensor<1x1x256x256x128xf32>, %arg2: tensor<1x256x10240xbf16>) -> (tensor<1x1x256x256x128x128xf32>, tensor<1x1x256x256x128x128xf32>) attributes {OperatorType = "Default"} {
// CHECK-NEXT:    %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:    %expanded = tensor.expand_shape %arg1 {{\[\[0\], \[1\], \[2\], \[3\], \[4, 5\]\]}} output_shape {{\[1, 1, 256, 256, 128, 1\]}} : tensor<1x1x256x256x128xf32> into tensor<1x1x256x256x128x1xf32>
// CHECK-NEXT:    %0 = tensor.empty() : tensor<1x1x256x256x128x1xf32>
// CHECK-NEXT:    %1 = linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %expanded : tensor<1x1x256x256x128x1xf32>, tensor<1x1x256x256x128x1xf32>) outs(%0 : tensor<1x1x256x256x128x1xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_8: f32, %out: f32):
// CHECK-NEXT:      %30 = arith.mulf %in, %in_8 : f32
// CHECK-NEXT:      linalg.yield %30 : f32
// CHECK-NEXT:    } -> tensor<1x1x256x256x128x1xf32>
// CHECK-NEXT:    %2 = tensor.empty() : tensor<1x1x256x256x128x128xf32>
// CHECK-NEXT:    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x1x256x256x128x1xf32>) outs(%2 : tensor<1x1x256x256x128x128xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      linalg.yield %in : f32
// CHECK-NEXT:    } -> tensor<1x1x256x256x128x128xf32>
// CHECK-NEXT:    %extracted_slice = tensor.extract_slice %arg2[0, 0, 8192] [1, 256, 1024] [1, 1, 1] : tensor<1x256x10240xbf16> to tensor<1x256x1024xbf16>
// CHECK-NEXT:    %4 = tensor.empty() : tensor<1x256x1024xf32>
// CHECK-NEXT:    %5 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x256x1024xbf16>) outs(%4 : tensor<1x256x1024xf32>) {
// CHECK-NEXT:    ^bb0(%in: bf16, %out: f32):
// CHECK-NEXT:      %30 = arith.extf %in : bf16 to f32
// CHECK-NEXT:      linalg.yield %30 : f32
// CHECK-NEXT:    } -> tensor<1x256x1024xf32>
// CHECK-NEXT:    %6 = tensor.empty() : tensor<1x256x1024xf32>
// CHECK-NEXT:    %7 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<1x256x1024xf32>) outs(%6 : tensor<1x256x1024xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      %30 = arith.negf %in : f32
// CHECK-NEXT:      %31 = math.exp %30 : f32
// CHECK-NEXT:      %32 = arith.addf %31, %cst : f32
// CHECK-NEXT:      %33 = arith.divf %cst, %32 : f32
// CHECK-NEXT:      linalg.yield %33 : f32
// CHECK-NEXT:    } -> tensor<1x256x1024xf32>
// CHECK-NEXT:    %8 = tensor.empty() : tensor<1x256x1024xf32>
// CHECK-NEXT:    %9 = linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %7 : tensor<1x256x1024xf32>, tensor<1x256x1024xf32>) outs(%8 : tensor<1x256x1024xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_8: f32, %out: f32):
// CHECK-NEXT:      %30 = arith.mulf %in, %in_8 : f32
// CHECK-NEXT:      linalg.yield %30 : f32
// CHECK-NEXT:    } -> tensor<1x256x1024xf32>
// CHECK-NEXT:    %10 = tensor.empty() : tensor<1x256x1024xbf16>
// CHECK-NEXT:    %11 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9 : tensor<1x256x1024xf32>) outs(%10 : tensor<1x256x1024xbf16>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: bf16):
// CHECK-NEXT:      %30 = arith.truncf %in : f32 to bf16
// CHECK-NEXT:      linalg.yield %30 : bf16
// CHECK-NEXT:    } -> tensor<1x256x1024xbf16>
// CHECK-NEXT:    %expanded_0 = tensor.expand_shape %11 {{\[\[0\], \[1\], \[2, 3\]\]}} output_shape {{\[1, 256, 8, 128\]}} : tensor<1x256x1024xbf16> into tensor<1x256x8x128xbf16>
// CHECK-NEXT:    %12 = tensor.empty() : tensor<1x256x8x128xf32>
// CHECK-NEXT:    %13 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_0 : tensor<1x256x8x128xbf16>) outs(%12 : tensor<1x256x8x128xf32>) {
// CHECK-NEXT:    ^bb0(%in: bf16, %out: f32):
// CHECK-NEXT:      %30 = arith.extf %in : bf16 to f32
// CHECK-NEXT:      linalg.yield %30 : f32
// CHECK-NEXT:    } -> tensor<1x256x8x128xf32>
// CHECK-NEXT:    %expanded_1 = tensor.expand_shape %13 {{\[\[0\], \[1\], \[2\], \[3, 4\]\]}} output_shape {{\[1, 256, 8, 1, 128\]}} : tensor<1x256x8x128xf32> into tensor<1x256x8x1x128xf32>
// CHECK-NEXT:    %14 = tensor.empty() : tensor<1x256x8x16x128xf32>
// CHECK-NEXT:    %15 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_1 : tensor<1x256x8x1x128xf32>) outs(%14 : tensor<1x256x8x16x128xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      linalg.yield %in : f32
// CHECK-NEXT:    } -> tensor<1x256x8x16x128xf32>
// CHECK-NEXT:    %collapsed = tensor.collapse_shape %15 {{\[\[0\], \[1\], \[2, 3\], \[4\]\]}} : tensor<1x256x8x16x128xf32> into tensor<1x256x128x128xf32>
// CHECK-NEXT:    %expanded_2 = tensor.expand_shape %collapsed {{\[\[0\], \[1, 2, 3\], \[4\], \[5\]\]}} output_shape {{\[1, 1, 1, 256, 128, 128\]}} : tensor<1x256x128x128xf32> into tensor<1x1x1x256x128x128xf32>
// CHECK-NEXT:    %16 = linalg.generic {indexing_maps = [#map9, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_2 : tensor<1x1x1x256x128x128xf32>) outs(%2 : tensor<1x1x256x256x128x128xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      linalg.yield %in : f32
// CHECK-NEXT:    } -> tensor<1x1x256x256x128x128xf32>
// CHECK-NEXT:    %17 = linalg.generic {indexing_maps = [#map10, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3, %16 : tensor<1x1x256x256x128x128xf32>, tensor<1x1x256x256x128x128xf32>) outs(%2 : tensor<1x1x256x256x128x128xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_8: f32, %out: f32):
// CHECK-NEXT:      %30 = arith.mulf %in, %in_8 : f32
// CHECK-NEXT:      linalg.yield %30 : f32
// CHECK-NEXT:    } -> tensor<1x1x256x256x128x128xf32>
// CHECK-NEXT:    %extracted_slice_3 = tensor.extract_slice %arg2[0, 0, 9216] [1, 256, 1024] [1, 1, 1] : tensor<1x256x10240xbf16> to tensor<1x256x1024xbf16>
// CHECK-NEXT:    %18 = tensor.empty() : tensor<1x256x1024xf32>
// CHECK-NEXT:    %19 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_3 : tensor<1x256x1024xbf16>) outs(%18 : tensor<1x256x1024xf32>) {
// CHECK-NEXT:    ^bb0(%in: bf16, %out: f32):
// CHECK-NEXT:      %30 = arith.extf %in : bf16 to f32
// CHECK-NEXT:      linalg.yield %30 : f32
// CHECK-NEXT:    } -> tensor<1x256x1024xf32>
// CHECK-NEXT:    %20 = tensor.empty() : tensor<1x256x1024xf32>
// CHECK-NEXT:    %21 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19 : tensor<1x256x1024xf32>) outs(%20 : tensor<1x256x1024xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      %30 = arith.negf %in : f32
// CHECK-NEXT:      %31 = math.exp %30 : f32
// CHECK-NEXT:      %32 = arith.addf %31, %cst : f32
// CHECK-NEXT:      %33 = arith.divf %cst, %32 : f32
// CHECK-NEXT:      linalg.yield %33 : f32
// CHECK-NEXT:    } -> tensor<1x256x1024xf32>
// CHECK-NEXT:    %22 = tensor.empty() : tensor<1x256x1024xf32>
// CHECK-NEXT:    %23 = linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%19, %21 : tensor<1x256x1024xf32>, tensor<1x256x1024xf32>) outs(%22 : tensor<1x256x1024xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_8: f32, %out: f32):
// CHECK-NEXT:      %30 = arith.mulf %in, %in_8 : f32
// CHECK-NEXT:      linalg.yield %30 : f32
// CHECK-NEXT:    } -> tensor<1x256x1024xf32>
// CHECK-NEXT:    %24 = tensor.empty() : tensor<1x256x1024xbf16>
// CHECK-NEXT:    %25 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%23 : tensor<1x256x1024xf32>) outs(%24 : tensor<1x256x1024xbf16>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: bf16):
// CHECK-NEXT:      %30 = arith.truncf %in : f32 to bf16
// CHECK-NEXT:      linalg.yield %30 : bf16
// CHECK-NEXT:    } -> tensor<1x256x1024xbf16>
// CHECK-NEXT:    %expanded_4 = tensor.expand_shape %25 {{\[\[0\], \[1\], \[2, 3\]\]}} output_shape {{\[1, 256, 8, 128\]}} : tensor<1x256x1024xbf16> into tensor<1x256x8x128xbf16>
// CHECK-NEXT:    %26 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4 : tensor<1x256x8x128xbf16>) outs(%12 : tensor<1x256x8x128xf32>) {
// CHECK-NEXT:    ^bb0(%in: bf16, %out: f32):
// CHECK-NEXT:      %30 = arith.extf %in : bf16 to f32
// CHECK-NEXT:      linalg.yield %30 : f32
// CHECK-NEXT:    } -> tensor<1x256x8x128xf32>
// CHECK-NEXT:    %expanded_5 = tensor.expand_shape %26 {{\[\[0\], \[1\], \[2\], \[3, 4\]\]}} output_shape {{\[1, 256, 8, 1, 128\]}} : tensor<1x256x8x128xf32> into tensor<1x256x8x1x128xf32>
// CHECK-NEXT:    %27 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_5 : tensor<1x256x8x1x128xf32>) outs(%14 : tensor<1x256x8x16x128xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      linalg.yield %in : f32
// CHECK-NEXT:    } -> tensor<1x256x8x16x128xf32>
// CHECK-NEXT:    %collapsed_6 = tensor.collapse_shape %27 {{\[\[0\], \[1\], \[2, 3\], \[4\]\]}} : tensor<1x256x8x16x128xf32> into tensor<1x256x128x128xf32>
// CHECK-NEXT:    %expanded_7 = tensor.expand_shape %collapsed_6 {{\[\[0\], \[1, 2\], \[3, 4\], \[5\]\]}} output_shape {{\[1, 1, 256, 1, 128, 128\]}} : tensor<1x256x128x128xf32> into tensor<1x1x256x1x128x128xf32>
// CHECK-NEXT:    %28 = linalg.generic {indexing_maps = [#map11, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_7 : tensor<1x1x256x1x128x128xf32>) outs(%2 : tensor<1x1x256x256x128x128xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      linalg.yield %in : f32
// CHECK-NEXT:    } -> tensor<1x1x256x256x128x128xf32>
// CHECK-NEXT:    %29 = linalg.generic {indexing_maps = [#map10, #map10, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3, %28 : tensor<1x1x256x256x128x128xf32>, tensor<1x1x256x256x128x128xf32>) outs(%2 : tensor<1x1x256x256x128x128xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_8: f32, %out: f32):
// CHECK-NEXT:      %30 = arith.mulf %in, %in_8 : f32
// CHECK-NEXT:      linalg.yield %30 : f32
// CHECK-NEXT:    } -> tensor<1x1x256x256x128x128xf32>
// CHECK-NEXT:    return %17, %29 : tensor<1x1x256x256x128x128xf32>, tensor<1x1x256x256x128x128xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @akg_fused_mul_reduce_div_slice(%arg0: tensor<1x256x10240xf32>, %arg1: tensor<1x256x10240xf32>) -> tensor<1x256x1024xf32> attributes {OperatorType = "Default"} {
// CHECK-NEXT:    %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %extracted_slice = tensor.extract_slice %arg0[0, 0, 0] [1, 256, 1024] [1, 1, 1] : tensor<1x256x10240xf32> to tensor<1x256x1024xf32>
// CHECK-NEXT:    %extracted_slice_0 = tensor.extract_slice %arg1[0, 0, 0] [1, 256, 1024] [1, 1, 1] : tensor<1x256x10240xf32> to tensor<1x256x1024xf32>
// CHECK-NEXT:    %0 = tensor.empty() : tensor<1x256x1024xf32>
// CHECK-NEXT:    %1 = linalg.generic {indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_0 : tensor<1x256x1024xf32>, tensor<1x256x1024xf32>) outs(%0 : tensor<1x256x1024xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_1: f32, %out: f32):
// CHECK-NEXT:      %7 = arith.mulf %in, %in_1 : f32
// CHECK-NEXT:      linalg.yield %7 : f32
// CHECK-NEXT:    } -> tensor<1x256x1024xf32>
// CHECK-NEXT:    %2 = tensor.empty() : tensor<1x256x1xf32>
// CHECK-NEXT:    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
// CHECK-NEXT:    %4 = linalg.generic {indexing_maps = [#map4, #map12], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : tensor<1x256x1024xf32>) outs(%3 : tensor<1x256x1xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      %7 = arith.addf %in, %out : f32
// CHECK-NEXT:      linalg.yield %7 : f32
// CHECK-NEXT:    } -> tensor<1x256x1xf32>
// CHECK-NEXT:    %5 = tensor.empty() : tensor<1x256x1024xf32>
// CHECK-NEXT:    %6 = linalg.generic {indexing_maps = [#map3, #map13, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %4 : tensor<1x256x1024xf32>, tensor<1x256x1xf32>) outs(%5 : tensor<1x256x1024xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_1: f32, %out: f32):
// CHECK-NEXT:      %7 = arith.divf %in, %in_1 : f32
// CHECK-NEXT:      linalg.yield %7 : f32
// CHECK-NEXT:    } -> tensor<1x256x1024xf32>
// CHECK-NEXT:    return %6 : tensor<1x256x1024xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @akg_fused__log_softmax_26(%arg0: tensor<512x151552xbf16>) -> tensor<511x151552xf32> attributes {OperatorType = "Default"} {
// CHECK-NEXT:    %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:    %cst = arith.constant 0xFF800000 : f32
// CHECK-NEXT:    %cst_0 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %expanded = tensor.expand_shape %arg0 {{\[\[0, 1\], \[2\]\]}} output_shape {{\[1, 512, 151552\]}} : tensor<512x151552xbf16> into tensor<1x512x151552xbf16>
// CHECK-NEXT:    %extracted_slice = tensor.extract_slice %expanded[0, 0, 0] [1, 511, 151552] [1, 1, 1] : tensor<1x512x151552xbf16> to tensor<1x511x151552xbf16>
// CHECK-NEXT:    %0 = tensor.empty() : tensor<1x511x151552xf32>
// CHECK-NEXT:    %1 = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice : tensor<1x511x151552xbf16>) outs(%0 : tensor<1x511x151552xf32>) {
// CHECK-NEXT:    ^bb0(%in: bf16, %out: f32):
// CHECK-NEXT:      %17 = arith.extf %in : bf16 to f32
// CHECK-NEXT:      linalg.yield %17 : f32
// CHECK-NEXT:    } -> tensor<1x511x151552xf32>
// CHECK-NEXT:    %collapsed = tensor.collapse_shape %1 {{\[\[0, 1\], \[2\]\]}} : tensor<1x511x151552xf32> into tensor<511x151552xf32>
// CHECK-NEXT:    %2 = tensor.empty() : tensor<511xi64>
// CHECK-NEXT:    %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<511xi64>) -> tensor<511xi64>
// CHECK-NEXT:    %4 = tensor.empty() : tensor<511xf32>
// CHECK-NEXT:    %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<511xf32>) -> tensor<511xf32>
// CHECK-NEXT:    %6:2 = linalg.generic {indexing_maps = [#map14, #map15, #map15], iterator_types = ["parallel", "reduction"]} ins(%collapsed : tensor<511x151552xf32>) outs(%5, %3 : tensor<511xf32>, tensor<511xi64>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32, %out_2: i64):
// CHECK-NEXT:      %17 = linalg.index 1 : index
// CHECK-NEXT:      %18 = arith.index_cast %17 : index to i64
// CHECK-NEXT:      %19 = arith.maximumf %in, %out : f32
// CHECK-NEXT:      %20 = arith.cmpf ogt, %in, %out : f32
// CHECK-NEXT:      %21 = arith.select %20, %18, %out_2 : i64
// CHECK-NEXT:      linalg.yield %19, %21 : f32, i64
// CHECK-NEXT:    } -> (tensor<511xf32>, tensor<511xi64>)
// CHECK-NEXT:    %expanded_1 = tensor.expand_shape %6#0 {{\[\[0, 1\]\] output_shape \[511, 1\]}} : tensor<511xf32> into tensor<511x1xf32>
// CHECK-NEXT:    %7 = tensor.empty() : tensor<511x151552xf32>
// CHECK-NEXT:    %8 = linalg.generic {indexing_maps = [#map16, #map14], iterator_types = ["parallel", "parallel"]} ins(%expanded_1 : tensor<511x1xf32>) outs(%7 : tensor<511x151552xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      linalg.yield %in : f32
// CHECK-NEXT:    } -> tensor<511x151552xf32>
// CHECK-NEXT:    %9 = linalg.generic {indexing_maps = [#map14, #map14, #map14], iterator_types = ["parallel", "parallel"]} ins(%collapsed, %8 : tensor<511x151552xf32>, tensor<511x151552xf32>) outs(%7 : tensor<511x151552xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_2: f32, %out: f32):
// CHECK-NEXT:      %17 = arith.subf %in, %in_2 : f32
// CHECK-NEXT:      linalg.yield %17 : f32
// CHECK-NEXT:    } -> tensor<511x151552xf32>
// CHECK-NEXT:    %10 = linalg.generic {indexing_maps = [#map14, #map14], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<511x151552xf32>) outs(%7 : tensor<511x151552xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      %17 = math.exp %in : f32
// CHECK-NEXT:      linalg.yield %17 : f32
// CHECK-NEXT:    } -> tensor<511x151552xf32>
// CHECK-NEXT:    %11 = tensor.empty() : tensor<511x1xf32>
// CHECK-NEXT:    %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<511x1xf32>) -> tensor<511x1xf32>
// CHECK-NEXT:    %13 = linalg.generic {indexing_maps = [#map14, #map16], iterator_types = ["parallel", "reduction"]} ins(%10 : tensor<511x151552xf32>) outs(%12 : tensor<511x1xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      %17 = arith.addf %in, %out : f32
// CHECK-NEXT:      linalg.yield %17 : f32
// CHECK-NEXT:    } -> tensor<511x1xf32>
// CHECK-NEXT:    %14 = linalg.generic {indexing_maps = [#map16, #map14], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<511x1xf32>) outs(%11 : tensor<511x1xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      %17 = math.log %in : f32
// CHECK-NEXT:      linalg.yield %17 : f32
// CHECK-NEXT:    } -> tensor<511x1xf32>
// CHECK-NEXT:    %15 = linalg.generic {indexing_maps = [#map16, #map14], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<511x1xf32>) outs(%7 : tensor<511x151552xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %out: f32):
// CHECK-NEXT:      linalg.yield %in : f32
// CHECK-NEXT:    } -> tensor<511x151552xf32>
// CHECK-NEXT:    %16 = linalg.generic {indexing_maps = [#map14, #map14, #map14], iterator_types = ["parallel", "parallel"]} ins(%9, %15 : tensor<511x151552xf32>, tensor<511x151552xf32>) outs(%7 : tensor<511x151552xf32>) {
// CHECK-NEXT:    ^bb0(%in: f32, %in_2: f32, %out: f32):
// CHECK-NEXT:      %17 = arith.subf %in, %in_2 : f32
// CHECK-NEXT:      linalg.yield %17 : f32
// CHECK-NEXT:    } -> tensor<511x151552xf32>
// CHECK-NEXT:    return %16 : tensor<511x151552xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:}

func.func @akg_fused_mul_70(%arg0: tensor<1x1x256x256x128x1xf32>, %arg1: tensor<1x1x256x256x128xf32>, %arg2: tensor<1x256x10240xbf16>) -> (tensor<1x1x256x256x128x128xf32>, tensor<1x1x256x256x128x128xf32>) attributes {OperatorType = "Default"} {
  %cst = arith.constant 1.000000e+00 : f32
  %expanded = tensor.expand_shape %arg1 [[0], [1], [2], [3], [4, 5]] output_shape [1, 1, 256, 256, 128, 1] : tensor<1x1x256x256x128xf32> into tensor<1x1x256x256x128x1xf32>
  %0 = tensor.empty() : tensor<1x1x256x256x128x1xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (0, 0, d2, d3, d4, 0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (0, 0, d2, d3, d4, 0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %expanded : tensor<1x1x256x256x128x1xf32>, tensor<1x1x256x256x128x1xf32>) outs(%0 : tensor<1x1x256x256x128x1xf32>) {
  ^bb0(%in: f32, %in_8: f32, %out: f32):
    %20 = arith.mulf %in, %in_8 : f32
    linalg.yield %20 : f32
  } -> tensor<1x1x256x256x128x1xf32>
  %2 = tensor.empty() : tensor<1x1x256x256x128x128xf32>
  %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, 0)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<1x1x256x256x128x1xf32>) outs(%2 : tensor<1x1x256x256x128x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x1x256x256x128x128xf32>
  %4 = tensor.empty() : tensor<1x256x10240xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x256x10240xbf16>) outs(%4 : tensor<1x256x10240xf32>) {
  ^bb0(%in: bf16, %out: f32):
    %20 = arith.extf %in : bf16 to f32
    linalg.yield %20 : f32
  } -> tensor<1x256x10240xf32>
  %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<1x256x10240xf32>) outs(%4 : tensor<1x256x10240xf32>) {
  ^bb0(%in: f32, %out: f32):
    %20 = arith.negf %in : f32
    %21 = math.exp %20 : f32
    %22 = arith.addf %21, %cst : f32
    %23 = arith.divf %cst, %22 : f32
    linalg.yield %23 : f32
  } -> tensor<1x256x10240xf32>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6 : tensor<1x256x10240xf32>, tensor<1x256x10240xf32>) outs(%4 : tensor<1x256x10240xf32>) {
  ^bb0(%in: f32, %in_8: f32, %out: f32):
    %20 = arith.mulf %in, %in_8 : f32
    linalg.yield %20 : f32
  } -> tensor<1x256x10240xf32>
  %8 = tensor.empty() : tensor<1x256x10240xbf16>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%7 : tensor<1x256x10240xf32>) outs(%8 : tensor<1x256x10240xbf16>) {
  ^bb0(%in: f32, %out: bf16):
    %20 = arith.truncf %in : f32 to bf16
    linalg.yield %20 : bf16
  } -> tensor<1x256x10240xbf16>
  %extracted_slice = tensor.extract_slice %9[0, 0, 8192] [1, 256, 1024] [1, 1, 1] : tensor<1x256x10240xbf16> to tensor<1x256x1024xbf16>
  %expanded_0 = tensor.expand_shape %extracted_slice [[0], [1], [2, 3]] output_shape [1, 256, 8, 128] : tensor<1x256x1024xbf16> into tensor<1x256x8x128xbf16>
  %10 = tensor.empty() : tensor<1x256x8x128xf32>
  %11 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_0 : tensor<1x256x8x128xbf16>) outs(%10 : tensor<1x256x8x128xf32>) {
  ^bb0(%in: bf16, %out: f32):
    %20 = arith.extf %in : bf16 to f32
    linalg.yield %20 : f32
  } -> tensor<1x256x8x128xf32>
  %expanded_1 = tensor.expand_shape %11 [[0], [1], [2], [3, 4]] output_shape [1, 256, 8, 1, 128] : tensor<1x256x8x128xf32> into tensor<1x256x8x1x128xf32>
  %12 = tensor.empty() : tensor<1x256x8x16x128xf32>
  %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_1 : tensor<1x256x8x1x128xf32>) outs(%12 : tensor<1x256x8x16x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x256x8x16x128xf32>
  %collapsed = tensor.collapse_shape %13 [[0], [1], [2, 3], [4]] : tensor<1x256x8x16x128xf32> into tensor<1x256x128x128xf32>
  %expanded_2 = tensor.expand_shape %collapsed [[0], [1, 2, 3], [4], [5]] output_shape [1, 1, 1, 256, 128, 128] : tensor<1x256x128x128xf32> into tensor<1x1x1x256x128x128xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, 0, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_2 : tensor<1x1x1x256x128x128xf32>) outs(%2 : tensor<1x1x256x256x128x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x1x256x256x128x128xf32>
  %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (0, 0, d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (0, 0, d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3, %14 : tensor<1x1x256x256x128x128xf32>, tensor<1x1x256x256x128x128xf32>) outs(%2 : tensor<1x1x256x256x128x128xf32>) {
  ^bb0(%in: f32, %in_8: f32, %out: f32):
    %20 = arith.mulf %in, %in_8 : f32
    linalg.yield %20 : f32
  } -> tensor<1x1x256x256x128x128xf32>
  %extracted_slice_3 = tensor.extract_slice %9[0, 0, 9216] [1, 256, 1024] [1, 1, 1] : tensor<1x256x10240xbf16> to tensor<1x256x1024xbf16>
  %expanded_4 = tensor.expand_shape %extracted_slice_3 [[0], [1], [2, 3]] output_shape [1, 256, 8, 128] : tensor<1x256x1024xbf16> into tensor<1x256x8x128xbf16>
  %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_4 : tensor<1x256x8x128xbf16>) outs(%10 : tensor<1x256x8x128xf32>) {
  ^bb0(%in: bf16, %out: f32):
    %20 = arith.extf %in : bf16 to f32
    linalg.yield %20 : f32
  } -> tensor<1x256x8x128xf32>
  %expanded_5 = tensor.expand_shape %16 [[0], [1], [2], [3, 4]] output_shape [1, 256, 8, 1, 128] : tensor<1x256x8x128xf32> into tensor<1x256x8x1x128xf32>
  %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_5 : tensor<1x256x8x1x128xf32>) outs(%12 : tensor<1x256x8x16x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x256x8x16x128xf32>
  %collapsed_6 = tensor.collapse_shape %17 [[0], [1], [2, 3], [4]] : tensor<1x256x8x16x128xf32> into tensor<1x256x128x128xf32>
  %expanded_7 = tensor.expand_shape %collapsed_6 [[0], [1, 2], [3, 4], [5]] output_shape [1, 1, 256, 1, 128, 128] : tensor<1x256x128x128xf32> into tensor<1x1x256x1x128x128xf32>
  %18 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, 0, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_7 : tensor<1x1x256x1x128x128xf32>) outs(%2 : tensor<1x1x256x256x128x128xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<1x1x256x256x128x128xf32>
  %19 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (0, 0, d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (0, 0, d2, d3, d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3, %18 : tensor<1x1x256x256x128x128xf32>, tensor<1x1x256x256x128x128xf32>) outs(%2 : tensor<1x1x256x256x128x128xf32>) {
  ^bb0(%in: f32, %in_8: f32, %out: f32):
    %20 = arith.mulf %in, %in_8 : f32
    linalg.yield %20 : f32
  } -> tensor<1x1x256x256x128x128xf32>
  return %15, %19 : tensor<1x1x256x256x128x128xf32>, tensor<1x1x256x256x128x128xf32>
}

func.func @akg_fused_mul_reduce_div_slice(%arg0: tensor<1x256x10240xf32>, %arg1: tensor<1x256x10240xf32>) -> tensor<1x256x1024xf32> attributes {OperatorType = "Default"} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<1x256x10240xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x256x10240xf32>, tensor<1x256x10240xf32>) outs(%0 : tensor<1x256x10240xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.mulf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x256x10240xf32>
  %2 = tensor.empty() : tensor<1x256x1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x256x1xf32>) -> tensor<1x256x1xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, 0)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1 : tensor<1x256x10240xf32>) outs(%3 : tensor<1x256x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> tensor<1x256x1xf32>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (0, d1, 0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %4 : tensor<1x256x10240xf32>, tensor<1x256x1xf32>) outs(%0 : tensor<1x256x10240xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %6 = arith.divf %in, %in_0 : f32
    linalg.yield %6 : f32
  } -> tensor<1x256x10240xf32>
  %extracted_slice = tensor.extract_slice %5[0, 0, 0] [1, 256, 1024] [1, 1, 1] : tensor<1x256x10240xf32> to tensor<1x256x1024xf32>
  return %extracted_slice : tensor<1x256x1024xf32>
}

func.func @akg_fused__log_softmax_26(%arg0: tensor<512x151552xbf16>) -> tensor<511x151552xf32> attributes {OperatorType = "Default"} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %expanded = tensor.expand_shape %arg0 [[0, 1], [2]] output_shape [1, 512, 151552] : tensor<512x151552xbf16> into tensor<1x512x151552xbf16>
  %0 = tensor.empty() : tensor<1x512x151552xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%expanded : tensor<1x512x151552xbf16>) outs(%0 : tensor<1x512x151552xf32>) {
  ^bb0(%in: bf16, %out: f32):
    %17 = arith.extf %in : bf16 to f32
    linalg.yield %17 : f32
  } -> tensor<1x512x151552xf32>
  %extracted_slice = tensor.extract_slice %1[0, 0, 0] [1, 511, 151552] [1, 1, 1] : tensor<1x512x151552xf32> to tensor<1x511x151552xf32>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2]] : tensor<1x511x151552xf32> into tensor<511x151552xf32>
  %2 = tensor.empty() : tensor<511xi64>
  %3 = linalg.fill ins(%c0_i64 : i64) outs(%2 : tensor<511xi64>) -> tensor<511xi64>
  %4 = tensor.empty() : tensor<511xf32>
  %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<511xf32>) -> tensor<511xf32>
  %6:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%collapsed : tensor<511x151552xf32>) outs(%5, %3 : tensor<511xf32>, tensor<511xi64>) {
  ^bb0(%in: f32, %out: f32, %out_2: i64):
    %17 = linalg.index 1 : index
    %18 = arith.index_cast %17 : index to i64
    %19 = arith.maximumf %in, %out : f32
    %20 = arith.cmpf ogt, %in, %out : f32
    %21 = arith.select %20, %18, %out_2 : i64
    linalg.yield %19, %21 : f32, i64
  } -> (tensor<511xf32>, tensor<511xi64>)
  %expanded_1 = tensor.expand_shape %6#0 [[0, 1]] output_shape [511, 1] : tensor<511xf32> into tensor<511x1xf32>
  %7 = tensor.empty() : tensor<511x151552xf32>
  %8 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%expanded_1 : tensor<511x1xf32>) outs(%7 : tensor<511x151552xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<511x151552xf32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%collapsed, %8 : tensor<511x151552xf32>, tensor<511x151552xf32>) outs(%7 : tensor<511x151552xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %17 = arith.subf %in, %in_2 : f32
    linalg.yield %17 : f32
  } -> tensor<511x151552xf32>
  %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9 : tensor<511x151552xf32>) outs(%7 : tensor<511x151552xf32>) {
  ^bb0(%in: f32, %out: f32):
    %17 = math.exp %in : f32
    linalg.yield %17 : f32
  } -> tensor<511x151552xf32>
  %11 = tensor.empty() : tensor<511x1xf32>
  %12 = linalg.fill ins(%cst_0 : f32) outs(%11 : tensor<511x1xf32>) -> tensor<511x1xf32>
  %13 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>], iterator_types = ["parallel", "reduction"]} ins(%10 : tensor<511x151552xf32>) outs(%12 : tensor<511x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %17 = arith.addf %in, %out : f32
    linalg.yield %17 : f32
  } -> tensor<511x1xf32>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%13 : tensor<511x1xf32>) outs(%11 : tensor<511x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %17 = math.log %in : f32
    linalg.yield %17 : f32
  } -> tensor<511x1xf32>
  %15 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%14 : tensor<511x1xf32>) outs(%7 : tensor<511x151552xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<511x151552xf32>
  %16 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%9, %15 : tensor<511x151552xf32>, tensor<511x151552xf32>) outs(%7 : tensor<511x151552xf32>) {
  ^bb0(%in: f32, %in_2: f32, %out: f32):
    %17 = arith.subf %in, %in_2 : f32
    linalg.yield %17 : f32
  } -> tensor<511x151552xf32>
  return %16 : tensor<511x151552xf32>
}
