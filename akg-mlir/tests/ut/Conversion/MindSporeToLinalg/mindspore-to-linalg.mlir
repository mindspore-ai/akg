// RUN: akg-opt %s --convert-mindspore-to-tosa -pass-pipeline="builtin.module(func.func(tosa-to-linalg))" | FileCheck %s

func.func @Fused_Cast_fusion_7941949564244712559(%arg0: tensor<1xf32>) -> tensor<1xbf16> {
  // CHECK: %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<1xf32>) outs(%0 : tensor<1xbf16>) {
  // CHECK-SAME: ^bb0(in: f32, %out: bf16):
  // CHECK-SAME:   %2 = arith.truncf %in : f32 to bf16
  // CHECK-SAME:   linalg.yield %2 : bf16
  // CHECK-SAME: } -> tensor<1xbf16> 
  %1 = "mindspore.cast"(%arg0) {dst_type = "bfloat16", ms_attr = {is_backend_cat = false}} : (tensor<1xf32>) -> tensor<1xbf16>
  return %1 : tensor<1xbf16>
}