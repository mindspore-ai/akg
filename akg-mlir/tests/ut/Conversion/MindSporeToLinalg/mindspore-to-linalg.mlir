// RUN: akg-opt %s -pass-pipeline="builtin.module(func.func(convert-mindspore-to-tosa,tosa-to-linalg))" | FileCheck %s

func.func @Fused_Cast_fusion_7941949564244712559(%arg0: tensor<1xf32>) -> tensor<1xbf16> {
  // CHECK: %[[GENERIC:.+]] = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel"]} ins(%arg0 : tensor<1xf32>) outs(%0 : tensor<1xbf16>) {
  // CHECK: ^bb0(%[[IN:.+]]: f32, %{{.+}}: bf16):
  // CHECK:   %[[TRUNCF:.+]] = arith.truncf %[[IN:.+]] : f32 to bf16
  // CHECK:   linalg.yield %[[TRUNCF:.+]] : bf16
  // CHECK: } -> tensor<1xbf16> 
  %1 = "mindspore.cast"(%arg0) {dst_type = "bfloat16", ms_attr = {is_backend_cat = false}} : (tensor<1xf32>) -> tensor<1xbf16>
  return %1 : tensor<1xbf16>
}