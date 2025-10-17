// RUN: akg-opt %s --convert-mindspore-to-linalg-named | FileCheck %s

func.func @Fused_Select_fusion_2159647779082411177(%arg0: tensor<1x3072xi1>, %arg1: tensor<1x3072xf32>, %arg2: tensor<1x3072xf32>) -> tensor<1x3072xf32> {
  // CHECK: linalg.select
  // CHECK-SAME: ins(%{{.+}}, %{{.+}}, %{{.+}} : tensor<1x3072xi1>, tensor<1x3072xf32>, tensor<1x3072xf32>)
  // CHECK-SAME: outs(%{{.+}} : tensor<1x3072xf32>)
  %1 = "mindspore.select"(%arg0, %arg1, %arg2) : (tensor<1x3072xi1>, tensor<1x3072xf32>, tensor<1x3072xf32>) -> tensor<1x3072xf32>
  return %1 : tensor<1x3072xf32>
}