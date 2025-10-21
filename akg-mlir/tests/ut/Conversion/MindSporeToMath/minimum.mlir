// RUN: akg-opt %s --convert-mindspore-to-linalg-named | FileCheck %s
// UNSUPPORTED: true

func.func @Fused_Minimum_fusion_2159647779082411177(%arg0: tensor<1x3072xbf16>, %arg1: tensor<1x3072xbf16>) -> tensor<1x3072xbf16> attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
  // CHECK: %[[VAL_2:.*]] = arith.minimumf %[[VAL_0:.*]], %[[VAL_1:.*]] : tensor<1x3072xbf16>
  %1 = "mindspore.minimum"(%arg0, %arg1) : (tensor<1x3072xbf16>, tensor<1x3072xbf16>) -> tensor<1x3072xbf16>
  return %1 : tensor<1x3072xbf16>
}
