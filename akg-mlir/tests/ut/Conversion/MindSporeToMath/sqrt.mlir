// RUN: akg-opt %s --convert-mindspore-to-linalg-named | FileCheck %s

func.func @Fused_Sqrt_fusion_2159647779082411177(%arg0: tensor<1x3072xbf16>) -> tensor<1x3072xbf16> attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
  // CHECK: %[[VAL_1:.*]] = math.sqrt %[[VAL_0:.*]]: tensor<1x3072xbf16>
  %1 = "mindspore.sqrt"(%arg0) : (tensor<1x3072xbf16>) -> tensor<1x3072xbf16>
  return %1 : tensor<1x3072xbf16>
}
