// RUN: akg-opt %s --convert-arith-to-linalg | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: arith.addf -> linalg.elemwise_binary with add
//===----------------------------------------------------------------------===//

func.func @Fused_Addf_fusion_2159647779082411177(%arg0: tensor<4xbf16>, %arg1: tensor<4xbf16>) -> tensor<4xbf16> attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
  // CHECK: %[[VAL_2:.*]] = linalg.elemwise_binary %[[VAL_0:.*]], %[[VAL_1:.*]] : tensor<4xbf16>
  %0 = arith.addf %arg0, %arg1 : tensor<4xbf16>
  return %0 : tensor<4xbf16>
}

//===----------------------------------------------------------------------===//
// Test: math.ceil -> linalg.elemwise_unary with ceil
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_ceil
func.func @Fused_Ceil_fusion_2159647779082411177(%arg0: tensor<4xbf16>) -> tensor<4xbf16> {
  // CHECK: %[[VAL_1:.*]] = linalg.elemwise_binary %[[VAL_0:.*]] : tensor<4xbf16>
  %0 = math.ceil %arg0 : tensor<4xbf16>
  return %0 : tensor<4xbf16>
}