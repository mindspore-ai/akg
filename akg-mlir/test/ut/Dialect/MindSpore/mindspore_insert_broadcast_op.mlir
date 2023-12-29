// RUN: akg-opt %s -pass-pipeline="builtin.module(any(infer-symbolic-shapes, make-dyn-broadcastable))" | FileCheck %s
// CHECK: mindspore.broadcast_to

module {
  func.func @Fused_Add(%arg0: tensor<32x12x12x?xf16>, %arg1: tensor<32x12x12x?xf16>) -> tensor<32x12x12x?xf16> {
    %0 = "mindspore.add"(%arg0, %arg1) : (tensor<32x12x12x?xf16>, tensor<32x12x12x?xf16>) -> tensor<32x12x12x?xf16>
    return %0 : tensor<32x12x12x?xf16>
  }
}