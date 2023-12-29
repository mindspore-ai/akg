// RUN: akg-opt %s --pass-pipeline="builtin.module(any(make-dyn-broadcastable{ignore-implicit-broadcast=true}))" | FileCheck %s --check-prefix=CHECK1
// RUN: akg-opt %s --pass-pipeline="builtin.module(any(make-dyn-broadcastable{ignore-implicit-broadcast=false}))" | FileCheck %s --check-prefix=CHECK2
// CHECK1: tensor.cast
// CHECK1-NEXT: "mindspore.add"(%cast, %1)
// CHECK2: shape.shape_of
// CHECK2-NEXT: mindspore.broadcast_to
// CHECK2-NEXT: mindspore.add

module {
  func.func @test_make_dymamic_broadcastable(%arg0: tensor<?x1x?x?xf32, {SymShapeAttr = ["s2", "1", "s0", "s1"]}>, %arg1: tensor<32x12x?x?xf32, {SymShapeAttr = ["32", "12", "s0", "s1"]}>) -> tensor<32x12x?x?xf32, {SymShapeAttr = ["32", "12", "s0", "s1"]}> {
    %0 = "mindspore.const"() {value = dense<1.250000e-01> : tensor<1xf32>} : () -> tensor<1xf32, {SymShapeAttr = ["1"]}>
    %1 = "mindspore.mul"(%0, %arg1) : (tensor<1xf32, {SymShapeAttr = ["1"]}>, tensor<32x12x?x?xf32, {SymShapeAttr = ["32", "12", "s0", "s1"]}>) -> tensor<32x12x?x?xf32, {SymShapeAttr = ["32", "12", "s0", "s1"]}>
    %2 = "mindspore.add"(%arg0, %1) : (tensor<?x1x?x?xf32, {SymShapeAttr = ["s2", "1", "s0", "s1"]}>, tensor<32x12x?x?xf32, {SymShapeAttr = ["32", "12", "s0", "s1"]}>) -> tensor<32x12x?x?xf32, {SymShapeAttr = ["32", "12", "s0", "s1"]}>
    return %2 : tensor<32x12x?x?xf32, {SymShapeAttr = ["32", "12", "s0", "s1"]}>
  }
}
