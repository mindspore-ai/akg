// RUN: akg-opt %s -split-input-file --eliminate-reshape | FileCheck %s

// CHECK-LABEL: func.func @dynamic_reshape_broadcast(
// CHECK:       "mindspore.reshape"

func.func @dynamic_reshape_broadcast(%arg0: tensor<?x1024xf16, {SymShapeAttr = ["s349", "1024"]}>, %arg1: tensor<?x1024xf16, {SymShapeAttr = ["s350", "1024"]}>, %arg2: tensor<3xi64, {SymShapeAttr = ["3"]}>) -> tensor<32x?x1024xf16, {SymShapeAttr = ["s352", "s353", "s354"]}> attributes {OperatorType = "Elementwise", compute_capability = "8.0", mindspore_kernel, process = "cuda"} {
  %cast = tensor.cast %arg0 : tensor<?x1024xf16, {SymShapeAttr = ["s349", "1024"]}> to tensor<?x1024xf16, {SymShapeAttr = ["s351", "1024"]}>
  %cast_0 = tensor.cast %arg1 : tensor<?x1024xf16, {SymShapeAttr = ["s350", "1024"]}> to tensor<?x1024xf16, {SymShapeAttr = ["s351", "1024"]}>
  %0 = "tosa.add"(%cast, %cast_0) : (tensor<?x1024xf16, {SymShapeAttr = ["s351", "1024"]}>, tensor<?x1024xf16, {SymShapeAttr = ["s351", "1024"]}>) -> tensor<?x1024xf16, {SymShapeAttr = ["s351", "1024"]}>
  %1 = "mindspore.reshape"(%0, %arg2) {ms_attr = {input_is_dynamic_shape = true, output_is_dynamic_shape = true}, ptr_address = "Gradients/Default/network/network/loss/gradReshape-expand/Reshape-op10089_670233"} : (tensor<?x1024xf16, {SymShapeAttr = ["s351", "1024"]}>, tensor<3xi64, {SymShapeAttr = ["3"]}>) -> tensor<32x?x1024xf16, {SymShapeAttr = ["s352", "s353", "s354"]}>
  return %1 : tensor<32x?x1024xf16, {SymShapeAttr = ["s352", "s353", "s354"]}>
}

// -----

// CHECK-LABEL: func.func @dynamic_reshape_elemwise(
// CHECK-NOT:   "mindspore.reshape"

func.func @dynamic_reshape_elemwise(%arg0: tensor<?x768xf32, {SymShapeAttr = ["s0", "768"]}>, %arg1: tensor<768xf32, {SymShapeAttr = ["768"]}>) -> tensor<?x768xf32, {SymShapeAttr = ["s0", "768"]}> attributes {OperatorType = "Broadcast", compute_capability = "", mindspore_kernel, process = "cuda"} {
  %0 = "mindspore.reshape"(%arg1) {new_shape = array<i64: 1>} : (tensor<768xf32, {SymShapeAttr = ["768"]}>) -> tensor<1x768xf32, {SymShapeAttr = ["1", "768"]}>
  %1 = "tosa.add"(%arg0, %0) : (tensor<?x768xf32, {SymShapeAttr = ["s0", "768"]}>, tensor<1x768xf32, {SymShapeAttr = ["1", "768"]}>) -> tensor<?x768xf32, {SymShapeAttr = ["s0", "768"]}>
  return %1 : tensor<?x768xf32, {SymShapeAttr = ["s0", "768"]}>
}

