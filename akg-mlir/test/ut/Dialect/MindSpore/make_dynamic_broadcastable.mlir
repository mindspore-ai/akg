// RUN: akg-opt %s --pass-pipeline="builtin.module(any(make-dyn-broadcastable{ignore-implicit-broadcast=true}))" | FileCheck %s

module {
  func.func @elem_broadcast_last_5(%arg0: tensor<4096x?xf32, {SymShapeAttr = ["4096", "s0"]}>, %arg1: tensor<?xf32, {SymShapeAttr = ["s1"]}>, %arg2: tensor<1x?xf32, {SymShapeAttr = ["1", "s3"]}>) -> tensor<4096x?xf32, {SymShapeAttr = ["4096", "s24"]}> attributes {enable_atomic_add = false, mindspore_kernel, process = "cuda"} {
    %0 = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, -1>} : (tensor<?xf32, {SymShapeAttr = ["s1"]}>) -> tensor<1x?xf32, {SymShapeAttr = ["1", "s1"]}>
	//CHECK: tensor.cast %arg0 : tensor<4096x?xf32, {SymShapeAttr = ["4096", "s0"]}> to tensor<4096x?xf32, {SymShapeAttr = ["4096", "s23"]}>
	//CHECK: tensor.cast %0 : tensor<1x?xf32, {SymShapeAttr = ["1", "s1"]}> to tensor<1x?xf32, {SymShapeAttr = ["1", "s23"]}>
    %1 = "tosa.add"(%arg0, %0) : (tensor<4096x?xf32, {SymShapeAttr = ["4096", "s0"]}>, tensor<1x?xf32, {SymShapeAttr = ["1", "s1"]}>) -> tensor<4096x?xf32, {SymShapeAttr = ["4096", "s23"]}>
	//CHECK: tensor.cast %arg2 : tensor<1x?xf32, {SymShapeAttr = ["1", "s3"]}> to tensor<1x?xf32, {SymShapeAttr = ["1", "s24"]}>
	//CHECK: tensor.cast %1 : tensor<4096x?xf32, {SymShapeAttr = ["4096", "s23"]}> to tensor<4096x?xf32, {SymShapeAttr = ["4096", "s24"]}>
    %2 = "tosa.mul"(%arg2, %1) {shift = 0 : i32} : (tensor<1x?xf32, {SymShapeAttr = ["1", "s3"]}>, tensor<4096x?xf32, {SymShapeAttr = ["4096", "s23"]}>) -> tensor<4096x?xf32, {SymShapeAttr = ["4096", "s24"]}>
    return %2 : tensor<4096x?xf32, {SymShapeAttr = ["4096", "s24"]}>
  }
}




