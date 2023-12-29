//RUN: akg-opt %s -convert-mindspore-to-linalg | FileCheck %s

module {
  //CHECK: linalg.generic
  //CHECK: arith.mulf
  //CHECK: arith.constant
  //CHECK: arith.addf
  //CHECK: math.sqrt
  //CHECK: arith.addf
  //CHECK: math.log
  //CHECK: linalg.yield
  func.func @Asinh_Test(%arg0: tensor<1x1x1x12xf32>) -> tensor<1x1x1x12xf32> attributes {enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
    %1 = "mindspore.asinh"(%arg0) {} : (tensor<1x1x1x12xf32>) -> tensor<1x1x1x12xf32>
    return %1 : tensor<1x1x1x12xf32>
  }

  //CHECK: linalg.generic
  //CHECK: arith.mulf
  //CHECK: arith.constant
  //CHECK: arith.subf
  //CHECK: math.sqrt
  //CHECK: arith.addf
  //CHECK: math.log
  //CHECK: linalg.yield
  func.func @Acosh_Test(%arg0: tensor<1x1x1x12xf32>) -> tensor<1x1x1x12xf32> attributes {enable_atomic_add = false, mindspore_kernel, process = "cpu"} {
    %1 = "mindspore.acosh"(%arg0) {} : (tensor<1x1x1x12xf32>) -> tensor<1x1x1x12xf32>
    return %1 : tensor<1x1x1x12xf32>
  }
}