//RUN: akg-opt %s -split-input-file -convert-mindspore-to-linalg | FileCheck %s
//CHECK: tensor.empty()
//CHECK：linalg.fill
//CHECK: linalg.matmul

module {
  func.func @matmul(%arg0: tensor<32x128xf32>, %arg1: tensor<128x64xf32>) -> tensor<32x64xf32> {
    %0 = "mindspore.matmul"(%arg0, %arg1)  : (tensor<32x128xf32>, tensor<128x64xf32>)  -> (tensor<32x64xf32>)
    return %0 : tensor<32x64xf32>
  }
}

// -----

//CHECK: tensor.empty()
//CHECK：linalg.fill
//CHECK: linalg.matmul_transpose_b

module {
  func.func @matmul(%arg0: tensor<32x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<32x64xf32> {
    %0 = "mindspore.matmul"(%arg0, %arg1) {transpose_b = true} : (tensor<32x128xf32>, tensor<64x128xf32>)  -> (tensor<32x64xf32>)
    return %0 : tensor<32x64xf32>
  }
}