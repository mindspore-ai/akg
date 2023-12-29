//RUN: akg-opt %s -split-input-file -convert-mindspore-to-linalg | FileCheck %s
//CHECK: tensor.empty()
//CHECK：linalg.fill
//CHECK: linalg.batch_matmul

module {
  func.func @batchmatmul(%arg0: tensor<3x32x128xf32>, %arg1: tensor<3x128x64xf32>) -> tensor<3x32x64xf32> {
    %0 = "mindspore.batchmatmul"(%arg0, %arg1) : (tensor<3x32x128xf32>, tensor<3x128x64xf32>)  -> (tensor<3x32x64xf32>)
    return %0 : tensor<3x32x64xf32>
  }
}

// -----

//CHECK: tensor.empty()
//CHECK：linalg.fill
//CHECK: linalg.batch_matmul_transpose_b

module {
  func.func @batchmatmul(%arg0: tensor<3x32x128xf32>, %arg1: tensor<3x64x128xf32>) -> tensor<3x32x64xf32> {
    %0 = "mindspore.batchmatmul"(%arg0, %arg1) {transpose_b = true} : (tensor<3x32x128xf32>, tensor<3x64x128xf32>)  -> (tensor<3x32x64xf32>)
    return %0 : tensor<3x32x64xf32>
  }
}

// -----

//CHECK: tensor.empty()
//CHECK：linalg.fill
//CHECK: linalg.batch_matmul_4d

module {
  func.func @batchmatmul(%arg0: tensor<3x4x32x128xf32>, %arg1: tensor<3x4x128x64xf32>) -> tensor<3x4x32x64xf32> {
    %0 = "mindspore.batchmatmul"(%arg0, %arg1)  : (tensor<3x4x32x128xf32>, tensor<3x4x128x64xf32>)  -> (tensor<3x4x32x64xf32>)
    return %0 : tensor<3x4x32x64xf32>
  }
}

// -----

//CHECK: tensor.empty()
//CHECK：linalg.fill
//CHECK: linalg.batch_matmul_4d_transpose_b

module {
  func.func @batchmatmul(%arg0: tensor<3x4x32x128xf32>, %arg1: tensor<3x4x64x128xf32>) -> tensor<3x4x32x64xf32> {
    %0 = "mindspore.batchmatmul"(%arg0, %arg1) {transpose_b = true} : (tensor<3x4x32x128xf32>, tensor<3x4x64x128xf32>)  -> (tensor<3x4x32x64xf32>)
    return %0 : tensor<3x4x32x64xf32>
  }
}