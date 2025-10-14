//RUN: akg-opt %s -convert-mindspore-to-linalg | FileCheck %s
//CHECK: tensor.empty()
//CHECK: linalgExt.unsorted_segment_sum

module {
  func.func @unsorted_segment_sum(%arg0: tensor<4x5x6x23xf32>, %arg1: tensor<4x5xi32>) -> tensor<20x6x23xf32> {
    %0 = "mindspore.unsorted_segment_sum"(%arg0, %arg1) {num_segments = 20 : i64}  : (tensor<4x5x6x23xf32>, tensor<4x5xi32>)  -> (tensor<20x6x23xf32>)
    return %0 : tensor<20x6x23xf32>
  }
}
