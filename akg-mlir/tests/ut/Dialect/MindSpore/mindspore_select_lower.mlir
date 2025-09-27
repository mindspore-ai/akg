//RUN: akg-opt %s -convert-mindspore-to-tosa | FileCheck %s
module {
//CHECK: tosa.select
func.func @test_select_i32(%arg0 : tensor<4xi1>, %arg1 : tensor<4xf32>, %arg2 : tensor<4xf32>) -> tensor<4xf32> {
    %0 = "mindspore.select"(%arg0, %arg1, %arg2) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
}