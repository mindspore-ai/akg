// RUN: mfusion-opt %s --mfuse-akg-cluster | FileCheck %s

// CHECK-LABEL: func.func @test_elementwise_chain(
// CHECK:         %[[FUSED:.*]] = mfuse.fused
// CHECK-SAME:      fusion_type = "akg"
// CHECK-NEXT:    ^bb0(
// CHECK-NEXT:      mfuse.add
// CHECK-NEXT:      mfuse.mul
// CHECK-NEXT:      mfuse.exp
// CHECK-NEXT:      mfuse.yield
func.func @test_elementwise_chain(%arg0: tensor<16x128xf32>, %arg1: tensor<16x128xf32>) -> tensor<16x128xf32> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<16x128xf32>, tensor<16x128xf32>) -> tensor<16x128xf32>
  %1 = mfuse.mul %0, %arg0 : (tensor<16x128xf32>, tensor<16x128xf32>) -> tensor<16x128xf32>
  %2 = mfuse.exp %1 : (tensor<16x128xf32>) -> tensor<16x128xf32>
  return %2 : tensor<16x128xf32>
}

// CHECK-LABEL: func.func @test_reduce(
// CHECK-NOT:     mfuse.fused
// CHECK:         mfuse.reduce_sum
func.func @test_reduce(%arg0: tensor<16x128xf32>) -> tensor<16x1xf32> {
  %0 = mfuse.reduce_sum %arg0 {dimensions = [1], keepdim = true} : (tensor<16x128xf32>) -> tensor<16x1xf32>
  return %0 : tensor<16x1xf32>
}

// CHECK-LABEL: func.func @test_unsupported_dynamic_shape(
// CHECK-NOT:     mfuse.fused
// CHECK:         mfuse.add
// CHECK:         mfuse.mul
func.func @test_unsupported_dynamic_shape(%arg0: tensor<?x128xf32>, %arg1: tensor<?x128xf32>) -> tensor<?x128xf32> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<?x128xf32>, tensor<?x128xf32>) -> tensor<?x128xf32>
  %1 = mfuse.mul %0, %arg0 : (tensor<?x128xf32>, tensor<?x128xf32>) -> tensor<?x128xf32>
  return %1 : tensor<?x128xf32>
}

// CHECK-LABEL: func.func @test_unsupported_complex_type(
// CHECK-NOT:     mfuse.fused
// CHECK:         mfuse.add
func.func @test_unsupported_complex_type(%arg0: tensor<16x128xcomplex<f32>>, %arg1: tensor<16x128xcomplex<f32>>) -> tensor<16x128xcomplex<f32>> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<16x128xcomplex<f32>>, tensor<16x128xcomplex<f32>>) -> tensor<16x128xcomplex<f32>>
  return %0 : tensor<16x128xcomplex<f32>>
}

// CHECK-LABEL: func.func @test_unsupported_op(
// CHECK-NOT:     mfuse.fused
// CHECK:         mfuse.matmul
func.func @test_unsupported_op(%arg0: tensor<16x128xf32>, %arg1: tensor<128x256xf32>) -> tensor<16x256xf32> {
  // MatMul should not be clustered by AKG (it is handled by CANN)
  %0 = mfuse.matmul %arg0, %arg1 {trans_x1 = false, trans_x2 = false} : (tensor<16x128xf32>, tensor<128x256xf32>) -> tensor<16x256xf32>
  return %0 : tensor<16x256xf32>
}
