// RUN: mfusion-opt %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @binary_scalar_valid
  func.func @binary_scalar_valid(%arg0: tensor<2xf32>, %arg1: tensor<2xf16>, %arg2: tensor<2xbf16>, %arg3: tensor<2xi32>) {
    // CHECK: dvm.binary_scalar Pow %arg0, 2 : tensor<2xf32>, i32 -> tensor<2xf32>
    %0 = dvm.binary_scalar Pow %arg0, 2 : tensor<2xf32>, i32 -> tensor<2xf32>
    // CHECK: dvm.binary_scalar Sub 0.000000e+00, %arg0 : f32, tensor<2xf32> -> tensor<2xf32>
    %1 = dvm.binary_scalar Sub 0.0, %arg0 : f32, tensor<2xf32> -> tensor<2xf32>
    // CHECK: dvm.binary_scalar Mul %arg1, 2.500000e+00 : tensor<2xf16>, f16 -> tensor<2xf16>
    %2 = dvm.binary_scalar Mul %arg1, 2.5 : tensor<2xf16>, f16 -> tensor<2xf16>
    // CHECK: dvm.binary_scalar Add %arg2, 1.250000e+00 : tensor<2xbf16>, bf16 -> tensor<2xbf16>
    %3 = dvm.binary_scalar Add %arg2, 1.25 : tensor<2xbf16>, bf16 -> tensor<2xbf16>
    // CHECK: dvm.binary_scalar Maximum %arg3, 0 : tensor<2xi32>, i32 -> tensor<2xi32>
    %4 = dvm.binary_scalar Maximum %arg3, 0 : tensor<2xi32>, i32 -> tensor<2xi32>
    return
  }
}
