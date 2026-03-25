// RUN: mfusion-opt %s -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @test_mfuse_constant() -> tensor<f32, {is_scalar = ""}> {
  // CHECK-LABEL: func @test_mfuse_constant
  %0 = mfuse.constant dense<42.0> : tensor<f32, {is_scalar = ""}>
  return %0 : tensor<f32, {is_scalar = ""}>
}

func.func @test_mfuse_constant_int() -> tensor<i32, {is_scalar = ""}> {
  // CHECK-LABEL: func @test_mfuse_constant_int
  %0 = mfuse.constant dense<100> : tensor<i32, {is_scalar = ""}>
  return %0 : tensor<i32, {is_scalar = ""}>
}

func.func @test_mfuse_constant_tensor() -> tensor<2x2xf32> {
  // CHECK-LABEL: func @test_mfuse_constant_tensor
  %0 = mfuse.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}