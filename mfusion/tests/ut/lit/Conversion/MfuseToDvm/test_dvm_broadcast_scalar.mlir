// RUN: mfusion-opt %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @broadcast_scalar_f32
  // CHECK: %[[FULL:.*]] = dvm.broadcast_scalar 1.500000e+00 shape [4, 2] type Float32 : f32 -> tensor<4x2xf32>
  // CHECK: return %[[FULL]] : tensor<4x2xf32>
  func.func @broadcast_scalar_f32() -> tensor<4x2xf32> {
    %0 = dvm.broadcast_scalar 1.5 shape [4, 2] type Float32 : f32 -> tensor<4x2xf32>
    return %0 : tensor<4x2xf32>
  }

  // CHECK-LABEL: func.func @broadcast_scalar_bool
  // CHECK: %[[FULL:.*]] = dvm.broadcast_scalar 1 shape [2] type Bool : i32 -> tensor<2xi1>
  // CHECK: return %[[FULL]] : tensor<2xi1>
  func.func @broadcast_scalar_bool() -> tensor<2xi1> {
    %0 = dvm.broadcast_scalar 1 shape [2] type Bool : i32 -> tensor<2xi1>
    return %0 : tensor<2xi1>
  }

  // CHECK-LABEL: func.func @broadcast_scalar_bool_from_f32
  // CHECK: %[[FULL:.*]] = dvm.broadcast_scalar 5.000000e-01 shape [2] type Bool : f32 -> tensor<2xi1>
  // CHECK: return %[[FULL]] : tensor<2xi1>
  func.func @broadcast_scalar_bool_from_f32() -> tensor<2xi1> {
    %0 = dvm.broadcast_scalar 0.5 shape [2] type Bool : f32 -> tensor<2xi1>
    return %0 : tensor<2xi1>
  }

  // CHECK-LABEL: func.func @broadcast_scalar_int64
  // CHECK: %[[FULL:.*]] = dvm.broadcast_scalar 7 shape [4] type Int64 : i32 -> tensor<4xi64>
  // CHECK: return %[[FULL]] : tensor<4xi64>
  func.func @broadcast_scalar_int64() -> tensor<4xi64> {
    %0 = dvm.broadcast_scalar 7 shape [4] type Int64 : i32 -> tensor<4xi64>
    return %0 : tensor<4xi64>
  }
}
