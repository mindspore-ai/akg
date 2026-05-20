// RUN: mfusion-opt %s --canonicalize | FileCheck %s
//
// cast(permute(x)) -> permute(cast(x)) so cast is adjacent to the permute's producer.

// CHECK-LABEL: func.func @cast_after_permute
// CHECK-DAG: mfuse.cast %arg0
// CHECK-DAG: mfuse.permute
func.func @cast_after_permute(%arg0: tensor<2x3x4xf16>) -> tensor<4x2x3xf32> {
  %p = mfuse.permute %arg0, [2, 0, 1] : (tensor<2x3x4xf16>) -> tensor<4x2x3xf16>
  %c = mfuse.cast %p : (tensor<4x2x3xf16>) -> tensor<4x2x3xf32>
  return %c : tensor<4x2x3xf32>
}

// Permute result is still used: keep original permute on %arg0; cast path becomes permute(cast(x)).
// Canonicalized order: permute(arg0); cast(arg0) to f32; permute(cast) for f32 branch; return.
// CHECK-LABEL: func.func @permute_result_shared
// CHECK-NEXT: %{{[0-9]+}} = mfuse.permute %arg0, [1, 0] : (tensor<2x3xf16>) -> tensor<3x2xf16>
// CHECK-NEXT: %{{[0-9]+}} = mfuse.cast %arg0 : (tensor<2x3xf16>) -> tensor<2x3xf32>
// CHECK-NEXT: %{{[0-9]+}} = mfuse.permute %{{[0-9]+}}, [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
// CHECK-NEXT: return %{{[0-9]+}}, %{{[0-9]+}} : tensor<3x2xf32>, tensor<3x2xf16>
func.func @permute_result_shared(%arg0: tensor<2x3xf16>) -> (tensor<3x2xf32>, tensor<3x2xf16>) {
  %p = mfuse.permute %arg0, [1, 0] : (tensor<2x3xf16>) -> tensor<3x2xf16>
  %c = mfuse.cast %p : (tensor<3x2xf16>) -> tensor<3x2xf32>
  return %c, %p : tensor<3x2xf32>, tensor<3x2xf16>
}
