// RUN: akg-opt %s --convert-linalg-to-pypto | FileCheck %s --implicit-check-not=linalg.

// CHECK-LABEL: func.func @matmul(
// CHECK: %[[R:.*]] = pypto.matmul %arg0, %arg1 init %arg2
// CHECK: return %[[R]]
func.func @matmul(%a: tensor<8x16xf16>, %b: tensor<16x12xf16>,
                  %init: tensor<8x12xf16>) -> tensor<8x12xf16> {
  %r = linalg.matmul ins(%a, %b : tensor<8x16xf16>, tensor<16x12xf16>)
      outs(%init : tensor<8x12xf16>) -> tensor<8x12xf16>
  return %r : tensor<8x12xf16>
}
