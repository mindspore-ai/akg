// RUN: akg-opt %s -split-input-file --linalg-template-named-ops="template-path=%S/../../../../compiler/lib/Dialect/Linalg/Transforms/TemplatedOpImpl/" --linalg-generalize-named-ops -linalg-fuse-template-ops="opt-reshape-by-expand=true" | FileCheck %s --check-prefix=CHECK-EXPAND
// RUN: akg-opt %s -split-input-file --linalg-template-named-ops="template-path=%S/../../../../compiler/lib/Dialect/Linalg/Transforms/TemplatedOpImpl/" --linalg-generalize-named-ops -linalg-fuse-template-ops="opt-reshape-by-collapse=true" | FileCheck %s --check-prefix=CHECK-COLLAPSE

// -----

// CHECK-EXPAND: opt_reshape_by_expand
// CHECK-EXPAND-NOT: linalg.generic
// CHECK-EXPAND: linalg.template

module {
  func.func @opt_reshape_by_expand(%A : tensor<16x8xf16>, %B: tensor<1x1x8x32xf16>, %C: tensor<1x1x16x32xf16>)->tensor<1x1x16x32xf16> {
    %A1 = tensor.empty() : tensor<16x8xf16>
    %A2 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
                                 ins(%A: tensor<16x8xf16>) outs(%A1: tensor<16x8xf16>) -> tensor<16x8xf16>
    %A3 = tensor.expand_shape %A2 [[0, 1, 2], [3]] : tensor<16x8xf16> into tensor<1x1x16x8xf16>
    %res = linalg.batch_matmul_4d ins(%A3, %B: tensor<1x1x16x8xf16>, tensor<1x1x8x32xf16>)
                         outs(%C: tensor<1x1x16x32xf16>)-> tensor<1x1x16x32xf16>
    return %res: tensor<1x1x16x32xf16>
  }
}

// -----

module {
  func.func @opt_reshape_by_collapse(%A : tensor<16x8xf16>, %B: tensor<32x8xf16>, %C: tensor<16x32xf16>)->tensor<1x1x16x32xf16> {
    %C1 = linalg.matmul_transpose_b ins(%A, %B: tensor<16x8xf16>, tensor<32x8xf16>)
                         outs(%C: tensor<16x32xf16>)-> tensor<16x32xf16>
    %C2 = tensor.expand_shape %C1 [[0, 1, 2], [3]] : tensor<16x32xf16> into tensor<1x1x16x32xf16>
    %C3 = tensor.empty() : tensor<1x1x16x32xf16>
    %C4 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
                                     ins(%C2: tensor<1x1x16x32xf16>) outs(%C3: tensor<1x1x16x32xf16>) -> tensor<1x1x16x32xf16>
    return %C4: tensor<1x1x16x32xf16>
  }
}

// CHECK-COLLAPSE: opt_reshape_by_collapse
// CHECK-COLLAPSE: linalg.template
// CHECK-COLLAPSE-NOT: linalg.generic
// CHECK-COLLAPSE: tensor.expand_shape

// -----

module {
  func.func @opt_reshape_by_collapse(%A : tensor<16x8x1xf16>, %B: tensor<8x32xf16>, %C: tensor<16x32xf16>)->tensor<16x32xf16> {
    %A1 = tensor.empty() : tensor<16x8x1xf16>
    %A2 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
                                 ins(%A: tensor<16x8x1xf16>) outs(%A1: tensor<16x8x1xf16>) -> tensor<16x8x1xf16>
    %A3 = tensor.collapse_shape %A2 [[0], [1, 2]] : tensor<16x8x1xf16> into tensor<16x8xf16>
    %res = linalg.matmul ins(%A3, %B: tensor<16x8xf16>, tensor<8x32xf16>)
                         outs(%C: tensor<16x32xf16>)-> tensor<16x32xf16>
    return %res: tensor<16x32xf16>
  }
}

// CHECK-COLLAPSE: opt_reshape_by_collapse
// CHECK-COLLAPSE-NOT: linalg.generic
// CHECK-COLLAPSE: linalg.template
