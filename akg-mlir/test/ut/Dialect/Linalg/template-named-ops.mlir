// RUN: akg-opt %s -split-input-file --linalg-template-named-ops="template-path=%S/../../../../compiler/lib/Dialect/Linalg/Transforms/TemplatedOpImpl/" | FileCheck %s

func.func @template_matmul_buffer(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>) {
  %res = linalg.matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                       outs(%C: tensor<16x32xf32>)-> tensor<16x32xf32>
  return
}


// CHECK: #[[A_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[B_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[C_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK: func.func @template_matmul

// CHECK: func @template_matmul_buffer
// CHECK-SAME: %[[A:.+]]: tensor<16x8xf32>
// CHECK-SAME: %[[B:.+]]: tensor<8x32xf32>
// CHECK-SAME: %[[C:.+]]: tensor<16x32xf32>

// CHECK: linalg.template
// CHECK-SAME: indexing_maps = [#[[A_MAP]], #[[B_MAP]], #[[C_MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A]], %[[B]]
// CHECK-SAME: outs(%[[C]]

// CHECK: ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK:   %[[MUL:.+]] = arith.mulf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK:   %[[ADD:.+]] = arith.addf %[[C_ARG]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[ADD]] : f32

// -----

func.func @template_matmul_transpose_b(%A : tensor<16x8xf32>, %B: tensor<32x8xf32>, %C: tensor<16x32xf32>) {
  linalg.matmul_transpose_b ins(%A, %B: tensor<16x8xf32>, tensor<32x8xf32>) outs(%C: tensor<16x32xf32>)-> tensor<16x32xf32>
  return
}

// CHECK: #[[G_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: #[[H_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[I_MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>

// CHECK: func.func @template_matmul_transpose_b

// CHECK: func @template_matmul_transpose_b
// CHECK-SAME: %[[A:.+]]: tensor<16x8xf32>
// CHECK-SAME: %[[B:.+]]: tensor<32x8xf32>
// CHECK-SAME: %[[C:.+]]: tensor<16x32xf32>
 
// CHECK: linalg.template
// CHECK-SAME: indexing_maps = [#[[H_MAP]], #[[I_MAP]], #[[G_MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A]], %[[B]]
// CHECK-SAME: outs(%[[C]]

// CHECK: ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK:   %[[MUL:.+]] = arith.mulf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK:   %[[ADD:.+]] = arith.addf %[[C_ARG]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[ADD]] : f32

// -----

func.func @template_matmul_transpose_b(%A : tensor<16x8xf16>, %B: tensor<32x8xf16>, %C: tensor<16x32xf16>) {
  linalg.matmul_transpose_b ins(%A, %B: tensor<16x8xf16>, tensor<32x8xf16>) outs(%C: tensor<16x32xf16>)-> tensor<16x32xf16>
  return
}

// CHECK: #[[G_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: #[[H_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[I_MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>

// CHECK: func.func @template_matmul_transpose_b
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf16>, %[[B:.+]]: tensor<?x?xf16>, %[[C:.+]]: tensor<?x?xf16>

// CHECK: func @template_matmul_transpose_b

// CHECK: linalg.template
// CHECK-SAME: indexing_maps = [#[[H_MAP]], #[[I_MAP]], #[[G_MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A]], %[[B]]
// CHECK-SAME: outs(%[[C]]

// CHECK: ^{{.*}}(%[[A_ARG:.+]]: f16, %[[B_ARG:.+]]: f16, %[[C_ARG:.+]]: f16)
// CHECK:   %[[MUL:.+]] = arith.mulf %[[A_ARG]], %[[B_ARG]] : f16
// CHECK:   %[[ADD:.+]] = arith.addf %[[C_ARG]], %[[MUL]] : f16
// CHECK:   linalg.yield %[[ADD]] : f16

// -----

func.func @template_batch_matmul_4d(%A : tensor<32x12x128x128xf16>, %B: tensor<32x12x128x64xf16>, %C: tensor<32x12x128x64xf16>) {
  linalg.batch_matmul_4d ins(%A, %B: tensor<32x12x128x128xf16>, tensor<32x12x128x64xf16>) outs(%C: tensor<32x12x128x64xf16>)-> tensor<32x12x128x64xf16>
  return
}

// CHECK: #[[G_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
// CHECK: #[[H_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK: #[[I_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>

// CHECK: func.func @template_batch_matmul_4d
// CHECK-SAME: %[[A:.+]]: tensor<?x?x?x?xf16>, %[[B:.+]]: tensor<?x?x?x?xf16>, %[[C:.+]]: tensor<?x?x?x?xf16>

// CHECK: func @template_batch_matmul_4d

// CHECK: linalg.template
// CHECK-SAME: indexing_maps = [#[[I_MAP]], #[[G_MAP]], #[[H_MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A]], %[[B]]
// CHECK-SAME: outs(%[[C]]

// -----

func.func @template_batch_matmul_4d_transpose_b(%A : tensor<32x12x128x64xf16>, %B: tensor<32x12x128x64xf16>, %C: tensor<32x12x128x128xf16>) {
  linalg.batch_matmul_4d_transpose_b ins(%A, %B: tensor<32x12x128x64xf16>, tensor<32x12x128x64xf16>) outs(%C: tensor<32x12x128x128xf16>)-> tensor<32x12x128x128xf16>
  return
}

// CHECK: #[[H_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK: #[[I_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK: #[[J_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>

// CHECK: func.func @template_batch_matmul_4d_transpose_b
// CHECK-SAME: %[[A:.+]]: tensor<?x?x?x?xf16>, %[[B:.+]]: tensor<?x?x?x?xf16>, %[[C:.+]]: tensor<?x?x?x?xf16>

// CHECK: func @template_batch_matmul_4d_transpose_b

// CHECK: linalg.template
// CHECK-SAME: indexing_maps = [#[[I_MAP]], #[[J_MAP]], #[[H_MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:  ins(%[[A]], %[[B]]
// CHECK-SAME: outs(%[[C]]
