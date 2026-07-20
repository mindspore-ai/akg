// Guard ND K=1 + bias against BiasAdd-before-K1 ordering bugs.
//
// FuseMatmulK1ToMul only matches MatmulOp. If BiasAdd runs first, K=1
// matmul+bias becomes MatmulWithBiasOp and K1 misses it; AFTER_MANUAL_FUSION
// later splits back to matmul+add with no mul.
//
// Manager / mfuse-fusion order is K1 then BiasAdd, so mul is produced first
// and bias remains a separate add.

// Correct Manager order: K1 then BiasAdd (+ optional AFTER_MANUAL_FUSION).
// RUN: mfusion-opt %s --matmul-optimization | FileCheck %s --check-prefix=OPT
// RUN: mfusion-opt %s --mfuse-fusion | FileCheck %s --check-prefix=OPT
// RUN: mfusion-opt %s --matmul-optimization --decompose="pattern-type=AFTER_MANUAL_FUSION op-list=matmul_with_bias" | FileCheck %s --check-prefix=OPT

// NEGATIVE / anti-pattern: BiasAdd then K1 (and after AFTER_MANUAL_FUSION).
// These RUNs document the broken ordering; do not "fix" expectations to make them green
// by changing Manager order. Keep as regression locks against reintroducing BiasAdd-before-K1.
// RUN: mfusion-opt %s --fuse-matmul-bias-add --fuse-matmul-k1-to-mul | FileCheck %s --check-prefix=BADORDER
// RUN: mfusion-opt %s --fuse-matmul-bias-add --fuse-matmul-k1-to-mul --decompose="pattern-type=AFTER_MANUAL_FUSION op-list=matmul_with_bias" | FileCheck %s --check-prefix=BADPIPE

module {
  // Baseline: ND K=1 without bias -> mul under both paths that reach K1.
  // OPT-LABEL: func @nd_k1_no_bias
  // OPT-NOT: mfuse.matmul
  // OPT: mfuse.mul
  // BADORDER-LABEL: func @nd_k1_no_bias
  // BADORDER-NOT: mfuse.matmul
  // BADORDER: mfuse.mul
  // BADPIPE-LABEL: func @nd_k1_no_bias
  // BADPIPE-NOT: mfuse.matmul
  // BADPIPE: mfuse.mul
  func.func @nd_k1_no_bias(%arg0: tensor<2x3x1xf32>, %arg1: tensor<2x1x4xf32>)
      -> tensor<2x3x4xf32> {
    %0 = mfuse.matmul %arg0, %arg1 {trans_x1 = false, trans_x2 = false}
        : (tensor<2x3x1xf32>, tensor<2x1x4xf32>) -> tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }

  // With bias: recommended order yields mul+add; wrong order yields matmul_with_bias
  // (or matmul+add after AFTER_MANUAL_FUSION) and never mul.
  // OPT-LABEL: func @nd_k1_with_bias
  // OPT-NOT: mfuse.matmul_with_bias
  // OPT-NOT: mfuse.matmul{{[^_]}}
  // OPT: mfuse.mul
  // OPT: mfuse.add
  // BADORDER-LABEL: func @nd_k1_with_bias
  // BADORDER-NOT: mfuse.mul
  // BADORDER: mfuse.matmul_with_bias
  // BADPIPE-LABEL: func @nd_k1_with_bias
  // BADPIPE-NOT: mfuse.mul
  // BADPIPE-NOT: mfuse.matmul_with_bias
  // BADPIPE: mfuse.matmul
  // BADPIPE: mfuse.add
  func.func @nd_k1_with_bias(%arg0: tensor<2x3x1xf32>, %arg1: tensor<2x1x4xf32>,
                               %bias: tensor<4xf32>) -> tensor<2x3x4xf32> {
    %0 = mfuse.matmul %arg0, %arg1 {trans_x1 = false, trans_x2 = false}
        : (tensor<2x3x1xf32>, tensor<2x1x4xf32>) -> tensor<2x3x4xf32>
    %1 = mfuse.add %0, %bias : (tensor<2x3x4xf32>, tensor<4xf32>) -> tensor<2x3x4xf32>
    return %1 : tensor<2x3x4xf32>
  }
}
