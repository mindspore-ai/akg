// RUN: mfusion-opt %s --decompose="pattern-type=AFTER_MANUAL_FUSION" | FileCheck %s --check-prefix=KEEP
// RUN: mfusion-opt %s --decompose="pattern-type=AFTER_MANUAL_FUSION extra-op-list=matmul_with_bias" | FileCheck %s --check-prefix=EXTRA
// AFTER empty op-list keeps matmul_with_bias; extra-op-list=matmul_with_bias splits it
// (selective op-list-only coverage remains in test_matmul_with_bias_decompose.mlir).

// KEEP-LABEL: func.func @matmul_with_bias_after
// KEEP: %[[V:.*]] = mfuse.matmul_with_bias %arg0, %arg1, %arg2
// KEEP-NEXT: return %[[V]]
// EXTRA-LABEL: func.func @matmul_with_bias_after
// EXTRA-NOT: mfuse.matmul_with_bias
// EXTRA: %[[MM:.*]] = mfuse.matmul
// EXTRA: %[[ADD:.*]] = mfuse.add %[[MM]], %arg2
// EXTRA: return %[[ADD]]
func.func @matmul_with_bias_after(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<4xf32>)
    -> tensor<2x4xf32> {
  %0 = mfuse.matmul_with_bias %arg0, %arg1, %arg2
      : (tensor<2x3xf32>, tensor<3x4xf32>, tensor<4xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}
