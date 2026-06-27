// RUN: mfusion-opt %s --pass-pipeline="builtin.module(func.func(mfuse-decompose-matmul-with-bias-for-dvm-cluster,mfuse-dvm-cluster),canonicalize)" | FileCheck %s

module {
  // CHECK-LABEL: func.func @f32_matmul_stays_outside_add_suffix_fuses
  // CHECK-SAME: %arg0: tensor<4096x16xf32>
  // CHECK-SAME: %arg1: tensor<16x16xf32>
  // CHECK-SAME: %arg2: tensor<16xf32>
  // CHECK-SAME: %arg3: tensor<4096x16xf32>
  // CHECK-NOT: mfuse.matmul_with_bias
  // CHECK: %[[MM:.*]] = mfuse.matmul %arg0, %arg1
  // CHECK-SAME: : (tensor<4096x16xf32>, tensor<16x16xf32>) -> tensor<4096x16xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[MM]], %arg2, %arg3
  // CHECK-SAME: {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<4096x16xf32>, tensor<16xf32>, tensor<4096x16xf32>) -> tensor<4096x16xf32>
  // CHECK: ^bb0(%[[INNER_MM:.*]]: tensor<4096x16xf32>, %[[INNER_BIAS:.*]]: tensor<16xf32>, %[[INNER_LIMIT:.*]]: tensor<4096x16xf32>):
  // CHECK: %[[ADD:.*]] = mfuse.add %[[INNER_MM]], %[[INNER_BIAS]]
  // CHECK-SAME: : (tensor<4096x16xf32>, tensor<16xf32>) -> tensor<4096x16xf32>
  // CHECK: %[[MAX:.*]] = mfuse.maximum %[[ADD]], %[[INNER_LIMIT]]
  // CHECK: %[[SUB:.*]] = mfuse.sub %[[MAX]], %[[INNER_LIMIT]]
  // CHECK: mfuse.yield %[[SUB]]
  // CHECK: return %[[FUSED]]
  func.func @f32_matmul_stays_outside_add_suffix_fuses(
      %arg0: tensor<4096x16xf32>,
      %arg1: tensor<16x16xf32>,
      %bias: tensor<16xf32>,
      %limit: tensor<4096x16xf32>) -> tensor<4096x16xf32> {
    %0 = mfuse.matmul_with_bias %arg0, %arg1, %bias
        : (tensor<4096x16xf32>, tensor<16x16xf32>, tensor<16xf32>) -> tensor<4096x16xf32>
    %1 = mfuse.maximum %0, %limit
        : (tensor<4096x16xf32>, tensor<4096x16xf32>) -> tensor<4096x16xf32>
    %2 = mfuse.sub %1, %limit
        : (tensor<4096x16xf32>, tensor<4096x16xf32>) -> tensor<4096x16xf32>
    return %2 : tensor<4096x16xf32>
  }
}
