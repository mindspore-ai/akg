// RUN: mfusion-opt %s --mfuse-fusion | FileCheck %s

module {
  // CHECK-LABEL: func.func @fusion_pipeline_recovers_lhs_scalar_before_fuse_num_to_tensor
  // CHECK: %[[C:.*]] = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
  // CHECK-NOT: mfuse.num_to_tensor
  // CHECK-NOT: mfuse.full
  // CHECK: %[[R:.*]] = mfuse.sub %[[C]], %arg0 : (tensor<i64, {is_scalar = ""}>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK: return %[[R]] : tensor<4x4xf32>
  func.func @fusion_pipeline_recovers_lhs_scalar_before_fuse_num_to_tensor(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c1 = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %n = mfuse.num_to_tensor %c1 : (tensor<i64, {is_scalar = ""}>) -> tensor<i64>
    %0 = mfuse.sub %n, %arg0 : (tensor<i64>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
