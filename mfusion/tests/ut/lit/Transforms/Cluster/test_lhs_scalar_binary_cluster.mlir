// RUN: mfusion-opt %s --mfuse-canonicalize-binary-scalar-operands --mfuse-dvm-cluster | FileCheck %s

module {
  // CHECK-LABEL: func @cluster_lhs_scalar_sub_after_recovering_num_to_tensor
  // CHECK-NOT: mfuse.num_to_tensor
  // CHECK-NOT: mfuse.full
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0 {fusion_type = "dvm"}
  // CHECK: ^bb0(%[[X:.*]]: tensor<4x4xf32>):
  // CHECK: %[[C:.*]] = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
  // CHECK: %[[MUL:.*]] = mfuse.mul %[[X]], %[[X]]
  // CHECK: %[[SUB:.*]] = mfuse.sub %[[C]], %[[MUL]]
  // CHECK: mfuse.yield %[[SUB]]
  // CHECK: return %[[FUSED]]
  func.func @cluster_lhs_scalar_sub_after_recovering_num_to_tensor(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c1 = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
    %n = mfuse.num_to_tensor %c1 : (tensor<i64, {is_scalar = ""}>) -> tensor<i64>
    %mul = mfuse.mul %arg0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %sub = mfuse.sub %n, %mul : (tensor<i64>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %sub : tensor<4x4xf32>
  }
}
