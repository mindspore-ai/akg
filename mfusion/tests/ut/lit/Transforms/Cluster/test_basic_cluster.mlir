// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// Basic cluster test: Simple sequence of supported operations should be clustered
// CHECK-LABEL: func @test_basic_add_mul_cluster
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG2]]
// CHECK: mfuse.yield %[[MUL]]
// CHECK: return %[[FUSED]]
func.func @test_basic_add_mul_cluster(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Add operation
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // Mul operation (uses add result)
  %1 = mfuse.mul %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Multiple operation chains that merge into a single cluster
// Operations from different chains can be clustered together when they have a common consumer
// CHECK-LABEL: func @test_multiple_clusters
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %arg2
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>, %[[ARG5:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD1:.*]] = mfuse.add %[[ARG3]], %[[ARG4]]
// CHECK: %[[MUL1:.*]] = mfuse.mul %[[ADD1]], %[[ARG3]]
// CHECK: %[[ADD2:.*]] = mfuse.add %[[ARG4]], %[[ARG5]]
// CHECK: %[[MUL2:.*]] = mfuse.mul %[[ADD2]], %[[ARG4]]
// CHECK: %[[ADD3:.*]] = mfuse.add %[[MUL1]], %[[MUL2]]
// CHECK: mfuse.yield %[[ADD3]]
// CHECK: return %[[FUSED]]
func.func @test_multiple_clusters(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.mul %0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mfuse.add %arg1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %3 = mfuse.mul %2, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %4 = mfuse.add %1, %3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %4 : tensor<4x4xf32>
}

// Test scenario: Long chain of element-wise operations with different tensor sizes
// All arithmetic operations (add, mul, sub, div) can be clustered together
// CHECK-LABEL: func @test_element_wise_chain
// CHECK-SAME: %arg0: tensor<8x8xf32>
// CHECK-SAME: %arg1: tensor<8x8xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<8x8xf32>, %[[ARG3:.*]]: tensor<8x8xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG2]]
// CHECK: %[[SUB:.*]] = mfuse.sub %[[MUL]], %[[ARG3]]
// CHECK: %[[DIV:.*]] = mfuse.div %[[SUB]], %[[ARG2]]
// CHECK: mfuse.yield %[[DIV]]
// CHECK: return %[[FUSED]]
func.func @test_element_wise_chain(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = mfuse.mul %0, %arg0 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %2 = mfuse.sub %1, %arg1 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %3 = mfuse.div %2, %arg0 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %3 : tensor<8x8xf32>
}

// Test scenario: Fuse operations after processing all inputs
// Non-clusterable operation (aclnn.sub) separates the computation into two phases
// The cluster pass can still fuse operations that depend on the non-clusterable op's output
// CHECK-LABEL: func @test_fuse_op_after_all_inputs
// CHECK-SAME: %arg0: tensor<2x2xf16>
// CHECK-SAME: %arg1: tensor<2x2xf16>
// CHECK: %[[CST:.*]] = mfuse.constant dense<1>
// CHECK-SAME: : tensor<i64, {is_scalar = ""}>
// CHECK: %[[SUB:.*]] = mfuse.aclnn.sub %arg0, %arg1, %[[CST]]
// CHECK-SAME: : (tensor<2x2xf16>, tensor<2x2xf16>, tensor<i64, {is_scalar = ""}>) -> tensor<2x2xf16>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %[[SUB]]
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<2x2xf16>, tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x2xf16>, %[[ARG3:.*]]: tensor<2x2xf16>, %[[ARG4:.*]]: tensor<2x2xf16>):
// CHECK: %[[MUL1:.*]] = mfuse.mul %[[ARG2]], %[[ARG3]]
// CHECK-SAME: : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
// CHECK: %[[MUL2:.*]] = mfuse.mul %[[MUL1]], %[[MUL1]]
// CHECK-SAME: : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
// CHECK: %[[MUL3:.*]] = mfuse.mul %[[ARG4]], %[[ARG4]]
// CHECK-SAME: : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
// CHECK: %[[MUL4:.*]] = mfuse.mul %[[MUL2]], %[[MUL3]]
// CHECK-SAME: : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
// CHECK: mfuse.yield %[[MUL4]]
// CHECK-SAME: : tensor<2x2xf16>
// CHECK: return %[[FUSED]]
// CHECK-SAME: : tensor<2x2xf16>
func.func @test_fuse_op_after_all_inputs(%arg0: tensor<2x2xf16>, %arg1: tensor<2x2xf16>) -> tensor<2x2xf16> {
  %cst = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
  // Non-clusterable operation: aclnn.sub is not in DVM clusterable ops list
  %0 = mfuse.aclnn.sub %arg0, %arg1, %cst : (tensor<2x2xf16>, tensor<2x2xf16>, tensor<i64, {is_scalar = ""}>) -> tensor<2x2xf16>
  // First chain: mul -> mul (clusterable, uses %arg0 and %arg1)
  %1 = mfuse.mul %arg0, %arg1 : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
  %2 = mfuse.mul %1, %1 : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
  // Second chain: mul (clusterable, uses %0 from aclnn.sub)
  %3 = mfuse.mul %0, %0 : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
  // Final mul: merges both chains (clusterable)
  %4 = mfuse.mul %2, %3 : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
  return %4 : tensor<2x2xf16>
}

// Test scenario: Intermediate value used by both non-clusterable and clusterable ops
// When a value produced by clusterable op is consumed by both a non-clusterable op
// and a clusterable op, the cluster creates multiple outputs to handle both cases
// CHECK-LABEL: func @test_cluster_with_external_output
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]]:2 = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>)
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ADD]]
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: mfuse.yield %[[ADD]], %[[MUL]]
// CHECK-SAME: : tensor<4x4xf32>, tensor<4x4xf32>
// CHECK: %[[TANH:.*]] = mfuse.aclnn.tanh %[[FUSED]]#0
// CHECK-SAME: : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: return %[[TANH]], %[[FUSED]]#1
// CHECK-SAME: : tensor<4x4xf32>, tensor<4x4xf32>
func.func @test_cluster_with_external_output(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  // Add operation (clusterable)
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // Tanh operation (non-clusterable, uses %0 as external output)
  %1 = mfuse.aclnn.tanh %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  // Mul operation (clusterable, uses %0 for internal data flow)
  %2 = mfuse.mul %0, %0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // Return both results
  return %1, %2 : tensor<4x4xf32>, tensor<4x4xf32>
}

// Test scenario: Insert point breaks dominance for non-cluster users
// When the insert point is moved forward due to external inputs,
// it may end up after a non-cluster user of an external output.
// In this case, the cluster cannot be created because the fused op
// would not dominate the non-cluster user.
// CHECK-LABEL: func @test_insert_point_breaks_dominance
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[CST:.*]] = mfuse.constant dense<1>
// CHECK: %[[ADD1:.*]] = mfuse.add %arg0, %arg1
// CHECK: %[[TANH:.*]] = mfuse.aclnn.tanh %[[ADD1]]
// CHECK: %[[SUB:.*]] = mfuse.aclnn.sub {{%arg2}}, {{%arg2}}, %[[CST]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[SUB]], %[[ADD1]]
// CHECK: %[[ADD2:.*]] = mfuse.add %[[MUL]], %arg0
// CHECK: return %[[TANH]], %[[ADD2]]
func.func @test_insert_point_breaks_dominance(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %cst = mfuse.constant dense<1> : tensor<i64, {is_scalar = ""}>
  // op1: add (clusterable), insert point starts here
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // non-cluster user of %0, appears before the external input provider
  %1 = mfuse.aclnn.tanh %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  // non-cluster op that provides external input to later cluster op
  %2 = mfuse.aclnn.sub %arg2, %arg2, %cst : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<4x4xf32>
  // op2: mul (clusterable), uses %2 as external input
  // This moves insert point to after aclnn.sub
  %3 = mfuse.mul %2, %0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // op3: add (clusterable)
  %4 = mfuse.add %3, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  // %0 is external output, %1 (aclnn.tanh) is its non-cluster user
  // insert point is after aclnn.sub, but aclnn.tanh is before aclnn.sub
  // So insert point is NOT before the non-cluster user, cluster cannot be created
  return %1, %4 : tensor<4x4xf32>, tensor<4x4xf32>
}
}