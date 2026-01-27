// RUN: mfusion-opt %s --muse-dvm-cluster | FileCheck %s

module {
// Test clustering with arithmetic constant operations
// NOTE: Single-element finite float constants ARE fused into the cluster
// when there are multiple operations to cluster
// CHECK-LABEL: func @test_cluster_with_scalar_constant
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-NOT: muse.fused
// CHECK: %[[CONST:.*]] = arith.constant dense<2.000000e+00> : tensor<f32>
// CHECK: %[[MUL:.*]] = muse.mul %arg0, %[[CONST]]
// CHECK-SAME: : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
// CHECK: return %[[MUL]]
func.func @test_cluster_with_scalar_constant(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = arith.constant dense<2.0> : tensor<f32>
  %1 = muse.mul %arg0, %0 : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test clustering with tensor constant
// NOTE: The broadcast and mul operations ARE clustered together
// The constant is also included in the cluster as input
// CHECK-LABEL: func @test_no_cluster_with_tensor_constant
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK: %[[CONST:.*]] = arith.constant dense<{{.*}}> : tensor<2x2xf32>
// CHECK: %[[FUSED:.*]] = muse.fused %[[CONST]], %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<2x2xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG1:.*]]: tensor<2x2xf32>, %[[ARG2:.*]]: tensor<4x4xf32>):
// CHECK: %[[BROADCAST:.*]] = muse.broadcast_to %[[ARG1]]
// CHECK-SAME: : (tensor<2x2xf32>, tensor<2xi64>) -> tensor<4x4xf32>
// CHECK: %[[MUL:.*]] = muse.mul %[[ARG2]], %[[BROADCAST]]
// CHECK: muse.yield %[[MUL]]
// CHECK: return %[[FUSED]]
func.func @test_no_cluster_with_tensor_constant(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %shape = arith.constant dense<[4, 4]> : tensor<2xi64>
  %1 = muse.broadcast_to %0, %shape : (tensor<2x2xf32>, tensor<2xi64>) -> tensor<4x4xf32>
  %2 = muse.mul %arg0, %1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// Test clustering with bool constant
// NOTE: Bool type constants are NOT fused into the cluster
// CHECK-LABEL: func @test_no_cluster_with_bool_constant
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK: %[[CONST:.*]] = arith.constant dense<true> : tensor<i1>
// CHECK: %[[GT:.*]] = muse.gt %arg0, %arg0
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
// CHECK: %[[SELECT:.*]] = muse.select %[[CONST]], %arg0, %arg1
// CHECK-SAME: : (tensor<i1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: return %[[SELECT]]
func.func @test_no_cluster_with_bool_constant(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = arith.constant dense<true> : tensor<i1>
  %1 = muse.gt %arg0, %arg0 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  %2 = muse.select %0, %arg0, %arg1 : (tensor<i1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// Test clustering with infinity constant
// NOTE: Infinity constants are NOT fused into the cluster (not finite values)
// CHECK-LABEL: func @test_no_cluster_with_infinity_constant
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK: %[[CONST:.*]] = arith.constant dense<0x7F800000> : tensor<f32>
// CHECK: %[[MUL:.*]] = muse.mul %arg0, %[[CONST]]
// CHECK-SAME: : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
// CHECK: return %[[MUL]]
func.func @test_no_cluster_with_infinity_constant(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = arith.constant dense<0x7F800000> : tensor<f32>
  %1 = muse.mul %arg0, %0 : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test clustering with multiple scalar constants
// NOTE: Multiple single-element finite float constants ARE fused into the cluster
// CHECK-LABEL: func @test_cluster_with_multiple_constants
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = muse.fused %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>):
// CHECK: %[[CONST1:.*]] = arith.constant dense<2.000000e+00> : tensor<f32>
// CHECK: %[[CONST2:.*]] = arith.constant dense<3.000000e+00> : tensor<f32>
// CHECK: %[[MUL:.*]] = muse.mul %[[ARG2]], %[[CONST1]]
// CHECK: %[[ADD:.*]] = muse.add %[[MUL]], %[[CONST2]]
// CHECK: muse.yield %[[ADD]]
// CHECK: return %[[FUSED]]
func.func @test_cluster_with_multiple_constants(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = arith.constant dense<2.0> : tensor<f32>
  %1 = arith.constant dense<3.0> : tensor<f32>
  %2 = muse.mul %arg0, %0 : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  %3 = muse.add %2, %1 : (tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %3 : tensor<4x4xf32>
}
}
