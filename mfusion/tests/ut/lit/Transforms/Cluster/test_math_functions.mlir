// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
// Test math function clustering: exp, log, sqrt, rsqrt, pow, reciprocal

// Test scenario: Exponential and logarithm operations in sequence
// CHECK-LABEL: func @test_exp_log_chain
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[EXP:.*]] = mfuse.exp %[[ADD]]
// CHECK: %[[LOG:.*]] = mfuse.log %[[EXP]]
// CHECK: mfuse.yield %[[LOG]]
// CHECK: return %[[FUSED]]
func.func @test_exp_log_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.exp %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mfuse.log %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// Test scenario: Square root and reciprocal square root operations in sequence
// CHECK-LABEL: func @test_sqrt_rsqrt_chain
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<8x8xf32>):
// CHECK: %[[SQRT:.*]] = mfuse.sqrt %[[ARG2]]
// CHECK: %[[RSQRT:.*]] = mfuse.rsqrt %[[SQRT]]
// CHECK: mfuse.yield %[[RSQRT]]
// CHECK: return %[[FUSED]]
func.func @test_sqrt_rsqrt_chain(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = mfuse.sqrt %arg0 : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = mfuse.rsqrt %0 : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// Test scenario: Power and reciprocal operations in sequence
// CHECK-LABEL: func @test_pow_reciprocal_chain
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[POW:.*]] = mfuse.pow %[[ARG2]], %[[ARG3]]
// CHECK: %[[RECIP:.*]] = mfuse.reciprocal %[[POW]]
// CHECK: mfuse.yield %[[RECIP]]
// CHECK: return %[[FUSED]]
func.func @test_pow_reciprocal_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.pow %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.reciprocal %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Complex chain combining arithmetic and math functions (mul, sqrt, exp)
// CHECK-LABEL: func @test_complex_math_chain
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[SQRT:.*]] = mfuse.sqrt %[[MUL]]
// CHECK: %[[EXP:.*]] = mfuse.exp %[[SQRT]]
// CHECK: mfuse.yield %[[EXP]]
// CHECK: return %[[FUSED]]
func.func @test_complex_math_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.mul %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = mfuse.sqrt %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = mfuse.exp %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}
}
