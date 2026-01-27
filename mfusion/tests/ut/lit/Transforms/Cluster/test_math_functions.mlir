// RUN: mfusion-opt %s --muse-dvm-cluster | FileCheck %s

module {
// Test math function clustering: exp, log, sqrt, rsqrt, pow, reciprocal

// Test scenario: Exponential and logarithm operations in sequence
// CHECK-LABEL: func @test_exp_log_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = muse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[EXP:.*]] = muse.exp %[[ADD]]
// CHECK: %[[LOG:.*]] = muse.log %[[EXP]]
// CHECK: muse.yield %[[LOG]]
// CHECK: return %[[FUSED]]
func.func @test_exp_log_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.add %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.exp %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = muse.log %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}

// Test scenario: Square root and reciprocal square root operations in sequence
// CHECK-LABEL: func @test_sqrt_rsqrt_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<8x8xf32>) -> tensor<8x8xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<8x8xf32>):
// CHECK: %[[SQRT:.*]] = muse.sqrt %[[ARG2]]
// CHECK: %[[RSQRT:.*]] = muse.rsqrt %[[SQRT]]
// CHECK: muse.yield %[[RSQRT]]
// CHECK: return %[[FUSED]]
func.func @test_sqrt_rsqrt_chain(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = muse.sqrt %arg0 : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = muse.rsqrt %0 : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// Test scenario: Power and reciprocal operations in sequence
// CHECK-LABEL: func @test_pow_reciprocal_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[POW:.*]] = muse.pow %[[ARG2]], %[[ARG3]]
// CHECK: %[[RECIP:.*]] = muse.reciprocal %[[POW]]
// CHECK: muse.yield %[[RECIP]]
// CHECK: return %[[FUSED]]
func.func @test_pow_reciprocal_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.pow %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.reciprocal %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// Test scenario: Complex chain combining arithmetic and math functions (mul, sqrt, exp)
// CHECK-LABEL: func @test_complex_math_chain
// CHECK: %[[FUSED:.*]] = muse.fused %arg0, %arg1
// CHECK-SAME: {fusion_type = "dvm"}
// CHECK-SAME: : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[MUL:.*]] = muse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[SQRT:.*]] = muse.sqrt %[[MUL]]
// CHECK: %[[EXP:.*]] = muse.exp %[[SQRT]]
// CHECK: muse.yield %[[EXP]]
// CHECK: return %[[FUSED]]
func.func @test_complex_math_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = muse.mul %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %1 = muse.sqrt %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  %2 = muse.exp %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %2 : tensor<4x4xf32>
}
}
