// RUN: mfusion-opt %s --split | FileCheck %s

module {
// Test exp and log chain
// CHECK-LABEL: func @test_exp_log_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[EXP:.*]] = mfuse.exp %[[ADD]]
// CHECK: %[[LOG:.*]] = mfuse.log %[[EXP]]
// CHECK: mfuse.yield %[[LOG]]
// CHECK: return %[[FUSED]]
func.func @test_exp_log_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.exp %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.log %2 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %3 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test sqrt and rsqrt chain
// CHECK-LABEL: func @test_sqrt_rsqrt_chain
// CHECK-SAME: %arg0: tensor<8x8xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG1:.*]]: tensor<8x8xf32>):
// CHECK: %[[SQRT:.*]] = mfuse.sqrt %[[ARG1]]
// CHECK: %[[RSQRT:.*]] = mfuse.rsqrt %[[SQRT]]
// CHECK: mfuse.yield %[[RSQRT]]
// CHECK: return %[[FUSED]]
func.func @test_sqrt_rsqrt_chain(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<8x8xf32>) -> tensor<8x8xf32> {
  ^bb0(%arg1: tensor<8x8xf32>):
    %1 = mfuse.sqrt %arg1 : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %2 = mfuse.rsqrt %1 : (tensor<8x8xf32>) -> tensor<8x8xf32>
    mfuse.yield %2 : tensor<8x8xf32>
  }
  return %0 : tensor<8x8xf32>
}

// Test pow and reciprocal chain
// CHECK-LABEL: func @test_pow_reciprocal_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[POW:.*]] = mfuse.pow %[[ARG2]], %[[ARG3]]
// CHECK: %[[RECIP:.*]] = mfuse.reciprocal %[[POW]]
// CHECK: mfuse.yield %[[RECIP]]
// CHECK: return %[[FUSED]]
func.func @test_pow_reciprocal_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.pow %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.reciprocal %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test complex math chain
// CHECK-LABEL: func @test_complex_math_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[SQRT:.*]] = mfuse.sqrt %[[MUL]]
// CHECK: %[[EXP:.*]] = mfuse.exp %[[SQRT]]
// CHECK: mfuse.yield %[[EXP]]
// CHECK: return %[[FUSED]]
func.func @test_complex_math_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.mul %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.sqrt %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.exp %2 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %3 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}
}
