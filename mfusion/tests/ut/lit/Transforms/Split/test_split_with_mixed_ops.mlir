// RUN: mfusion-opt %s --split | FileCheck %s

module {
// Test split with mixed operations
// CHECK-LABEL: func @test_split_with_mixed_ops
// CHECK-SAME: %arg0: tensor<2x4xf32>
// CHECK-SAME: %arg1: tensor<2x4xf32>
// CHECK: %[[FUSED1:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x4xf32>, %[[ARG3:.*]]: tensor<2x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[SUB:.*]] = mfuse.sub %[[ADD]], %[[ARG2]]
// CHECK: mfuse.yield %[[SUB]]
// CHECK: %[[FUSED2:.*]] = mfuse.fused %arg0 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x4xf32>):
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG2]], %[[ARG2]]
// CHECK: %[[DIV:.*]] = mfuse.div %[[MUL]], %[[ARG2]]
// CHECK: mfuse.yield %[[DIV]]
// CHECK: return %[[FUSED1]], %[[FUSED2]]
func.func @test_split_with_mixed_ops(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  %0:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  ^bb0(%arg2: tensor<2x4xf32>, %arg3: tensor<2x4xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = mfuse.sub %1, %arg2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %3 = mfuse.mul %arg2, %arg2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %4 = mfuse.div %3, %arg2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    mfuse.yield %2, %4 : tensor<2x4xf32>, tensor<2x4xf32>
  }
  return %0#0, %0#1 : tensor<2x4xf32>, tensor<2x4xf32>
}

// Test split with math functions
// CHECK-LABEL: func @test_split_with_math_functions
// CHECK-SAME: %arg0: tensor<2x4xf32>
// CHECK-SAME: %arg1: tensor<2x4xf32>
// CHECK: %[[FUSED1:.*]] = mfuse.fused %arg0 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x4xf32>):
// CHECK: %[[SQRT:.*]] = mfuse.sqrt %[[ARG2]]
// CHECK: %[[ABS:.*]] = mfuse.abs %[[SQRT]]
// CHECK: mfuse.yield %[[ABS]]
// CHECK: %[[FUSED2:.*]] = mfuse.fused %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x4xf32>):
// CHECK: %[[EXP:.*]] = mfuse.exp %[[ARG2]]
// CHECK: %[[LOG:.*]] = mfuse.log %[[EXP]]
// CHECK: mfuse.yield %[[LOG]]
// CHECK: return %[[FUSED1]], %[[FUSED2]]
func.func @test_split_with_math_functions(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  %0:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  ^bb0(%arg2: tensor<2x4xf32>, %arg3: tensor<2x4xf32>):
    %1 = mfuse.sqrt %arg2 : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %2 = mfuse.abs %1 : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %3 = mfuse.exp %arg3 : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %4 = mfuse.log %3 : (tensor<2x4xf32>) -> tensor<2x4xf32>
    mfuse.yield %2, %4 : tensor<2x4xf32>, tensor<2x4xf32>
  }
  return %0#0, %0#1 : tensor<2x4xf32>, tensor<2x4xf32>
}

// Test split with comparisons
// CHECK-LABEL: func @test_split_with_comparisons
// CHECK-SAME: %arg0: tensor<2x4xf32>
// CHECK-SAME: %arg1: tensor<2x4xf32>
// CHECK: %[[GT:.*]] = mfuse.gt %arg0, %arg1
// CHECK: %[[LE:.*]] = mfuse.le %arg0, %arg1
// CHECK: return %[[GT]], %[[LE]]
func.func @test_split_with_comparisons(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>) -> (tensor<2x4xi1>, tensor<2x4xi1>) {
  %0:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xi1>, tensor<2x4xi1>) {
  ^bb0(%arg2: tensor<2x4xf32>, %arg3: tensor<2x4xf32>):
    %1 = mfuse.gt %arg2, %arg3 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
    %2 = mfuse.le %arg2, %arg3 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xi1>
    mfuse.yield %1, %2 : tensor<2x4xi1>, tensor<2x4xi1>
  }
  return %0#0, %0#1 : tensor<2x4xi1>, tensor<2x4xi1>
}

// Test split into two operations
// CHECK-LABEL: func @test_split_into_two
// CHECK-SAME: %arg0: tensor<2x4xf32>
// CHECK-SAME: %arg1: tensor<2x4xf32>
// CHECK-SAME: %arg2: tensor<2x4xf32>
// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00> : tensor<f64>
// CHECK: %[[CST_0:.*]] = arith.constant dense<2.000000e+00> : tensor<f64>
// CHECK: %[[ADD:.*]] = mfuse.add %arg0, %arg1
// CHECK: %[[MUL:.*]] = mfuse.mul %arg2, %arg2
// CHECK: return %[[ADD]], %[[MUL]]
func.func @test_split_into_two(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xf32>, %arg2: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  %0 = arith.constant dense<1.000000e+00> : tensor<f64>
  %1 = arith.constant dense<2.000000e+00> : tensor<f64>
  // Fused operation that contains two independent computation chains
  %2:2 = mfuse.fused %arg0, %arg1, %arg2, %0, %1 {fusion_type = "dvm"} : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>, tensor<f64>, tensor<f64>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  ^bb0(%arg3: tensor<2x4xf32>, %arg4: tensor<2x4xf32>, %arg5: tensor<2x4xf32>, %arg6: tensor<f64>, %arg7: tensor<f64>):
    // First computation chain: arg0 + arg1 -> result1
    %3 = mfuse.add %arg3, %arg4 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    // Second computation chain: arg2 * arg2 -> result2
    %4 = mfuse.mul %arg5, %arg5 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    mfuse.yield %3, %4 : tensor<2x4xf32>, tensor<2x4xf32>
  }
  return %2#0, %2#1 : tensor<2x4xf32>, tensor<2x4xf32>
}
}
