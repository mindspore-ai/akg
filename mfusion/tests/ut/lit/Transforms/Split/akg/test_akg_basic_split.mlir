// RUN: mfusion-opt %s --pass-pipeline="builtin.module(func.func(split{kernel-generator=akg}))" | FileCheck %s

module {
// Test 1: Single FusedOp with multiple outputs split into two FusedOps
// This demonstrates the core split functionality: one FusedOp becomes two
// CHECK-LABEL: func @test_split_into_two_fused_ops
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// Original: one FusedOp with 2 outputs
// After split: two separate FusedOps
// CHECK: %[[FUSED1:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "akg"}
// CHECK-NEXT: ^bb0(%[[A1:.*]]: tensor<4x4xf32>, %[[A2:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[A1]], %[[A2]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[A1]]
// CHECK: mfuse.yield %[[MUL]]
// CHECK: %[[FUSED2:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "akg"}
// CHECK-NEXT: ^bb0(%[[B1:.*]]: tensor<4x4xf32>, %[[B2:.*]]: tensor<4x4xf32>):
// CHECK: %[[SUB:.*]] = mfuse.sub %[[B1]], %[[B2]]
// CHECK: %[[DIV:.*]] = mfuse.div %[[SUB]], %[[B2]]
// CHECK: mfuse.yield %[[DIV]]
// CHECK: %[[POW:.*]] = mfuse.pow %[[FUSED1]], %[[FUSED2]]
// CHECK: return %[[POW]]
func.func @test_split_into_two_fused_ops(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0:2 = mfuse.fused %arg0, %arg1 {fusion_type = "akg"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.mul %1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.sub %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = mfuse.div %3, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2, %4 : tensor<4x4xf32>, tensor<4x4xf32>
  }
  %5 = mfuse.pow %0#0, %0#1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %5 : tensor<4x4xf32>
}

// Test 2: Single FusedOp with 3 outputs split into three FusedOps
// Each output has a data dependency chain, so they will be kept as FusedOps
// CHECK-LABEL: func @test_split_into_three_fused_ops
// CHECK-SAME: %arg0: tensor<4x4xf32>
// Original: one FusedOp with 3 outputs (each with dependency chain)
// After split: three separate FusedOps
// CHECK: %[[FUSED1:.*]] = mfuse.fused %arg0 {fusion_type = "akg"}
// CHECK: mfuse.exp
// CHECK: mfuse.sqrt
// CHECK: mfuse.yield
// CHECK: %[[FUSED2:.*]] = mfuse.fused %arg0 {fusion_type = "akg"}
// CHECK: mfuse.log
// CHECK: mfuse.reciprocal
// CHECK: mfuse.yield
// CHECK: %[[FUSED3:.*]] = mfuse.fused %arg0 {fusion_type = "akg"}
// CHECK: mfuse.abs
// CHECK: mfuse.neg
// CHECK: mfuse.yield
// CHECK: %[[ADD1:.*]] = mfuse.add %[[FUSED1]], %[[FUSED2]]
// CHECK: %[[ADD2:.*]] = mfuse.add %[[ADD1]], %[[FUSED3]]
// CHECK: return %[[ADD2]]
func.func @test_split_into_three_fused_ops(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0:3 = mfuse.fused %arg0 {fusion_type = "akg"} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  ^bb0(%arg1: tensor<4x4xf32>):
    %1 = mfuse.exp %arg1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.sqrt %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.log %arg1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = mfuse.reciprocal %3 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %5 = mfuse.abs %arg1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %6 = mfuse.neg %5 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2, %4, %6 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
  }
  %7 = mfuse.add %0#0, %0#1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  %8 = mfuse.add %7, %0#2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %8 : tensor<4x4xf32>
}

// Test 3: Complex FusedOp with arithmetic operations split into multiple FusedOps
// CHECK-LABEL: func @test_split_arithmetic_chain
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// Original: one FusedOp with complex arithmetic chain
// After split: multiple FusedOps
// CHECK: %[[FUSED1:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "akg"}
// CHECK: mfuse.add
// CHECK: mfuse.mul
// CHECK: mfuse.yield
// CHECK: %[[FUSED2:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "akg"}
// CHECK: mfuse.sub
// CHECK: mfuse.div
// CHECK: mfuse.yield
// CHECK: %[[RESULT:.*]] = mfuse.add %[[FUSED1]], %[[FUSED2]]
// CHECK: return %[[RESULT]]
func.func @test_split_arithmetic_chain(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0:2 = mfuse.fused %arg0, %arg1 {fusion_type = "akg"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.mul %1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.sub %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = mfuse.div %3, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2, %4 : tensor<4x4xf32>, tensor<4x4xf32>
  }
  %5 = mfuse.add %0#0, %0#1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %5 : tensor<4x4xf32>
}

// Test 4: FusedOp with math functions split into multiple FusedOps
// CHECK-LABEL: func @test_split_math_functions
// CHECK-SAME: %arg0: tensor<4x4xf32>
// Original: one FusedOp with multiple math functions
// After split: multiple FusedOps
// CHECK: %[[FUSED1:.*]] = mfuse.fused %arg0 {fusion_type = "akg"}
// CHECK: mfuse.log
// CHECK: mfuse.sqrt
// CHECK: mfuse.yield
// CHECK: %[[FUSED2:.*]] = mfuse.fused %arg0 {fusion_type = "akg"}
// CHECK: mfuse.exp
// CHECK: mfuse.reciprocal
// CHECK: mfuse.yield
// CHECK: %[[MUL:.*]] = mfuse.mul %[[FUSED1]], %[[FUSED2]]
// CHECK: return %[[MUL]]
func.func @test_split_math_functions(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0:2 = mfuse.fused %arg0 {fusion_type = "akg"} : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  ^bb0(%arg1: tensor<4x4xf32>):
    %1 = mfuse.log %arg1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.sqrt %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.exp %arg1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = mfuse.reciprocal %3 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2, %4 : tensor<4x4xf32>, tensor<4x4xf32>
  }
  %5 = mfuse.mul %0#0, %0#1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %5 : tensor<4x4xf32>
}

// Test 5: Verify fusion_type filtering - AKG processor skips DVM FusedOps
// CHECK-LABEL: func @test_fusion_type_filter
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// DVM FusedOp should remain unchanged (not split by AKG processor)
// CHECK: %[[DVM:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK-NEXT: ^bb0(%[[D1:.*]]: tensor<4x4xf32>, %[[D2:.*]]: tensor<4x4xf32>):
// CHECK: %[[DADD:.*]] = mfuse.add %[[D1]], %[[D2]]
// CHECK: mfuse.yield %[[DADD]]
// AKG FusedOp with data dependency chains should be split into two FusedOps
// CHECK: %[[AKG1:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "akg"}
// CHECK: mfuse.mul
// CHECK: mfuse.sqrt
// CHECK: mfuse.yield
// CHECK: %[[AKG2:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "akg"}
// CHECK: mfuse.div
// CHECK: mfuse.reciprocal
// CHECK: mfuse.yield
// CHECK: return %[[DVM]], %[[AKG1]], %[[AKG2]]
func.func @test_fusion_type_filter(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %1 : tensor<4x4xf32>
  }
  %2:2 = mfuse.fused %arg0, %arg1 {fusion_type = "akg"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %3 = mfuse.mul %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = mfuse.sqrt %3 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %5 = mfuse.div %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %6 = mfuse.reciprocal %5 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %4, %6 : tensor<4x4xf32>, tensor<4x4xf32>
  }
  return %0, %2#0, %2#1 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>
}
}
