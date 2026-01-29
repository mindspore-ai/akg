// RUN: mfusion-opt %s --split | FileCheck %s

module {
// Test basic add-mul split
// CHECK-LABEL: func @test_basic_add_mul_split
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG2]]
// CHECK: mfuse.yield %[[MUL]]
// CHECK: return %[[FUSED]]
func.func @test_basic_add_mul_split(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.mul %1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test multiple splits
// CHECK-LABEL: func @test_multiple_splits
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>, %[[ARG5:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD1:.*]] = mfuse.add %[[ARG3]], %[[ARG4]]
// CHECK: %[[MUL1:.*]] = mfuse.mul %[[ADD1]], %[[ARG3]]
// CHECK: %[[ADD2:.*]] = mfuse.add %[[ARG4]], %[[ARG5]]
// CHECK: %[[MUL2:.*]] = mfuse.mul %[[ADD2]], %[[ARG4]]
// CHECK: %[[ADD3:.*]] = mfuse.add %[[MUL1]], %[[MUL2]]
// CHECK: mfuse.yield %[[ADD3]]
// CHECK: return %[[FUSED]]
func.func @test_multiple_splits(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>, %arg5: tensor<4x4xf32>):
    %1 = mfuse.add %arg3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.mul %1, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.add %arg4, %arg5 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = mfuse.mul %3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %5 = mfuse.add %2, %4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %5 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test element-wise chain
// CHECK-LABEL: func @test_element_wise_chain
// CHECK-SAME: %arg0: tensor<8x8xf32>
// CHECK-SAME: %arg1: tensor<8x8xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<8x8xf32>, %[[ARG3:.*]]: tensor<8x8xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG2]]
// CHECK: %[[SUB:.*]] = mfuse.sub %[[MUL]], %[[ARG3]]
// CHECK: %[[DIV:.*]] = mfuse.div %[[SUB]], %[[ARG2]]
// CHECK: mfuse.yield %[[DIV]]
// CHECK: return %[[FUSED]]
func.func @test_element_wise_chain(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32> {
  ^bb0(%arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    %2 = mfuse.mul %1, %arg2 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    %3 = mfuse.sub %2, %arg3 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    %4 = mfuse.div %3, %arg2 : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    mfuse.yield %4 : tensor<8x8xf32>
  }
  return %0 : tensor<8x8xf32>
}

// Test fuse op after all inputs
// CHECK-LABEL: func @test_fuse_op_after_all_inputs
// CHECK-SAME: %arg0: tensor<2x2xf16>
// CHECK-SAME: %arg1: tensor<2x2xf16>
// CHECK: %[[CST:.*]] = arith.constant dense<1> : tensor<i64>
// CHECK: %[[SUB:.*]] = mfuse.aclnn.sub %arg0, %arg1, %[[CST]]
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %[[SUB]] {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x2xf16>, %[[ARG3:.*]]: tensor<2x2xf16>, %[[ARG4:.*]]: tensor<2x2xf16>):
// CHECK: %[[MUL1:.*]] = mfuse.mul %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL2:.*]] = mfuse.mul %[[MUL1]], %[[MUL1]]
// CHECK: %[[MUL3:.*]] = mfuse.mul %[[ARG4]], %[[ARG4]]
// CHECK: %[[MUL4:.*]] = mfuse.mul %[[MUL2]], %[[MUL3]]
// CHECK: mfuse.yield %[[MUL4]]
// CHECK: return %[[FUSED]]
func.func @test_fuse_op_after_all_inputs(%arg0: tensor<2x2xf16>, %arg1: tensor<2x2xf16>) -> tensor<2x2xf16> {
  %cst = arith.constant dense<1> : tensor<i64>
  %0 = mfuse.aclnn.sub %arg0, %arg1, %cst : (tensor<2x2xf16>, tensor<2x2xf16>, tensor<i64>) -> tensor<2x2xf16>
  %1 = mfuse.fused %arg0, %arg1, %0 {fusion_type = "dvm"} : (tensor<2x2xf16>, tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16> {
  ^bb0(%arg2: tensor<2x2xf16>, %arg3: tensor<2x2xf16>, %arg4: tensor<2x2xf16>):
    %2 = mfuse.mul %arg2, %arg3 : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
    %3 = mfuse.mul %2, %2 : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
    %4 = mfuse.mul %arg4, %arg4 : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
    %5 = mfuse.mul %3, %4 : (tensor<2x2xf16>, tensor<2x2xf16>) -> tensor<2x2xf16>
    mfuse.yield %5 : tensor<2x2xf16>
  }
  return %1 : tensor<2x2xf16>
}

// Test split with external output
// CHECK-LABEL: func @test_split_with_external_output
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]]:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<4x4xf32>, %[[ARG3:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ADD]]
// CHECK: mfuse.yield %[[ADD]], %[[MUL]]
// CHECK: %[[TANH:.*]] = mfuse.aclnn.tanh %[[FUSED]]#0
// CHECK: return %[[TANH]], %[[FUSED]]#1
func.func @test_split_with_external_output(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  %0:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4x4xf32>) {
  ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
    %2 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.mul %2, %2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2, %3 : tensor<4x4xf32>, tensor<4x4xf32>
  }
  %1 = mfuse.aclnn.tanh %0#0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %1, %0#1 : tensor<4x4xf32>, tensor<4x4xf32>
}
}
