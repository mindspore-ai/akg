// RUN: mfusion-opt %s --split | FileCheck %s

module {
// Test logical and/or operations
// CHECK-LABEL: func @test_logical_and_or
// CHECK-SAME: %arg0: tensor<4x4xi1>
// CHECK-SAME: %arg1: tensor<4x4xi1>
// CHECK: %[[AND:.*]] = mfuse.logical_and %arg0, %arg1
// CHECK: %[[OR:.*]] = mfuse.logical_or %[[AND]], %arg0
// CHECK: return %[[OR]]
func.func @test_logical_and_or(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xi1>) -> tensor<4x4xi1> {
  %0 = mfuse.logical_and %arg0, %arg1 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  %1 = mfuse.logical_or %0, %arg0 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  return %1 : tensor<4x4xi1>
}

// Test select operation
// CHECK-LABEL: func @test_select_operation
// CHECK-SAME: %arg0: tensor<4x4xi1>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg1, %arg2, %arg0 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>, %[[ARG5:.*]]: tensor<4x4xi1>):
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG3]], %[[ARG4]]
// CHECK: %[[SELECT:.*]] = mfuse.select %[[ARG5]], %[[MUL]], %[[ARG3]]
// CHECK: mfuse.yield %[[SELECT]]
// CHECK: return %[[FUSED]]
func.func @test_select_operation(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg1, %arg2, %arg0 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1>) -> tensor<4x4xf32> {
  ^bb0(%arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>, %arg5: tensor<4x4xi1>):
    %1 = mfuse.mul %arg3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %2 = mfuse.select %arg5, %1, %arg3 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %2 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test complex logical select
// CHECK-LABEL: func @test_complex_logical_select
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>, %[[ARG5:.*]]: tensor<4x4xf32>):
// CHECK: %[[GT:.*]] = mfuse.gt %[[ARG3]], %[[ARG4]]
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG3]], %[[ARG5]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ARG4]], %[[ARG5]]
// CHECK: %[[SELECT:.*]] = mfuse.select %[[GT]], %[[ADD]], %[[MUL]]
// CHECK: mfuse.yield %[[SELECT]]
// CHECK: return %[[FUSED]]
func.func @test_complex_logical_select(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>, %arg5: tensor<4x4xf32>):
    %1 = mfuse.gt %arg3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %2 = mfuse.add %arg3, %arg5 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.mul %arg4, %arg5 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %4 = mfuse.select %1, %2, %3 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %4 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// Test logical comparison mix
// CHECK-LABEL: func @test_logical_comparison_mix
// CHECK-SAME: %arg0: tensor<4x4xf32>
// CHECK-SAME: %arg1: tensor<4x4xf32>
// CHECK-SAME: %arg2: tensor<4x4xf32>
// CHECK: %[[GT:.*]] = mfuse.gt %arg0, %arg1
// CHECK: %[[LT:.*]] = mfuse.lt %arg1, %arg2
// CHECK: %[[AND:.*]] = mfuse.logical_and %[[GT]], %[[LT]]
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg2, %[[AND]], %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<4x4xf32>, %[[ARG4:.*]]: tensor<4x4xf32>, %[[ARG5:.*]]: tensor<4x4xi1>, %[[ARG6:.*]]: tensor<4x4xf32>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG3]], %[[ARG4]]
// CHECK: %[[SELECT:.*]] = mfuse.select %[[ARG5]], %[[ADD]], %[[ARG6]]
// CHECK: mfuse.yield %[[SELECT]]
// CHECK: return %[[FUSED]]
func.func @test_logical_comparison_mix(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.gt %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  %1 = mfuse.lt %arg1, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  %2 = mfuse.logical_and %0, %1 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  %3 = mfuse.fused %arg0, %arg2, %2, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1>, tensor<4x4xf32>) -> tensor<4x4xf32> {
  ^bb0(%arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>, %arg5: tensor<4x4xi1>, %arg6: tensor<4x4xf32>):
    %4 = mfuse.add %arg3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    %5 = mfuse.select %arg5, %4, %arg6 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    mfuse.yield %5 : tensor<4x4xf32>
  }
  return %3 : tensor<4x4xf32>
}
}
