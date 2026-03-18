// RUN: mfusion-opt %s --split | FileCheck %s

module {
// Test: dynamic shape without symbolic info - should NOT fuse
// CHECK-LABEL: func @test_dynamic_shape_no_symbol
// CHECK-SAME: %arg0: tensor<?xf32>
// CHECK-SAME: %arg1: tensor<?xf32>
// CHECK-NOT: mfuse.fused
// CHECK: mfuse.add
// CHECK: mfuse.mul
// CHECK: return
func.func @test_dynamic_shape_no_symbol(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32> {
  ^bb0(%arg2: tensor<?xf32>, %arg3: tensor<?xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    %2 = mfuse.mul %1, %arg2 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    mfuse.yield %2 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

// Test: dynamic shape with different symbols - should NOT fuse
// CHECK-LABEL: func @test_dynamic_shape_different_symbols
// CHECK-SAME: %arg0: tensor<?xf32, #mfuse.symshape<["s0"]>>
// CHECK-SAME: %arg1: tensor<?xf32, #mfuse.symshape<["s1"]>>
// CHECK-NOT: mfuse.fused
// CHECK: mfuse.add
// CHECK: mfuse.mul
// CHECK: return
func.func @test_dynamic_shape_different_symbols(%arg0: tensor<?xf32, #mfuse.symshape<["s0"]>>, %arg1: tensor<?xf32, #mfuse.symshape<["s1"]>>) -> tensor<?xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s1"]>>) -> tensor<?xf32> {
  ^bb0(%arg2: tensor<?xf32, #mfuse.symshape<["s0"]>>, %arg3: tensor<?xf32, #mfuse.symshape<["s1"]>>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s1"]>>) -> tensor<?xf32>
    %2 = mfuse.mul %1, %arg2 : (tensor<?xf32>, tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32>
    mfuse.yield %2 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

// Test: dynamic shape with same symbols - should fuse
// CHECK-LABEL: func @test_dynamic_shape_same_symbols
// CHECK-SAME: %arg0: tensor<?xf32, #mfuse.symshape<["s0"]>>
// CHECK-SAME: %arg1: tensor<?xf32, #mfuse.symshape<["s0"]>>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<?xf32, #mfuse.symshape<["s0"]>>, %[[ARG3:.*]]: tensor<?xf32, #mfuse.symshape<["s0"]>>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG2]]
// CHECK: mfuse.yield %[[MUL]]
// CHECK: return %[[FUSED]]
func.func @test_dynamic_shape_same_symbols(%arg0: tensor<?xf32, #mfuse.symshape<["s0"]>>, %arg1: tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32, #mfuse.symshape<["s0"]>> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32, #mfuse.symshape<["s0"]>> {
  ^bb0(%arg2: tensor<?xf32, #mfuse.symshape<["s0"]>>, %arg3: tensor<?xf32, #mfuse.symshape<["s0"]>>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32, #mfuse.symshape<["s0"]>>
    %2 = mfuse.mul %1, %arg2 : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32, #mfuse.symshape<["s0"]>>
    mfuse.yield %2 : tensor<?xf32, #mfuse.symshape<["s0"]>>
  }
  return %0 : tensor<?xf32, #mfuse.symshape<["s0"]>>
}

// Test: 2D dynamic shape with same symbols - should fuse
// CHECK-LABEL: func @test_2d_dynamic_shape_same_symbols
// CHECK-SAME: %arg0: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
// CHECK-SAME: %arg1: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, %[[ARG3:.*]]: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG2]]
// CHECK: mfuse.yield %[[MUL]]
// CHECK: return %[[FUSED]]
func.func @test_2d_dynamic_shape_same_symbols(%arg0: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, %arg1: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>) -> tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>) -> tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>> {
  ^bb0(%arg2: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, %arg3: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>) -> tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
    %2 = mfuse.mul %1, %arg2 : (tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>) -> tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
    mfuse.yield %2 : tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
  }
  return %0 : tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
}

// Test: 2D dynamic shape with partially different symbols - should NOT fuse
// CHECK-LABEL: func @test_2d_dynamic_shape_partial_different_symbols
// CHECK-SAME: %arg0: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
// CHECK-SAME: %arg1: tensor<?x?xf32, #mfuse.symshape<["s0", "s2"]>>
// CHECK-NOT: mfuse.fused
// CHECK: mfuse.add
// CHECK: mfuse.mul
// CHECK: return
func.func @test_2d_dynamic_shape_partial_different_symbols(%arg0: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, %arg1: tensor<?x?xf32, #mfuse.symshape<["s0", "s2"]>>) -> tensor<?x?xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, tensor<?x?xf32, #mfuse.symshape<["s0", "s2"]>>) -> tensor<?x?xf32> {
  ^bb0(%arg2: tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, %arg3: tensor<?x?xf32, #mfuse.symshape<["s0", "s2"]>>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, tensor<?x?xf32, #mfuse.symshape<["s0", "s2"]>>) -> tensor<?x?xf32>
    %2 = mfuse.mul %1, %arg2 : (tensor<?x?xf32>, tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>) -> tensor<?x?xf32>
    mfuse.yield %2 : tensor<?x?xf32>
  }
  return %0 : tensor<?x?xf32>
}

// Test: mixed static and dynamic shape with same symbols - should fuse
// CHECK-LABEL: func @test_mixed_static_dynamic_same_symbols
// CHECK-SAME: %arg0: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
// CHECK-SAME: %arg1: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG2:.*]]: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, %[[ARG3:.*]]: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>):
// CHECK: %[[ADD:.*]] = mfuse.add %[[ARG2]], %[[ARG3]]
// CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ARG2]]
// CHECK: mfuse.yield %[[MUL]]
// CHECK: return %[[FUSED]]
func.func @test_mixed_static_dynamic_same_symbols(%arg0: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, %arg1: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>> {
  ^bb0(%arg2: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, %arg3: tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
    %2 = mfuse.mul %1, %arg2 : (tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
    mfuse.yield %2 : tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  }
  return %0 : tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
}

// Test: one input has symbol, another has no symbol - should NOT fuse
// CHECK-LABEL: func @test_one_symbolic_one_no_symbol
// CHECK-SAME: %arg0: tensor<?xf32, #mfuse.symshape<["s0"]>>
// CHECK-SAME: %arg1: tensor<?xf32>
// CHECK-NOT: mfuse.fused
// CHECK: mfuse.add
// CHECK: mfuse.mul
// CHECK: return
func.func @test_one_symbolic_one_no_symbol(%arg0: tensor<?xf32, #mfuse.symshape<["s0"]>>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32>) -> tensor<?xf32> {
  ^bb0(%arg2: tensor<?xf32, #mfuse.symshape<["s0"]>>, %arg3: tensor<?xf32>):
    %1 = mfuse.add %arg2, %arg3 : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32>) -> tensor<?xf32>
    %2 = mfuse.mul %1, %arg2 : (tensor<?xf32>, tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32>
    mfuse.yield %2 : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

// Test: three inputs with same symbols - should fuse
// CHECK-LABEL: func @test_three_inputs_same_symbols
// CHECK-SAME: %arg0: tensor<?xf32, #mfuse.symshape<["s0"]>>
// CHECK-SAME: %arg1: tensor<?xf32, #mfuse.symshape<["s0"]>>
// CHECK-SAME: %arg2: tensor<?xf32, #mfuse.symshape<["s0"]>>
// CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"}
// CHECK: ^bb0(%[[ARG3:.*]]: tensor<?xf32, #mfuse.symshape<["s0"]>>, %[[ARG4:.*]]: tensor<?xf32, #mfuse.symshape<["s0"]>>, %[[ARG5:.*]]: tensor<?xf32, #mfuse.symshape<["s0"]>>):
// CHECK: %[[ADD1:.*]] = mfuse.add %[[ARG3]], %[[ARG4]]
// CHECK: %[[MUL1:.*]] = mfuse.mul %[[ADD1]], %[[ARG3]]
// CHECK: %[[ADD2:.*]] = mfuse.add %[[ARG4]], %[[ARG5]]
// CHECK: %[[MUL2:.*]] = mfuse.mul %[[ADD2]], %[[ARG4]]
// CHECK: %[[ADD3:.*]] = mfuse.add %[[MUL1]], %[[MUL2]]
// CHECK: mfuse.yield %[[ADD3]]
// CHECK: return %[[FUSED]]
func.func @test_three_inputs_same_symbols(%arg0: tensor<?xf32, #mfuse.symshape<["s0"]>>, %arg1: tensor<?xf32, #mfuse.symshape<["s0"]>>, %arg2: tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32, #mfuse.symshape<["s0"]>> {
  %0 = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"} : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32, #mfuse.symshape<["s0"]>> {
  ^bb0(%arg3: tensor<?xf32, #mfuse.symshape<["s0"]>>, %arg4: tensor<?xf32, #mfuse.symshape<["s0"]>>, %arg5: tensor<?xf32, #mfuse.symshape<["s0"]>>):
    %1 = mfuse.add %arg3, %arg4 : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32, #mfuse.symshape<["s0"]>>
    %2 = mfuse.mul %1, %arg3 : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32, #mfuse.symshape<["s0"]>>
    %3 = mfuse.add %arg4, %arg5 : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32, #mfuse.symshape<["s0"]>>
    %4 = mfuse.mul %3, %arg4 : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32, #mfuse.symshape<["s0"]>>
    %5 = mfuse.add %2, %4 : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>) -> tensor<?xf32, #mfuse.symshape<["s0"]>>
    mfuse.yield %5 : tensor<?xf32, #mfuse.symshape<["s0"]>>
  }
  return %0 : tensor<?xf32, #mfuse.symshape<["s0"]>>
}
}
