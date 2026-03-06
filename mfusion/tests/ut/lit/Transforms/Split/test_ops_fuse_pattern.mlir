// RUN: mfusion-opt %s --split -mlir-print-ir-after-all | FileCheck %s

module {
  // Test elemwise operations with complex dependencies
  // CHECK-LABEL: func @test_elemwise_operations
  // CHECK-SAME: %arg0: tensor<4x4xf32>
  // CHECK-SAME: %arg1: tensor<4x4xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
  // CHECK: mfuse.add
  // CHECK: mfuse.sub
  // CHECK: mfuse.mul
  // CHECK: mfuse.div
  // CHECK: mfuse.pow
  // CHECK: mfuse.abs
  // CHECK: mfuse.neg
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @test_elemwise_operations(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
      %1 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %2 = mfuse.sub %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %3 = mfuse.mul %1, %2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %4 = mfuse.div %3, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %5 = mfuse.pow %4, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %6 = mfuse.abs %5 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      %7 = mfuse.neg %6 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %7 : tensor<4x4xf32>
    }
    return %0 : tensor<4x4xf32>
  }

  // Test math functions with complex dependencies
  // CHECK-LABEL: func @test_math_functions
  // CHECK-SAME: %arg0: tensor<4x4xf32>
  // CHECK: %[[EXP:.*]] = mfuse.exp %arg0
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0
  // CHECK: mfuse.log
  // CHECK: mfuse.sqrt
  // CHECK: mfuse.reciprocal
  // CHECK: mfuse.ceil
  // CHECK: mfuse.trunc
  // CHECK: mfuse.yield
  // CHECK: %[[RSQRT:.*]] = mfuse.rsqrt %[[EXP]]
  // CHECK: %[[FLOOR:.*]] = mfuse.floor %[[RSQRT]]
  // CHECK: return %[[FUSED]]
  func.func @test_math_functions(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = mfuse.exp %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %1 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%arg1: tensor<4x4xf32>):
      %4 = mfuse.log %arg1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      %5 = mfuse.sqrt %4 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      %6 = mfuse.reciprocal %5 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      %7 = mfuse.ceil %6 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      %8 = mfuse.trunc %7 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %8 : tensor<4x4xf32>
    }
    %2 = mfuse.rsqrt %0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    %3 = mfuse.floor %2 : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }

  // Test comparison operations with complex dependencies
  // CHECK-LABEL: func @test_comparison_operations
  // CHECK-SAME: %arg0: tensor<4x4xf32>
  // CHECK-SAME: %arg1: tensor<4x4xf32>
  // CHECK: %[[FUSED:.*]]:2 = mfuse.fused %arg0, %arg1
  // CHECK: mfuse.add
  // CHECK: mfuse.sub
  // CHECK: mfuse.eq
  // CHECK: mfuse.gt
  // CHECK: mfuse.ge
  // CHECK: mfuse.lt
  // CHECK: mfuse.yield
  // CHECK: %[[NE:.*]] = mfuse.ne %arg0, %arg1
  // CHECK: %[[LE:.*]] = mfuse.le %[[FUSED]]#0, %[[FUSED]]#1
  // CHECK: return %[[LE]]
  func.func @test_comparison_operations(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
    %0:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> (tensor<4x4xi1>, tensor<4x4xi1>) {
    ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
      %3 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %4 = mfuse.sub %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %5 = mfuse.eq %3, %4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
      %6 = mfuse.gt %3, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
      %7 = mfuse.ge %4, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
      %8 = mfuse.lt %3, %4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
      mfuse.yield %6, %7 : tensor<4x4xi1>, tensor<4x4xi1>
    }
    %1 = mfuse.ne %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %2 = mfuse.le %0#0, %0#1 : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
    return %2 : tensor<4x4xi1>
  }

  // Test logical operations with complex dependencies
  // CHECK-LABEL: func @test_logical_operations
  // CHECK-SAME: %arg0: tensor<4x4xf32>
  // CHECK-SAME: %arg1: tensor<4x4xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
  // CHECK: mfuse.logical_and
  // CHECK: mfuse.logical_not
  // CHECK: mfuse.logical_not
  // CHECK: mfuse.logical_and
  // CHECK: mfuse.logical_or
  // CHECK: mfuse.yield
  // CHECK: %[[LOGICAL_OR:.*]] = mfuse.logical_or %arg0, %arg1
  // CHECK: return %[[FUSED]]
  func.func @test_logical_operations(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
      %2 = mfuse.logical_and %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %3 = mfuse.logical_not %arg2 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      %4 = mfuse.logical_not %arg3 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      %5 = mfuse.logical_and %3, %4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %6 = mfuse.logical_or %2, %5 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %6 : tensor<4x4xf32>
    }
    %1 = mfuse.logical_or %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  // Test select operation with complex dependencies
  // CHECK-LABEL: func @test_select_operation
  // CHECK-SAME: %arg0: tensor<4x4xi1>
  // CHECK-SAME: %arg1: tensor<4x4xf32>
  // CHECK-SAME: %arg2: tensor<4x4xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg1, %arg2, %arg0
  // CHECK: mfuse.add
  // CHECK: mfuse.sub
  // CHECK: mfuse.select
  // CHECK: mfuse.abs
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @test_select_operation(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = mfuse.fused %arg1, %arg2, %arg0 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1>) -> tensor<4x4xf32> {
    ^bb0(%arg3: tensor<4x4xf32>, %arg4: tensor<4x4xf32>, %arg5: tensor<4x4xi1>):
      %1 = mfuse.add %arg3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %2 = mfuse.sub %arg3, %arg4 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %3 = mfuse.select %arg5, %1, %2 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %4 = mfuse.abs %3 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %4 : tensor<4x4xf32>
    }
    return %0 : tensor<4x4xf32>
  }

  // Test max min operations with complex dependencies
  // CHECK-LABEL: func @test_max_min_operations
  // CHECK-SAME: %arg0: tensor<4x4xf32>
  // CHECK-SAME: %arg1: tensor<4x4xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
  // CHECK: mfuse.maximum
  // CHECK: mfuse.minimum
  // CHECK: mfuse.add
  // CHECK: mfuse.div
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @test_max_min_operations(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
      %1 = mfuse.maximum %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %2 = mfuse.minimum %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %3 = mfuse.add %1, %2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %4 = mfuse.div %3, %arg2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %4 : tensor<4x4xf32>
    }
    return %0 : tensor<4x4xf32>
  }

  // Test reshape permute operations with complex dependencies
  // CHECK-LABEL: func @test_reshape_permute_operations
  // CHECK-SAME: %arg0: tensor<2x4xf32>
  // CHECK: %[[RESHAPE1:.*]] = mfuse.reshape %arg0
  // CHECK: %[[RESHAPE2:.*]] = mfuse.reshape %[[RESHAPE1]]
  // CHECK: %[[PERMUTE:.*]] = mfuse.permute %[[RESHAPE2]], []
  // CHECK: return %[[PERMUTE]]
  func.func @test_reshape_permute_operations(%arg0: tensor<2x4xf32>) -> tensor<4x2xf32> {
    %0 = mfuse.reshape %arg0 : (tensor<2x4xf32>) -> tensor<8xf32>
    %1 = mfuse.reshape %0 : (tensor<8xf32>) -> tensor<4x2xf32>
    %2 = mfuse.permute %1, [] : (tensor<4x2xf32>) -> tensor<4x2xf32>
    return %2 : tensor<4x2xf32>
  }

  // Test broadcast_to operation with complex dependencies
  // CHECK-LABEL: func @test_broadcast_to_operation
  // CHECK-SAME: %arg0: tensor<4xf32>
  // CHECK: %[[FUSED1:.*]] = mfuse.fused %arg0
  // CHECK: mfuse.add
  // CHECK: mfuse.mul
  // CHECK: mfuse.yield
  // CHECK: %[[FUSED2:.*]] = mfuse.fused %[[FUSED1]]
  // CHECK: mfuse.broadcast_to
  // CHECK: mfuse.add
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED2]]
  func.func @test_broadcast_to_operation(%arg0: tensor<4xf32>) -> tensor<2x4xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4xf32>) -> tensor<2x4xf32> {
    ^bb0(%arg1: tensor<4xf32>):
      %1 = mfuse.add %arg1, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %2 = mfuse.mul %1, %1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %3 = mfuse.broadcast_to %2 : (tensor<4xf32>) -> tensor<2x4xf32>
      %4 = mfuse.add %3, %3 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %5 = mfuse.add %4, %3 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %6 = mfuse.add %4, %5 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      mfuse.yield %6 : tensor<2x4xf32>
    }
    return %0 : tensor<2x4xf32>
  }

  // Test reduce_sum operation with complex dependencies
  // CHECK-LABEL: func @test_reduce_sum_operation
  // CHECK-SAME: %arg0: tensor<2x4xf32>
  // CHECK: %[[FUSED1:.*]] = mfuse.fused %arg0
  // CHECK: mfuse.add
  // CHECK: mfuse.mul
  // CHECK: mfuse.sqrt
  // CHECK: mfuse.abs
  // CHECK: mfuse.reduce_sum
  // CHECK: mfuse.yield
  // CHECK: %[[FUSED2:.*]] = mfuse.fused %arg0, %[[FUSED1]]
  // CHECK: mfuse.add
  // CHECK: mfuse.mul
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED2]]
  func.func @test_reduce_sum_operation(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<2x4xf32>) -> tensor<2x4xf32> {
    ^bb0(%arg1: tensor<2x4xf32>):
      %1 = mfuse.add %arg1, %arg1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %2 = mfuse.mul %1, %1 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      %3 = mfuse.sqrt %2 : (tensor<2x4xf32>) -> tensor<2x4xf32>
      %4 = mfuse.abs %3 : (tensor<2x4xf32>) -> tensor<2x4xf32>
      %5 = mfuse.reduce_sum %4 {dimensions = [0], dtype = f32, keepdim = true} : (tensor<2x4xf32>) -> tensor<1x4xf32>
      %6 = mfuse.add %arg1, %5 : (tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
      %7 = mfuse.mul %6, %6 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
      mfuse.yield %7 : tensor<2x4xf32>
    }
    return %0 : tensor<2x4xf32>
  }

  // Test matmul operation with complex dependencies
  // CHECK-LABEL: func @test_matmul_operation
  // CHECK-SAME: %arg0: tensor<2x4xf32>
  // CHECK-SAME: %arg1: tensor<4x3xf32>
  // CHECK: %[[MATMUL:.*]] = mfuse.matmul %arg0, %arg1
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[MATMUL]]
  // CHECK: mfuse.add
  // CHECK: mfuse.mul
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @test_matmul_operation(%arg0: tensor<2x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<2x3xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %1 = mfuse.fused %0 {fusion_type = "dvm"} : (tensor<2x3xf32>) -> tensor<2x3xf32> {
    ^bb0(%arg2: tensor<2x3xf32>):
      %2 = mfuse.add %arg2, %arg2 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
      %3 = mfuse.mul %2, %arg2 : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
      mfuse.yield %3 : tensor<2x3xf32>
    }
    return %1 : tensor<2x3xf32>
  }

  // Test batch_matmul operation with complex dependencies
  // CHECK-LABEL: func @test_batch_matmul_operation
  // CHECK-SAME: %arg0: tensor<2x3x4xf32>
  // CHECK-SAME: %arg1: tensor<2x4x5xf32>
  // CHECK: %[[BATCH_MATMUL:.*]] = mfuse.batch_matmul %arg0, %arg1
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[BATCH_MATMUL]]
  // CHECK: mfuse.add
  // CHECK: mfuse.mul
  // CHECK: mfuse.exp
  // CHECK: mfuse.abs
  // CHECK: mfuse.log
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @test_batch_matmul_operation(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
    %0 = mfuse.batch_matmul %arg0, %arg1 : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
    %1 = mfuse.fused %0 {fusion_type = "dvm"} : (tensor<2x3x5xf32>) -> tensor<2x3x5xf32> {
    ^bb0(%arg2: tensor<2x3x5xf32>):
      %2 = mfuse.add %arg2, %arg2 : (tensor<2x3x5xf32>, tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
      %3 = mfuse.mul %2, %arg2 : (tensor<2x3x5xf32>, tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
      %4 = mfuse.exp %3 : (tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
      %5 = mfuse.abs %4 : (tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
      %6 = mfuse.log %5 : (tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
      mfuse.yield %6 : tensor<2x3x5xf32>
    }
    return %1 : tensor<2x3x5xf32>
  }

  // Test cast operation with complex dependencies
  // CHECK-LABEL: func @test_cast_operation
  // CHECK-SAME: %arg0: tensor<4x4xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0
  // CHECK: mfuse.cast
  // CHECK: mfuse.cast
  // CHECK: mfuse.add
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @test_cast_operation(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%arg1: tensor<4x4xf32>):
      %1 = mfuse.cast %arg1 {dtype = f16} : (tensor<4x4xf32>) -> tensor<4x4xf16>
      %2 = mfuse.cast %1 {dtype = f32} : (tensor<4x4xf16>) -> tensor<4x4xf32>
      %3 = mfuse.add %arg1, %2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %3 : tensor<4x4xf32>
    }
    return %0 : tensor<4x4xf32>
  }

  // Test real_div operation with complex dependencies
  // CHECK-LABEL: func @test_real_div_operation
  // CHECK-SAME: %arg0: tensor<4x4xf32>
  // CHECK-SAME: %arg1: tensor<4x4xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
  // CHECK: mfuse.add
  // CHECK: mfuse.sub
  // CHECK: mfuse.real_div
  // CHECK: mfuse.abs
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @test_real_div_operation(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
      %1 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %2 = mfuse.sub %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %3 = mfuse.real_div %1, %2 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %4 = mfuse.abs %3 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %4 : tensor<4x4xf32>
    }
    return %0 : tensor<4x4xf32>
  }

  // Test mixed operations with complex dependencies
  // CHECK-LABEL: func @test_mixed_operations
  // CHECK-SAME: %arg0: tensor<4x4xf32>
  // CHECK-SAME: %arg1: tensor<4x4xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %arg0, %arg1
  // CHECK: mfuse.add
  // CHECK: mfuse.exp
  // CHECK: mfuse.log
  // CHECK: mfuse.mul
  // CHECK: mfuse.abs
  // CHECK: mfuse.gt
  // CHECK: mfuse.select
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @test_mixed_operations(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
      %1 = mfuse.add %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %2 = mfuse.exp %1 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      %3 = mfuse.log %arg2 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      %4 = mfuse.mul %2, %3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      %5 = mfuse.abs %4 : (tensor<4x4xf32>) -> tensor<4x4xf32>
      %6 = mfuse.gt %5, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
      %7 = mfuse.select %6, %5, %arg3 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %7 : tensor<4x4xf32>
    }
    return %0 : tensor<4x4xf32>
  }
}

