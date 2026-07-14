// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
  // CHECK-LABEL: func.func @i64_add_sub_cast
  // CHECK: %[[FUSED:.*]] = mfuse.fused
  // CHECK: mfuse.add
  // CHECK: mfuse.sub
  // CHECK: mfuse.cast
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @i64_add_sub_cast(%arg0: tensor<4xi64>, %arg1: tensor<4xi64>) -> tensor<4xi32> {
    %0 = mfuse.add %arg0, %arg1 : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    %1 = mfuse.sub %0, %arg1 : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    %2 = mfuse.cast %1 : (tensor<4xi64>) -> tensor<4xi32>
    return %2 : tensor<4xi32>
  }

  // CHECK-LABEL: func.func @i64_cast_result
  // CHECK: %[[FUSED:.*]] = mfuse.fused
  // CHECK: mfuse.cast
  // CHECK: mfuse.add
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @i64_cast_result(%arg0: tensor<4xi32>, %arg1: tensor<4xi64>) -> tensor<4xi64> {
    %0 = mfuse.cast %arg0 : (tensor<4xi32>) -> tensor<4xi64>
    %1 = mfuse.add %0, %arg1 : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    return %1 : tensor<4xi64>
  }

  // CHECK-LABEL: func.func @i64_select_cast
  // CHECK: %[[FUSED:.*]] = mfuse.fused
  // CHECK: mfuse.select
  // CHECK: mfuse.cast
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @i64_select_cast(
      %cond: tensor<4xi1>, %arg0: tensor<4xi64>, %arg1: tensor<4xi64>) -> tensor<4xi32> {
    %0 = mfuse.select %cond, %arg0, %arg1 : (tensor<4xi1>, tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    %1 = mfuse.cast %0 : (tensor<4xi64>) -> tensor<4xi32>
    return %1 : tensor<4xi32>
  }

  // CHECK-LABEL: func.func @i64_mul_not_clustered
  // CHECK-NOT: mfuse.fused
  // CHECK: mfuse.mul
  // CHECK-NOT: mfuse.fused
  // CHECK: return
  func.func @i64_mul_not_clustered(%arg0: tensor<4xi64>, %arg1: tensor<4xi64>) -> tensor<4xi64> {
    %0 = mfuse.mul %arg0, %arg1 : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    return %0 : tensor<4xi64>
  }

  // CHECK-LABEL: func.func @i64_scalar_add_sub
  // CHECK: %[[FUSED:.*]] = mfuse.fused
  // CHECK: mfuse.constant dense<2> : tensor<i64, {is_scalar = ""}>
  // CHECK: mfuse.add
  // CHECK: mfuse.sub
  // CHECK: mfuse.yield
  // CHECK: return %[[FUSED]]
  func.func @i64_scalar_add_sub(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %c = mfuse.constant dense<2> : tensor<i64, {is_scalar = ""}>
    %0 = mfuse.add %arg0, %c : (tensor<4xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<4xf32>
    %1 = mfuse.sub %c, %0 : (tensor<i64, {is_scalar = ""}>, tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
