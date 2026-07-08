// RUN: mfusion-opt %s --mfuse-dvm-cluster | FileCheck %s

module {
  // CHECK-LABEL: func @test_reduce_sum_skip_large_semiprime_non_reduce_axis
  // CHECK-SAME: %[[ARG0:.*]]: tensor<111547x4xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[ARG0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<111547x4xf32>) -> tensor<111547x4xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<111547x4xf32>):
  // CHECK: %[[ADD:.*]] = mfuse.add %[[IN0]], %[[IN0]] : (tensor<111547x4xf32>, tensor<111547x4xf32>) -> tensor<111547x4xf32>
  // CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ADD]] : (tensor<111547x4xf32>, tensor<111547x4xf32>) -> tensor<111547x4xf32>
  // CHECK: mfuse.yield %[[MUL]] : tensor<111547x4xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[FUSED]] {dimensions = [1], keepdim = false} : (tensor<111547x4xf32>) -> tensor<111547xf32>
  // CHECK: return %[[REDUCE]] : tensor<111547xf32>
  func.func @test_reduce_sum_skip_large_semiprime_non_reduce_axis(%arg0: tensor<111547x4xf32>)
      -> tensor<111547xf32> {
    %0 = mfuse.add %arg0, %arg0 : (tensor<111547x4xf32>, tensor<111547x4xf32>) -> tensor<111547x4xf32>
    %1 = mfuse.mul %0, %0 : (tensor<111547x4xf32>, tensor<111547x4xf32>) -> tensor<111547x4xf32>
    %2 = mfuse.reduce_sum %1 {dimensions = [1], keepdim = false} : (tensor<111547x4xf32>) -> tensor<111547xf32>
    return %2 : tensor<111547xf32>
  }

  // CHECK-LABEL: func @test_reduce_sum_skip_shape_4x197951_axis0
  // CHECK-SAME: %[[ARG0:.*]]: tensor<4x197951xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[ARG0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<4x197951xf32>) -> tensor<4x197951xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<4x197951xf32>):
  // CHECK: %[[ADD:.*]] = mfuse.add %[[IN0]], %[[IN0]] : (tensor<4x197951xf32>, tensor<4x197951xf32>) -> tensor<4x197951xf32>
  // CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ADD]] : (tensor<4x197951xf32>, tensor<4x197951xf32>) -> tensor<4x197951xf32>
  // CHECK: mfuse.yield %[[MUL]] : tensor<4x197951xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[FUSED]] {dimensions = [0], keepdim = false} : (tensor<4x197951xf32>) -> tensor<197951xf32>
  // CHECK: return %[[REDUCE]] : tensor<197951xf32>
  func.func @test_reduce_sum_skip_shape_4x197951_axis0(%arg0: tensor<4x197951xf32>)
      -> tensor<197951xf32> {
    %0 = mfuse.add %arg0, %arg0 : (tensor<4x197951xf32>, tensor<4x197951xf32>) -> tensor<4x197951xf32>
    %1 = mfuse.mul %0, %0 : (tensor<4x197951xf32>, tensor<4x197951xf32>) -> tensor<4x197951xf32>
    %2 = mfuse.reduce_sum %1 {dimensions = [0], keepdim = false} : (tensor<4x197951xf32>) -> tensor<197951xf32>
    return %2 : tensor<197951xf32>
  }

  // CHECK-LABEL: func @test_reduce_sum_keep_large_non_semiprime_non_reduce_axis
  // CHECK-SAME: %[[ARG0:.*]]: tensor<100100x4xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[ARG0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<100100x4xf32>) -> tensor<100100xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<100100x4xf32>):
  // CHECK: %[[ADD:.*]] = mfuse.add %[[IN0]], %[[IN0]] : (tensor<100100x4xf32>, tensor<100100x4xf32>) -> tensor<100100x4xf32>
  // CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ADD]] : (tensor<100100x4xf32>, tensor<100100x4xf32>) -> tensor<100100x4xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[MUL]] {dimensions = [1], keepdim = false} : (tensor<100100x4xf32>) -> tensor<100100xf32>
  // CHECK: mfuse.yield %[[REDUCE]] : tensor<100100xf32>
  // CHECK: return %[[FUSED]] : tensor<100100xf32>
  func.func @test_reduce_sum_keep_large_non_semiprime_non_reduce_axis(%arg0: tensor<100100x4xf32>)
      -> tensor<100100xf32> {
    %0 = mfuse.add %arg0, %arg0 : (tensor<100100x4xf32>, tensor<100100x4xf32>) -> tensor<100100x4xf32>
    %1 = mfuse.mul %0, %0 : (tensor<100100x4xf32>, tensor<100100x4xf32>) -> tensor<100100x4xf32>
    %2 = mfuse.reduce_sum %1 {dimensions = [1], keepdim = false} : (tensor<100100x4xf32>) -> tensor<100100xf32>
    return %2 : tensor<100100xf32>
  }

  // CHECK-LABEL: func @test_reduce_sum_skip_rank3_large_semiprime_non_reduce_axis
  // CHECK-SAME: %[[ARG0:.*]]: tensor<2x111547x4xf32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[ARG0]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<2x111547x4xf32>) -> tensor<2x111547x4xf32>
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<2x111547x4xf32>):
  // CHECK: %[[ADD:.*]] = mfuse.add %[[IN0]], %[[IN0]] : (tensor<2x111547x4xf32>, tensor<2x111547x4xf32>) -> tensor<2x111547x4xf32>
  // CHECK: %[[MUL:.*]] = mfuse.mul %[[ADD]], %[[ADD]] : (tensor<2x111547x4xf32>, tensor<2x111547x4xf32>) -> tensor<2x111547x4xf32>
  // CHECK: mfuse.yield %[[MUL]] : tensor<2x111547x4xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[FUSED]] {dimensions = [0], keepdim = false} : (tensor<2x111547x4xf32>) -> tensor<111547x4xf32>
  // CHECK: return %[[REDUCE]] : tensor<111547x4xf32>
  func.func @test_reduce_sum_skip_rank3_large_semiprime_non_reduce_axis(%arg0: tensor<2x111547x4xf32>)
      -> tensor<111547x4xf32> {
    %0 = mfuse.add %arg0, %arg0 : (tensor<2x111547x4xf32>, tensor<2x111547x4xf32>) -> tensor<2x111547x4xf32>
    %1 = mfuse.mul %0, %0 : (tensor<2x111547x4xf32>, tensor<2x111547x4xf32>) -> tensor<2x111547x4xf32>
    %2 = mfuse.reduce_sum %1 {dimensions = [0], keepdim = false} : (tensor<2x111547x4xf32>) -> tensor<111547x4xf32>
    return %2 : tensor<111547x4xf32>
  }

  // External rank0 input is now supported here, so the compare/select tail can
  // be clustered together with the reshape+reduce_sum consumer chain.
  // CHECK-LABEL: func @test_reduce_sum_after_non_matmul_reshape_cluster
  // CHECK-SAME: %[[ARG0:.*]]: tensor<16x64x128x128xf32>, %[[ARG1:.*]]: tensor<16x64x128x128xf32>, %[[ARG2:.*]]: tensor<f32>
  // CHECK: %[[FUSED:.*]] = mfuse.fused %[[ARG0]], %[[ARG2]], %[[ARG1]] {fusion_type = "dvm"}
  // CHECK-SAME: : (tensor<16x64x128x128xf32>, tensor<f32>, tensor<16x64x128x128xf32>) -> tensor<1024xf32>
  // CHECK: ^bb0(%[[VAL0:.*]]: tensor<16x64x128x128xf32>, %[[VAL1:.*]]: tensor<f32>, %[[VAL2:.*]]: tensor<16x64x128x128xf32>):
  // CHECK: %[[ZERO:.*]] = mfuse.constant dense<0> : tensor<i64, {is_scalar = ""}>
  // CHECK: %[[LE:.*]] = mfuse.le %[[VAL0]], %[[ZERO]] : (tensor<16x64x128x128xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<16x64x128x128xi1>
  // CHECK: %[[SELECT:.*]] = mfuse.select %[[LE]], %[[VAL1]], %[[VAL2]] : (tensor<16x64x128x128xi1>, tensor<f32>, tensor<16x64x128x128xf32>) -> tensor<16x64x128x128xf32>
  // CHECK: %[[RESHAPE:.*]] = mfuse.reshape %[[SELECT]] : (tensor<16x64x128x128xf32>) -> tensor<1x1024x128x128xf32>
  // CHECK: %[[REDUCE:.*]] = mfuse.reduce_sum %[[RESHAPE]] {dimensions = [0, 2, 3], keepdim = false} : (tensor<1x1024x128x128xf32>) -> tensor<1024xf32>
  // CHECK: mfuse.yield %[[REDUCE]] : tensor<1024xf32>
  // CHECK: return %[[FUSED]] : tensor<1024xf32>
  func.func @test_reduce_sum_after_non_matmul_reshape_cluster(
      %arg0: tensor<16x64x128x128xf32>,
      %arg1: tensor<16x64x128x128xf32>,
      %arg2: tensor<f32>) -> tensor<1024xf32> {
    %0 = mfuse.constant dense<0> : tensor<i64, {is_scalar = ""}>
    %1 = mfuse.le %arg0, %0 : (tensor<16x64x128x128xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<16x64x128x128xi1>
    %2 = mfuse.select %1, %arg2, %arg1 : (tensor<16x64x128x128xi1>, tensor<f32>, tensor<16x64x128x128xf32>) -> tensor<16x64x128x128xf32>
    %3 = mfuse.reshape %2 : (tensor<16x64x128x128xf32>) -> tensor<1x1024x128x128xf32>
    %4 = mfuse.reduce_sum %3 {dimensions = [0, 2, 3], keepdim = false} : (tensor<1x1024x128x128xf32>) -> tensor<1024xf32>
    return %4 : tensor<1024xf32>
  }
}
