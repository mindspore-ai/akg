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
}
