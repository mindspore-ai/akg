// RUN: mfusion-opt %s --mfuse-raise-reduction-precision | FileCheck %s

module {
  // CHECK-LABEL: func @raise_reduce_sum_precision
  func.func @raise_reduce_sum_precision(%arg0: tensor<4x8xf16>) -> tensor<4xf16> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} :
        (tensor<4x8xf16>) -> tensor<4xf16> {
      ^bb0(%x0: tensor<4x8xf16>):
        %1 = mfuse.reduce_sum %x0 {dimensions = [1], keepdim = false} : (tensor<4x8xf16>) -> tensor<4xf16>
        // After precision raising: input is cast to float32, sum is performed, then result is cast back to float16
        // CHECK: mfuse.cast %{{.*}} : (tensor<4x8xf16>) -> tensor<4x8xf32>
        // CHECK: mfuse.reduce_sum {{.*}} : (tensor<4x8xf32>) -> tensor<4xf32>
        // CHECK: mfuse.cast {{.*}} : (tensor<4xf32>) -> tensor<4xf16>
        mfuse.yield %1 : tensor<4xf16>
    }
    return %0 : tensor<4xf16>
  }

  // CHECK-LABEL: func @raise_reduce_sum_keep_dims
  func.func @raise_reduce_sum_keep_dims(%arg0: tensor<2x4x8xf16>) -> tensor<2x1x8xf16> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} :
        (tensor<2x4x8xf16>) -> tensor<2x1x8xf16> {
      ^bb0(%x0: tensor<2x4x8xf16>):
        %1 = mfuse.reduce_sum %x0 {dimensions = [1], keepdim = true} : (tensor<2x4x8xf16>) -> tensor<2x1x8xf16>
        // After precision raising with keep_dims=True
        // CHECK: mfuse.cast %{{.*}} : (tensor<2x4x8xf16>) -> tensor<2x4x8xf32>
        // CHECK: mfuse.reduce_sum {{.*}} : (tensor<2x4x8xf32>) -> tensor<2x1x8xf32>
        // CHECK: mfuse.cast {{.*}} : (tensor<2x1x8xf32>) -> tensor<2x1x8xf16>
        mfuse.yield %1 : tensor<2x1x8xf16>
    }
    return %0 : tensor<2x1x8xf16>
  }

  // CHECK-LABEL: func @float32_no_raise
  func.func @float32_no_raise(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} :
        (tensor<4x8xf32>) -> tensor<4xf32> {
      ^bb0(%x0: tensor<4x8xf32>):
        %1 = mfuse.reduce_sum %x0 {dimensions = [1], keepdim = false} : (tensor<4x8xf32>) -> tensor<4xf32>
        // No precision raising needed for float32 inputs
        // CHECK-NOT: mfuse.cast
        // CHECK: mfuse.reduce_sum
        mfuse.yield %1 : tensor<4xf32>
    }
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: func @multi_axis_reduce
  func.func @multi_axis_reduce(%arg0: tensor<2x4x8xf16>) -> tensor<1x1x1xf16> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} :
        (tensor<2x4x8xf16>) -> tensor<1x1x1xf16> {
      ^bb0(%x0: tensor<2x4x8xf16>):
        %1 = mfuse.reduce_sum %x0 {dimensions = [0, 1, 2], keepdim = true} : (tensor<2x4x8xf16>) -> tensor<1x1x1xf16>
        // After precision raising for multi-axis reduction
        // CHECK: mfuse.cast %{{.*}} : (tensor<2x4x8xf16>) -> tensor<2x4x8xf32>
        // CHECK: mfuse.reduce_sum {{.*}} : (tensor<2x4x8xf32>) -> tensor<1x1x1xf32>
        // CHECK: mfuse.cast {{.*}} : (tensor<1x1x1xf32>) -> tensor<1x1x1xf16>
        mfuse.yield %1 : tensor<1x1x1xf16>
    }
    return %0 : tensor<1x1x1xf16>
  }
}
