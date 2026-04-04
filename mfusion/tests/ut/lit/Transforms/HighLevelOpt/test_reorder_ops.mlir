// RUN: mfusion-opt %s --mfuse-reorder-ops | FileCheck %s

module {
  // CHECK-LABEL: func @cast_down_after_reshape
  // Test: CastDown (fp32 -> fp16) after Reshape
  // Pattern: Reshape(fp32) -CastDown-> Cast(fp16)
  // Transform to: CastDown(fp32->fp16) -> Reshape(fp16)
  func.func @cast_down_after_reshape(%arg0: tensor<8x2xf32>) -> tensor<16xf16> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} :
        (tensor<8x2xf32>) -> tensor<16xf16> {
      ^bb0(%x0: tensor<8x2xf32>):
        %1 = mfuse.reshape %x0 : (tensor<8x2xf32>) -> tensor<16xf32>
        %2 = mfuse.cast %1 : (tensor<16xf32>) -> tensor<16xf16>
        // After reordering: cast first (fp32->fp16), then reshape
        // CHECK: %[[CAST:.*]] = mfuse.cast
        // CHECK: mfuse.reshape %[[CAST]]
        mfuse.yield %2 : tensor<16xf16>
    }
    return %0 : tensor<16xf16>
  }

  // CHECK-LABEL: func @cast_down_after_permute
  // Test: CastDown (fp32 -> fp16) after Permute
  // Pattern: Permute(fp32) -CastDown-> Cast(fp16)
  // Transform to: CastDown(fp32->fp16) -> Permute(fp16)
  func.func @cast_down_after_permute(%arg0: tensor<2x4x8xf32>) -> tensor<2x8x4xf16> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} :
        (tensor<2x4x8xf32>) -> tensor<2x8x4xf16> {
      ^bb0(%x0: tensor<2x4x8xf32>):
        %1 = mfuse.permute %x0, [0, 2, 1] : (tensor<2x4x8xf32>) -> tensor<2x8x4xf32>
        %2 = mfuse.cast %1 : (tensor<2x8x4xf32>) -> tensor<2x8x4xf16>
        // After reordering: cast first (fp32->fp16), then permute
        // CHECK: %[[CAST:.*]] = mfuse.cast
        // CHECK: mfuse.permute %[[CAST]]
        mfuse.yield %2 : tensor<2x8x4xf16>
    }
    return %0 : tensor<2x8x4xf16>
  }

  // CHECK-LABEL: func @cast_down_after_relu
  // Test: CastDown (fp32 -> fp16) after Relu (unary element-wise)
  // Pattern: Relu(fp32) -CastDown-> Cast(fp16)
  // Transform to: CastDown(fp32->fp16) -> Relu(fp16)
  func.func @cast_down_after_relu(%arg0: tensor<4x8xf32>) -> tensor<4x8xf16> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4x8xf32>) -> tensor<4x8xf16> {
      ^bb0(%x0: tensor<4x8xf32>):
        %1 = mfuse.relu %x0 : (tensor<4x8xf32>) -> tensor<4x8xf32>
        %2 = mfuse.cast %1 : (tensor<4x8xf32>) -> tensor<4x8xf16>
        // After reordering: cast first (fp32->fp16), then relu
        // CHECK: %[[CAST:.*]] = mfuse.cast
        // CHECK: mfuse.relu %[[CAST]]
        mfuse.yield %2 : tensor<4x8xf16>
    }
    return %0 : tensor<4x8xf16>
  }

  // CHECK-LABEL: func @cast_down_after_broadcast_to
  // Test: CastDown (fp32 -> fp16) after BroadcastTo
  // Pattern: BroadcastTo(fp32) -CastDown-> Cast(fp16)
  // Transform to: CastDown(fp32->fp16) -> BroadcastTo(fp16)
  func.func @cast_down_after_broadcast_to(%arg0: tensor<1x1xf32>) -> tensor<4x4xf16> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} :
        (tensor<1x1xf32>) -> tensor<4x4xf16> {
      ^bb0(%x0: tensor<1x1xf32>):
        %1 = mfuse.broadcast_to %x0 : (tensor<1x1xf32>) -> tensor<4x4xf32>
        %2 = mfuse.cast %1 : (tensor<4x4xf32>) -> tensor<4x4xf16>
        // After reordering: cast first (fp32->fp16), then broadcast_to
        // CHECK: %[[CAST:.*]] = mfuse.cast
        // CHECK: mfuse.broadcast_to %[[CAST]]
        mfuse.yield %2 : tensor<4x4xf16>
    }
    return %0 : tensor<4x4xf16>
  }

  // CHECK-LABEL: func @cast_up_before_reshape
  // Test: CastUp (fp16 -> fp32) before Reshape
  // Pattern: CastUp(fp16->fp32) -Reshape-> Reshape(fp32)
  // Transform to: Reshape(fp16) -CastUp-> Cast(fp32)
  func.func @cast_up_before_reshape(%arg0: tensor<8x2xf16>) -> tensor<16xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<8x2xf16>) -> tensor<16xf32> {
      ^bb0(%x0: tensor<8x2xf16>):
        %1 = mfuse.cast %x0 : (tensor<8x2xf16>) -> tensor<8x2xf32>
        %2 = mfuse.reshape %1 : (tensor<8x2xf32>) -> tensor<16xf32>
        // After reordering: reshape first (fp16), then cast (fp16->fp32)
        // CHECK: mfuse.reshape
        // CHECK: mfuse.cast
        mfuse.yield %2 : tensor<16xf32>
    }
    return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func @cast_up_before_permute
  // Test: CastUp (fp16 -> fp32) before Permute
  // Pattern: CastUp(fp16->fp32) -Permute-> Permute(fp32)
  // Transform to: Permute(fp16) -CastUp-> Cast(fp32)
  func.func @cast_up_before_permute(%arg0: tensor<2x4x8xf16>) -> tensor<2x8x4xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<2x4x8xf16>) -> tensor<2x8x4xf32> {
      ^bb0(%x0: tensor<2x4x8xf16>):
        %1 = mfuse.cast %x0 : (tensor<2x4x8xf16>) -> tensor<2x4x8xf32>
        %2 = mfuse.permute %1, [0, 2, 1] : (tensor<2x4x8xf32>) -> tensor<2x8x4xf32>
        // After reordering: permute first (fp16), then cast (fp16->fp32)
        // CHECK: mfuse.permute
        // CHECK: mfuse.cast
        mfuse.yield %2 : tensor<2x8x4xf32>
    }
    return %0 : tensor<2x8x4xf32>
  }

  // CHECK-LABEL: func @cast_up_before_relu
  // Test: CastUp (fp16 -> fp32) before Relu
  // Pattern: CastUp(fp16->fp32) -Relu-> Relu(fp32)
  // Transform to: Relu(fp16) -CastUp-> Cast(fp32)
  func.func @cast_up_before_relu(%arg0: tensor<4x8xf16>) -> tensor<4x8xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4x8xf16>) -> tensor<4x8xf32> {
      ^bb0(%x0: tensor<4x8xf16>):
        %1 = mfuse.cast %x0 : (tensor<4x8xf16>) -> tensor<4x8xf32>
        %2 = mfuse.relu %1 : (tensor<4x8xf32>) -> tensor<4x8xf32>
        // After reordering: relu first (fp16), then cast (fp16->fp32)
        // CHECK: mfuse.relu
        // CHECK: mfuse.cast
        mfuse.yield %2 : tensor<4x8xf32>
    }
    return %0 : tensor<4x8xf32>
  }

  // CHECK-LABEL: func @cast_down_after_maximum
  // Test: CastDown (fp32 -> fp16) after Maximum (binary op)
  // Pattern: Maximum(fp32, fp32) -CastDown-> Cast(fp16)
  // Transform to: CastDown -> Maximum(fp16, fp16)
  func.func @cast_down_after_maximum(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf16> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf16> {
      ^bb0(%x0: tensor<4x8xf32>, %x1: tensor<4x8xf32>):
        %1 = mfuse.maximum %x0, %x1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
        %2 = mfuse.cast %1 : (tensor<4x8xf32>) -> tensor<4x8xf16>
        // After reordering: cast both inputs (fp32->fp16), then maximum
        // CHECK: %[[CAST0:.*]] = mfuse.cast
        // CHECK: %[[CAST1:.*]] = mfuse.cast
        // CHECK: mfuse.maximum %[[CAST0]], %[[CAST1]]
        mfuse.yield %2 : tensor<4x8xf16>
    }
    return %0 : tensor<4x8xf16>
  }

  // CHECK-LABEL: func @cast_up_before_maximum
  // Test: CastUp (fp16 -> fp32) before Maximum
  // Pattern: CastUp(fp16->fp32), CastUp(fp16->fp32) -Maximum-> Maximum(fp32)
  // Transform to: Maximum(fp16, fp16) -CastUp-> Cast(fp32)
  func.func @cast_up_before_maximum(%arg0: tensor<4x8xf16>, %arg1: tensor<4x8xf16>) -> tensor<4x8xf32> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<4x8xf32> {
      ^bb0(%x0: tensor<4x8xf16>, %x1: tensor<4x8xf16>):
        %1 = mfuse.cast %x0 : (tensor<4x8xf16>) -> tensor<4x8xf32>
        %2 = mfuse.cast %x1 : (tensor<4x8xf16>) -> tensor<4x8xf32>
        %3 = mfuse.maximum %1, %2 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
        // After reordering: maximum first (fp16), then cast (fp16->fp32)
        // CHECK: mfuse.maximum
        // CHECK: mfuse.cast
        mfuse.yield %3 : tensor<4x8xf32>
    }
    return %0 : tensor<4x8xf32>
  }

  // CHECK-LABEL: func @cast_down_after_select
  // Test: CastDown (fp32 -> fp16) after Select
  // Select has condition at index 0, data inputs at index 1, 2
  // Pattern: Select(cond, fp32, fp32) -CastDown-> Cast(fp16)
  // Transform to: CastDown -> Select(cond, fp16, fp16)
  func.func @cast_down_after_select(%arg0: tensor<4x8xi1>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf16> {
    %0 = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"} : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf16> {
      ^bb0(%x0: tensor<4x8xi1>, %x1: tensor<4x8xf32>, %x2: tensor<4x8xf32>):
        %1 = mfuse.select %x0, %x1, %x2 : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
        %2 = mfuse.cast %1 : (tensor<4x8xf32>) -> tensor<4x8xf16>
        // After reordering: cast both data inputs (fp32->fp16), then select
        // CHECK: %[[CAST1:.*]] = mfuse.cast
        // CHECK: %[[CAST2:.*]] = mfuse.cast
        // CHECK: mfuse.select %{{.*}}, %[[CAST1]], %[[CAST2]]
        mfuse.yield %2 : tensor<4x8xf16>
    }
    return %0 : tensor<4x8xf16>
  }

  // CHECK-LABEL: func @cast_up_before_select
  // Test: CastUp (fp16 -> fp32) before Select
  // Pattern: CastUp(fp16->fp32), CastUp(fp16->fp32) -Select-> Select(fp32)
  // Transform to: Select(cond, fp16, fp16) -CastUp-> Cast(fp32)
  func.func @cast_up_before_select(%arg0: tensor<4x8xi1>, %arg1: tensor<4x8xf16>, %arg2: tensor<4x8xf16>) -> tensor<4x8xf32> {
    %0 = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"} : (tensor<4x8xi1>, tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<4x8xf32> {
      ^bb0(%x0: tensor<4x8xi1>, %x1: tensor<4x8xf16>, %x2: tensor<4x8xf16>):
        %1 = mfuse.cast %x1 : (tensor<4x8xf16>) -> tensor<4x8xf32>
        %2 = mfuse.cast %x2 : (tensor<4x8xf16>) -> tensor<4x8xf32>
        %3 = mfuse.select %x0, %1, %2 : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
        // After reordering: select first (fp16), then cast (fp16->fp32)
        // CHECK: mfuse.select
        // CHECK: mfuse.cast
        mfuse.yield %3 : tensor<4x8xf32>
    }
    return %0 : tensor<4x8xf32>
  }

  // CHECK-LABEL: func @no_reorder_reshape_already_correct
  // Test: Should NOT reorder when reshape output type matches cast input type
  // This is a negative test case
  func.func @no_reorder_reshape_already_correct(%arg0: tensor<8x2xf16>) -> tensor<16xf32> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<8x2xf16>) -> tensor<16xf32> {
      ^bb0(%x0: tensor<8x2xf16>):
        %1 = mfuse.reshape %x0 : (tensor<8x2xf16>) -> tensor<16xf16>
        %2 = mfuse.cast %1 : (tensor<16xf16>) -> tensor<16xf32>
        // When reshape output is already same type as cast input, should not reorder
        // CHECK: mfuse.reshape
        // CHECK: mfuse.cast
        mfuse.yield %2 : tensor<16xf32>
    }
    return %0 : tensor<16xf32>
  }

  // CHECK-LABEL: func @cast_up_before_minimum
  // Test: CastUp (fp16 -> fp32) before Minimum (binary op)
  func.func @cast_up_before_minimum(%arg0: tensor<4x8xf16>, %arg1: tensor<4x8xf16>) -> tensor<4x8xf32> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x8xf16>, tensor<4x8xf16>) -> tensor<4x8xf32> {
      ^bb0(%x0: tensor<4x8xf16>, %x1: tensor<4x8xf16>):
        %1 = mfuse.cast %x0 : (tensor<4x8xf16>) -> tensor<4x8xf32>
        %2 = mfuse.cast %x1 : (tensor<4x8xf16>) -> tensor<4x8xf32>
        %3 = mfuse.minimum %1, %2 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
        // CHECK: mfuse.minimum
        // CHECK: mfuse.cast
        mfuse.yield %3 : tensor<4x8xf32>
    }
    return %0 : tensor<4x8xf32>
  }
}
