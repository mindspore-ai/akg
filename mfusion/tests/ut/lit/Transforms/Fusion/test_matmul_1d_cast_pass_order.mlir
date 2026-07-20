// Guard 1D matmul+cast (+bias): Cast must run before Unsqueeze.
// Unsqueeze-first inserts reshape and yields matmul -> reshape -> cast, so
// FuseMatMulCast misses; Cast-first absorbs cast, then Unsqueeze / ReshapeBiasAdd
// can still normalize shape and fuse bias.
//
// Actual shape after Cast->Unsqueeze: reshape(lhs) + reshape(rhs, may be identity),
// matmul/matmul_with_bias keeps the original 1D result type, then output reshape.
//
// RUN: mfusion-opt %s --matmul-optimization | FileCheck %s
// RUN: mfusion-opt %s --mfuse-fusion | FileCheck %s

module {
  // 1D x 2D matmul(f16) + cast(f32): cast absorbed, then unsqueeze wraps f32 matmul.
  // CHECK-LABEL: func @one_d_matmul_cast
  // CHECK-SAME: (%[[A:.*]]: tensor<4xf16>, %[[B:.*]]: tensor<4x8xf16>)
  func.func @one_d_matmul_cast(%arg0: tensor<4xf16>, %arg1: tensor<4x8xf16>) -> tensor<8xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<4xf16>, tensor<4x8xf16>) -> tensor<8xf16>
    %1 = mfuse.cast %0 : (tensor<8xf16>) -> tensor<8xf32>
    // CHECK-NOT: mfuse.cast
    // CHECK: %[[R0:.*]] = mfuse.reshape %[[A]] : (tensor<4xf16>) -> tensor<1x4xf16>
    // CHECK: %[[R1:.*]] = mfuse.reshape %[[B]] : (tensor<4x8xf16>) -> tensor<4x8xf16>
    // CHECK: %[[MM:.*]] = mfuse.matmul %[[R0]], %[[R1]]
    // CHECK-SAME: -> tensor<8xf32>
    // CHECK: %[[R2:.*]] = mfuse.reshape %[[MM]] : (tensor<8xf32>) -> tensor<8xf32>
    // CHECK: return %[[R2]]
    return %1 : tensor<8xf32>
  }

  // 1D x 2D matmul(f16) + cast(f32) + bias: cast first, then unsqueeze + ReshapeBiasAdd.
  // CHECK-LABEL: func @one_d_matmul_cast_bias
  // CHECK-SAME: (%[[A:.*]]: tensor<4xf16>, %[[B:.*]]: tensor<4x8xf16>, %[[BIAS:.*]]: tensor<8xf32>)
  func.func @one_d_matmul_cast_bias(%arg0: tensor<4xf16>, %arg1: tensor<4x8xf16>,
                                      %bias: tensor<8xf32>) -> tensor<8xf32> {
    %0 = mfuse.matmul %arg0, %arg1 : (tensor<4xf16>, tensor<4x8xf16>) -> tensor<8xf16>
    %1 = mfuse.cast %0 : (tensor<8xf16>) -> tensor<8xf32>
    %2 = mfuse.add %1, %bias : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
    // CHECK-NOT: mfuse.cast
    // CHECK-NOT: mfuse.add
    // CHECK: %[[R0:.*]] = mfuse.reshape %[[A]] : (tensor<4xf16>) -> tensor<1x4xf16>
    // CHECK: %[[R1:.*]] = mfuse.reshape %[[B]] : (tensor<4x8xf16>) -> tensor<4x8xf16>
    // CHECK: %[[MM:.*]] = mfuse.matmul_with_bias %[[R0]], %[[R1]], %[[BIAS]]
    // CHECK-SAME: -> tensor<8xf32>
    // CHECK: %[[R2:.*]] = mfuse.reshape %[[MM]] : (tensor<8xf32>) -> tensor<8xf32>
    // CHECK: return %[[R2]]
    return %2 : tensor<8xf32>
  }
}
