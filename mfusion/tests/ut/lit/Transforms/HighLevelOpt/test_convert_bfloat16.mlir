// RUN: mfusion-opt %s --mfuse-convert-bfloat16 | FileCheck %s

module {
  // Test 1: Add + Less - bf16 inputs converted to f32
  //   sub_graph(p0: bf16, p1: bf16, p2: bf16) {
  //     %0(bf16) = Add(p0, p1)
  //     %1(bool) = Less(%0, p2)
  //     return %1
  //   }
  //   ---------->
  //   sub_graph(p0: bf16, p1: bf16, p2: bf16) {
  //     %0(fp32) = Cast(p0, fp32)
  //     %1(fp32) = Cast(p1, fp32)
  //     %2(fp32) = Add(%0, %1)
  //     %3(bf16) = Cast(%2, bf16)   // Add output cast back for Less
  //     %4(fp32) = Cast(%3, fp32)   // Less needs f32 input
  //     %5(fp32) = Cast(p2, fp32)
  //     %6(bool) = Less(%4, %5)
  //     return %6
  //   }
  // CHECK-LABEL: func @test_add_less
  func.func @test_add_less(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<32x32xbf16>) -> tensor<32x32xi1> {
    %0 = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"} : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xi1> {
      ^bb0(%x0: tensor<32x32xbf16>, %x1: tensor<32x32xbf16>, %x2: tensor<32x32xbf16>):
        %1 = mfuse.add %x0, %x1 : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        %2 = mfuse.lt %1, %x2 : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xi1>
        mfuse.yield %2 : tensor<32x32xi1>
    }
    return %0 : tensor<32x32xi1>
  }
  // CHECK: mfuse.cast %{{.*}}
  // CHECK: mfuse.cast %{{.*}}
  // CHECK: mfuse.add
  // CHECK: mfuse.cast {{.*}} : (tensor<32x32xf32>) -> tensor<32x32xbf16>
  // CHECK: mfuse.cast {{.*}} : (tensor<32x32xbf16>) -> tensor<32x32xf32>
  // CHECK: mfuse.cast %{{.*}}
  // CHECK: mfuse.lt

  // Test 2: Add + Abs + MatMul - MatMul keeps bf16 for specific inputs
  //   sub_graph(p0: bf16, p1: bf16, p2: bf16) {
  //     %0(bf16) = Add(p0, p1)
  //     %1(bf16) = Abs(p2)
  //     %2(bf16) = MatMul(%0, %1)
  //     return %2
  //   }
  //   ---------->
  //   sub_graph(p0: bf16, p1: bf16, p2: bf16) {
  //     %0(fp32) = Cast(p0, fp32)
  //     %1(fp32) = Cast(p1, fp32)
  //     %2(fp32) = Add(%0, %1)
  //     %3(bf16) = Cast(%2, bf16)  // Add output cast back for MatMul
  //     %4(fp32) = Cast(p2, fp32)
  //     %5(fp32) = Abs(%4)
  //     %6(bf16) = Cast(%5, bf16)  // Abs output cast back for MatMul
  //     %7(bf16) = MatMul(%3, %6)  // MatMul (NeedKeepBF16) keeps bf16
  //     return %7
  //   }
  // CHECK-LABEL: func @test_add_abs_matmul
  func.func @test_add_abs_matmul(%arg0: tensor<1024x1024xbf16>, %arg1: tensor<1024x1024xbf16>, %arg2: tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16> {
    %0 = mfuse.fused %arg0, %arg1, %arg2 {fusion_type = "dvm"} : (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16> {
      ^bb0(%x0: tensor<1024x1024xbf16>, %x1: tensor<1024x1024xbf16>, %x2: tensor<1024x1024xbf16>):
        %1 = mfuse.add %x0, %x1 : (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
        %2 = mfuse.abs %x2 : (tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
        %3 = mfuse.matmul %1, %2 : (tensor<1024x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
        mfuse.yield %3 : tensor<1024x1024xbf16>
    }
    return %0 : tensor<1024x1024xbf16>
  }
  // CHECK: mfuse.cast %{{.*}}
  // CHECK: mfuse.cast %{{.*}}
  // CHECK: mfuse.add
  // CHECK: mfuse.cast {{.*}} : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
  // CHECK: mfuse.cast %{{.*}}
  // CHECK: mfuse.abs
  // CHECK: mfuse.cast {{.*}} : (tensor<1024x1024xf32>) -> tensor<1024x1024xbf16>
  // CHECK: mfuse.matmul

  // Test 3: Add + Cast - multiple output test
  //   sub_graph(p0: bf16, p1: bf16) {
  //     %0(bf16) = Add(p0, p1)
  //     %1(fp32) = Cast(%0, fp32)
  //     return %0, %1
  //   }
  //   ---------->
  //   sub_graph(p0: bf16, p1: bf16) {
  //     %0(fp32) = Cast(p0, fp32)
  //     %1(fp32) = Cast(p1, fp32)
  //     %2(fp32) = Add(%0, %1)
  //     %3(bf16) = Cast(%2, bf16)  // Add output cast back for return
  //     %4(fp32) = Cast(%3, fp32)  // Cast keeps bf16->fp32
  //     return %3, %4
  //   }
  // CHECK-LABEL: func @test_add_cast_multi_return
  func.func @test_add_cast_multi_return(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> (tensor<32x32xbf16>, tensor<32x32xf32>) {
    %0:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> (tensor<32x32xbf16>, tensor<32x32xf32>) {
      ^bb0(%x0: tensor<32x32xbf16>, %x1: tensor<32x32xbf16>):
        %1 = mfuse.add %x0, %x1 : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        %2 = mfuse.cast %1 : (tensor<32x32xbf16>) -> tensor<32x32xf32>
        mfuse.yield %1, %2 : tensor<32x32xbf16>, tensor<32x32xf32>
    }
    return %0#0, %0#1 : tensor<32x32xbf16>, tensor<32x32xf32>
  }
  // CHECK: mfuse.cast %{{.*}}
  // CHECK: mfuse.cast %{{.*}}
  // CHECK: mfuse.add
  // CHECK: mfuse.cast {{.*}} : (tensor<32x32xf32>) -> tensor<32x32xbf16>
  // CHECK: mfuse.cast {{.*}} : (tensor<32x32xbf16>) -> tensor<32x32xf32>

  // Test 4: Reshape + Abs + Reshape - Reshape keeps bf16 (CanKeepBF16)
  //   sub_graph(p0: bf16) {
  //     %0(bf16) = Reshape(p0)     // CanKeepBF16 - keeps bf16
  //     %1(bf16) = Abs(%0)         // Abs converts to f32
  //     %2(bf16) = Reshape(%1)     // Reshape with f32 input keeps f32
  //     return %2
  //   }
  //   ---------->
  //   sub_graph(p0: bf16) {
  //     %0(bf16) = Reshape(p0)     // First Reshape keeps bf16
  //     %1(fp32) = Cast(%0, fp32)  // Abs needs f32
  //     %2(fp32) = Abs(%1)
  //     %3(bf16) = Cast(%2, bf16)  // Cast back for output
  //     %4(bf16) = Reshape(%3)     // Second Reshape keeps bf16 (but input is bf16 now)
  //     return %4
  //   }
  // CHECK-LABEL: func @test_reshape_abs_reshape
  func.func @test_reshape_abs_reshape(%arg0: tensor<2x6xbf16>) -> tensor<3x4xbf16> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<2x6xbf16>) -> tensor<3x4xbf16> {
      ^bb0(%x0: tensor<2x6xbf16>):
        %1 = mfuse.reshape %x0 {shape = array<i64: 12>} : (tensor<2x6xbf16>) -> tensor<12xbf16>
        %2 = mfuse.abs %1 : (tensor<12xbf16>) -> tensor<12xbf16>
        %3 = mfuse.reshape %2 {shape = array<i64: 3, 4>} : (tensor<12xbf16>) -> tensor<3x4xbf16>
        mfuse.yield %3 : tensor<3x4xbf16>
    }
    return %0 : tensor<3x4xbf16>
  }
  // First Reshape keeps bf16 (CanKeepBF16)
  // CHECK: mfuse.reshape
  // CHECK-NOT: mfuse.cast
  // Abs converts bf16 to f32
  // CHECK: mfuse.cast
  // CHECK: mfuse.abs
  // Cast back to bf16 for final result
  // CHECK: mfuse.cast
  // CHECK: mfuse.reshape

  // Test 5: Matmul - NeedKeepBF16, skipped completely
  // CHECK-LABEL: func @convert_matmul_bfloat16
  func.func @convert_matmul_bfloat16(%arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16> {
      ^bb0(%x0: tensor<4x4xbf16>, %x1: tensor<4x4xbf16>):
        %1 = mfuse.matmul %x0, %x1 : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
        mfuse.yield %1 : tensor<4x4xbf16>
    }
    return %0 : tensor<4x4xbf16>
  }
  // MatMul is NeedKeepBF16, skipped completely
  // CHECK-NOT: mfuse.cast
  // CHECK: mfuse.matmul
  // CHECK-NOT: mfuse.cast

  // Test 6: Mixed types - no conversion needed
  // CHECK-LABEL: func @mixed_types_no_conversion
  func.func @mixed_types_no_conversion(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
      ^bb0(%x0: tensor<4x4xf32>, %x1: tensor<4x4xf32>):
        %1 = mfuse.add %x0, %x1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
        mfuse.yield %1 : tensor<4x4xf32>
    }
    return %0 : tensor<4x4xf32>
  }
  // CHECK-NOT: mfuse.cast
  // CHECK: mfuse.add

  // Test 7: Partially bf16 inputs - MatMul is NeedKeepBF16, skipped
  // CHECK-LABEL: func @partially_bf16_inputs
  func.func @partially_bf16_inputs(%arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xbf16> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xbf16>, tensor<4x4xf32>) -> tensor<4x4xbf16> {
      ^bb0(%x0: tensor<4x4xbf16>, %x1: tensor<4x4xf32>):
        %1 = mfuse.matmul %x0, %x1 : (tensor<4x4xbf16>, tensor<4x4xf32>) -> tensor<4x4xbf16>
        mfuse.yield %1 : tensor<4x4xbf16>
    }
    return %0 : tensor<4x4xbf16>
  }
  // MatMul is NeedKeepBF16, skipped completely
  // CHECK-NOT: mfuse.cast
  // CHECK: mfuse.matmul
  // CHECK-NOT: mfuse.cast

  // Test 8: BroadcastTo - should keep bf16 (CanKeepBF16)
  // CHECK-LABEL: func @broadcast_to_keep_bf16
  func.func @broadcast_to_keep_bf16(%arg0: tensor<4x1xbf16>) -> tensor<4x4xbf16> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<4x1xbf16>) -> tensor<4x4xbf16> {
      ^bb0(%x0: tensor<4x1xbf16>):
        %1 = mfuse.broadcast_to %x0 {shape = array<i64: 4, 4>} : (tensor<4x1xbf16>) -> tensor<4x4xbf16>
        mfuse.yield %1 : tensor<4x4xbf16>
    }
    return %0 : tensor<4x4xbf16>
  }
  // CHECK-NOT: mfuse.cast
  // CHECK: mfuse.broadcast_to

  // Test 9: BatchMatmul - NeedKeepBF16, skipped completely
  // CHECK-LABEL: func @batch_matmul_keep_bf16
  func.func @batch_matmul_keep_bf16(%arg0: tensor<2x4x4xbf16>, %arg1: tensor<2x4x4xbf16>) -> tensor<2x4x4xbf16> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<2x4x4xbf16>, tensor<2x4x4xbf16>) -> tensor<2x4x4xbf16> {
      ^bb0(%x0: tensor<2x4x4xbf16>, %x1: tensor<2x4x4xbf16>):
        %1 = mfuse.batch_matmul %x0, %x1 : (tensor<2x4x4xbf16>, tensor<2x4x4xbf16>) -> tensor<2x4x4xbf16>
        mfuse.yield %1 : tensor<2x4x4xbf16>
    }
    return %0 : tensor<2x4x4xbf16>
  }
  // BatchMatmul is NeedKeepBF16, skipped completely
  // CHECK-NOT: mfuse.cast
  // CHECK: mfuse.batch_matmul
  // CHECK-NOT: mfuse.cast

  // Test 10: Slice - NeedKeepBF16 input, skipped
  // CHECK-LABEL: func @slice_keep_bf16
  func.func @slice_keep_bf16(%arg0: tensor<8x8xbf16>) -> tensor<4x8xbf16> {
    %0 = mfuse.fused %arg0 {fusion_type = "dvm"} : (tensor<8x8xbf16>) -> tensor<4x8xbf16> {
      ^bb0(%x0: tensor<8x8xbf16>):
        %1 = mfuse.slice %x0 dim 0 start 0 end 4 step 1 : (tensor<8x8xbf16>) -> tensor<4x8xbf16>
        mfuse.yield %1 : tensor<4x8xbf16>
    }
    return %0 : tensor<4x8xbf16>
  }
  // Slice is NeedKeepBF16, skipped completely
  // CHECK-NOT: mfuse.cast
  // CHECK: mfuse.slice
  // CHECK-NOT: mfuse.cast

  // Test 11: Add + Slice - Slice needs bf16 input
  // CHECK-LABEL: func @add_slice_convert
  func.func @add_slice_convert(%arg0: tensor<8x8xbf16>, %arg1: tensor<8x8xbf16>) -> tensor<4x8xbf16> {
    %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<8x8xbf16>, tensor<8x8xbf16>) -> tensor<4x8xbf16> {
      ^bb0(%x0: tensor<8x8xbf16>, %x1: tensor<8x8xbf16>):
        %1 = mfuse.add %x0, %x1 : (tensor<8x8xbf16>, tensor<8x8xbf16>) -> tensor<8x8xbf16>
        %2 = mfuse.slice %1 dim 0 start 0 end 4 step 1 : (tensor<8x8xbf16>) -> tensor<4x8xbf16>
        mfuse.yield %2 : tensor<4x8xbf16>
    }
    return %0 : tensor<4x8xbf16>
  }
  // Add converts to f32, then cast back to bf16 for Slice
  // CHECK: mfuse.cast %{{.*}}
  // CHECK: mfuse.cast %{{.*}}
  // CHECK: mfuse.add
  // CHECK: mfuse.cast {{.*}} : (tensor<8x8xf32>) -> tensor<8x8xbf16>
  // CHECK: mfuse.slice
}
