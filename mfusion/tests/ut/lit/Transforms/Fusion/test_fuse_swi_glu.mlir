// RUN: mfusion-opt %s --fuse-swi-glu | FileCheck %s

module {
  // CHECK-LABEL: func @test_swiglu_fusion
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<2x4xf32>, %[[DIM:.*]]: tensor<i64>)
  func.func @test_swiglu_fusion(%input: tensor<2x4xf32>, %dim: tensor<i64>) -> tensor<2x2xf32> {
    // SplitWithSize operation: Split input into 2 tensors along dim
    %split_size = arith.constant dense<[2, 2]> : tensor<2xi64>
    %split:2 = muse.aclnn.split_with_size %input, %split_size, %dim : (tensor<2x4xf32>, tensor<2xi64>, tensor<i64>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

    // SiLU operation: SiLU(split#1)
    %silu_res = muse.aclnn.silu %split#1 : (tensor<2x2xf32>) -> tensor<2x2xf32>

    // Mul operation: split#0 * silu_res
    %res = muse.mul %split#0, %silu_res : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

    // After fusion:
    // - muse.aclnn.split_with_size, muse.aclnn.silu and muse.mul should be replaced by muse.aclnn.swi_glu
    // CHECK-NOT: muse.aclnn.split_with_size
    // CHECK-NOT: muse.aclnn.silu
    // CHECK-NOT: muse.mul
    // CHECK: %[[RES:.*]] = muse.aclnn.swi_glu %[[INPUT]], %[[DIM]]
    // CHECK: return %[[RES]]

    return %res : tensor<2x2xf32>
  }

  // CHECK-LABEL: func @test_swiglu_fusion_commutative
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<2x4xf32>, %[[DIM:.*]]: tensor<i64>)
  func.func @test_swiglu_fusion_commutative(%input: tensor<2x4xf32>, %dim: tensor<i64>) -> tensor<2x2xf32> {
    // Test Mul commutativity: Silu(split#1) * split#0 should also fuse
    %split_size = arith.constant dense<[2, 2]> : tensor<2xi64>
    %split:2 = muse.aclnn.split_with_size %input, %split_size, %dim : (tensor<2x4xf32>, tensor<2xi64>, tensor<i64>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

    %silu_res = muse.aclnn.silu %split#1 : (tensor<2x2xf32>) -> tensor<2x2xf32>

    // Mul operation: silu_res * split#0 (swapped order)
    %res = muse.mul %silu_res, %split#0 : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

    // After fusion:
    // CHECK-NOT: muse.aclnn.split_with_size
    // CHECK-NOT: muse.aclnn.silu
    // CHECK-NOT: muse.mul
    // CHECK: %[[RES:.*]] = muse.aclnn.swi_glu %[[INPUT]], %[[DIM]]
    // CHECK: return %[[RES]]

    return %res : tensor<2x2xf32>
  }

  // CHECK-LABEL: func @test_no_fusion_wrong_split_output
  func.func @test_no_fusion_wrong_split_output(%input: tensor<2x4xf32>, %dim: tensor<i64>) -> tensor<2x2xf32> {
    %split_size = arith.constant dense<[2, 2]> : tensor<2xi64>
    %split:2 = muse.aclnn.split_with_size %input, %split_size, %dim : (tensor<2x4xf32>, tensor<2xi64>, tensor<i64>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

    // SiLU on split#0 instead of split#1 (Mismatch pattern)
    %silu_res = muse.aclnn.silu %split#0 : (tensor<2x2xf32>) -> tensor<2x2xf32>

    %res = muse.mul %split#1, %silu_res : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>

    // Should not fuse
    // CHECK: muse.aclnn.split_with_size
    // CHECK: muse.aclnn.silu
    // CHECK: muse.mul
    // CHECK-NOT: muse.aclnn.swi_glu

    return %res : tensor<2x2xf32>
  }

  // CHECK-LABEL: func @test_swiglu_with_reshape_fusion
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<8xf16>, %[[DIM:.*]]: tensor<i64>)
  func.func @test_swiglu_with_reshape_fusion(%input: tensor<8xf16>, %dim: tensor<i64>) -> tensor<1x1x2x2xf16> {
    // Reshape input: 1D tensor -> 2D tensor
    %reshape_in_shape = arith.constant dense<[2, 4]> : tensor<2xi64>
    %reshape_in = muse.reshape %input, %reshape_in_shape : (tensor<8xf16>, tensor<2xi64>) -> tensor<2x4xf16>

    // SplitWithSize: split along dim=1 into [2, 2]
    %split_size = arith.constant dense<[2, 2]> : tensor<2xi64>
    %split:2 = muse.aclnn.split_with_size %reshape_in, %split_size, %dim : (tensor<2x4xf16>, tensor<2xi64>, tensor<i64>) -> (tensor<2x2xf16>, tensor<2x2xf16>)

    // Reshape split outputs: 2D -> 4D
    %reshape_shape_4d = arith.constant dense<[1, 1, 2, 2]> : tensor<4xi64>
    %reshape_split1 = muse.reshape %split#1, %reshape_shape_4d : (tensor<2x2xf16>, tensor<4xi64>) -> tensor<1x1x2x2xf16>
    %reshape_split0 = muse.reshape %split#0, %reshape_shape_4d : (tensor<2x2xf16>, tensor<4xi64>) -> tensor<1x1x2x2xf16>

    // Silu on Reshape(Split[0])
    %silu_res = muse.aclnn.silu %reshape_split0 : (tensor<1x1x2x2xf16>) -> tensor<1x1x2x2xf16>

    // Mul: Reshape(Split[1]) * Silu(...)
    %res = muse.mul %reshape_split1, %silu_res : (tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16>) -> tensor<1x1x2x2xf16>

    // After fusion:
    // CHECK-NOT: muse.aclnn.split_with_size
    // CHECK-NOT: muse.aclnn.silu
    // CHECK-NOT: muse.mul
    // CHECK: %[[RES:.*]] = muse.aclnn.swi_glu %[[INPUT]], %[[DIM]]
    // CHECK: return %[[RES]]

    return %res : tensor<1x1x2x2xf16>
  }

  // CHECK-LABEL: func @test_swiglu_with_reshape_fusion_commutative
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<8xf16>, %[[DIM:.*]]: tensor<i64>)
  func.func @test_swiglu_with_reshape_fusion_commutative(%input: tensor<8xf16>, %dim: tensor<i64>) -> tensor<1x1x2x2xf16> {
    // Test Mul commutativity with reshape pattern
    %reshape_in_shape = arith.constant dense<[2, 4]> : tensor<2xi64>
    %reshape_in = muse.reshape %input, %reshape_in_shape : (tensor<8xf16>, tensor<2xi64>) -> tensor<2x4xf16>

    %split_size = arith.constant dense<[2, 2]> : tensor<2xi64>
    %split:2 = muse.aclnn.split_with_size %reshape_in, %split_size, %dim : (tensor<2x4xf16>, tensor<2xi64>, tensor<i64>) -> (tensor<2x2xf16>, tensor<2x2xf16>)

    %reshape_shape_4d = arith.constant dense<[1, 1, 2, 2]> : tensor<4xi64>
    %reshape_split1 = muse.reshape %split#1, %reshape_shape_4d : (tensor<2x2xf16>, tensor<4xi64>) -> tensor<1x1x2x2xf16>
    %reshape_split0 = muse.reshape %split#0, %reshape_shape_4d : (tensor<2x2xf16>, tensor<4xi64>) -> tensor<1x1x2x2xf16>

    %silu_res = muse.aclnn.silu %reshape_split0 : (tensor<1x1x2x2xf16>) -> tensor<1x1x2x2xf16>

    // Mul: Silu(...) * Reshape(Split[1]) (swapped order)
    %res = muse.mul %silu_res, %reshape_split1 : (tensor<1x1x2x2xf16>, tensor<1x1x2x2xf16>) -> tensor<1x1x2x2xf16>

    // After fusion:
    // CHECK-NOT: muse.aclnn.split_with_size
    // CHECK-NOT: muse.aclnn.silu
    // CHECK-NOT: muse.mul
    // CHECK: %[[RES:.*]] = muse.aclnn.swi_glu %[[INPUT]], %[[DIM]]
    // CHECK: return %[[RES]]

    return %res : tensor<1x1x2x2xf16>
  }

  // CHECK-LABEL: func @test_no_fusion_with_reshape_dtype_mismatch
  func.func @test_no_fusion_with_reshape_dtype_mismatch(%input: tensor<8xf32>, %dim: tensor<i64>) -> tensor<1x1x2x2xf32> {
    // Input is f32, but pass requires f16 or bf16
    %reshape_in_shape = arith.constant dense<[2, 4]> : tensor<2xi64>
    %reshape_in = muse.reshape %input, %reshape_in_shape : (tensor<8xf32>, tensor<2xi64>) -> tensor<2x4xf32>

    %split_size = arith.constant dense<[2, 2]> : tensor<2xi64>
    %split:2 = muse.aclnn.split_with_size %reshape_in, %split_size, %dim : (tensor<2x4xf32>, tensor<2xi64>, tensor<i64>) -> (tensor<2x2xf32>, tensor<2x2xf32>)

    %reshape_shape_4d = arith.constant dense<[1, 1, 2, 2]> : tensor<4xi64>
    %reshape_split1 = muse.reshape %split#1, %reshape_shape_4d : (tensor<2x2xf32>, tensor<4xi64>) -> tensor<1x1x2x2xf32>
    %reshape_split0 = muse.reshape %split#0, %reshape_shape_4d : (tensor<2x2xf32>, tensor<4xi64>) -> tensor<1x1x2x2xf32>

    %silu_res = muse.aclnn.silu %reshape_split0 : (tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>
    %res = muse.mul %reshape_split1, %silu_res : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x2x2xf32>

    // Should not fuse due to dtype check
    // CHECK: muse.aclnn.split_with_size
    // CHECK: muse.aclnn.silu
    // CHECK: muse.mul
    // CHECK-NOT: muse.aclnn.swi_glu

    return %res : tensor<1x1x2x2xf32>
  }

  // CHECK-LABEL: func @test_no_fusion_with_reshape_reshape_dim_too_large
  func.func @test_no_fusion_with_reshape_reshape_dim_too_large(%input: tensor<16xf16>, %dim: tensor<i64>) -> tensor<1x1x1x2x1xf16> {
    // Input Reshape has 4 dims, but pass requires <= 3
    %reshape_in_shape = arith.constant dense<[2, 2, 2, 2]> : tensor<4xi64>
    %reshape_in = muse.reshape %input, %reshape_in_shape : (tensor<16xf16>, tensor<4xi64>) -> tensor<2x2x2x2xf16>

    %split_size = arith.constant dense<[1, 1]> : tensor<2xi64>
    %split:2 = muse.aclnn.split_with_size %reshape_in, %split_size, %dim : (tensor<2x2x2x2xf16>, tensor<2xi64>, tensor<i64>) -> (tensor<2x2x2x1xf16>, tensor<2x2x2x1xf16>)

    %reshape_shape_5d = arith.constant dense<[1, 1, 1, 2, 1]> : tensor<5xi64>
    %reshape_split1 = muse.reshape %split#1, %reshape_shape_5d : (tensor<2x2x2x1xf16>, tensor<5xi64>) -> tensor<1x1x1x2x1xf16>
    %reshape_split0 = muse.reshape %split#0, %reshape_shape_5d : (tensor<2x2x2x1xf16>, tensor<5xi64>) -> tensor<1x1x1x2x1xf16>

    %silu_res = muse.aclnn.silu %reshape_split0 : (tensor<1x1x1x2x1xf16>) -> tensor<1x1x1x2x1xf16>
    %res = muse.mul %reshape_split1, %silu_res : (tensor<1x1x1x2x1xf16>, tensor<1x1x1x2x1xf16>) -> tensor<1x1x1x2x1xf16>

    // Should not fuse due to reshape dim check
    // CHECK-NOT: muse.aclnn.swi_glu

    return %res : tensor<1x1x1x2x1xf16>
  }
}
