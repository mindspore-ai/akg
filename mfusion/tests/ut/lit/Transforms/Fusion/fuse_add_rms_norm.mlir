// RUN: mfusion-opt %s --fuse-addrmsnorm | FileCheck %s

module {
  // CHECK-LABEL: func @test_add_rms_norm_fusion
  // CHECK-SAME: (%[[X1:.*]]: tensor<2x4xf32>, %[[X2:.*]]: tensor<2x4xf32>, %[[GAMMA:.*]]: tensor<2x4xf32>)
  func.func @test_add_rms_norm_fusion(%x1: tensor<2x4xf32>, %x2: tensor<2x4xf32>, %gamma: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
    // alpha = 1.0 (will be fused)
    %alpha_val = arith.constant dense<1.0> : tensor<f64>
    // Add operation: x1 + x2 * alpha (where alpha = 1.0)
    %add_res = muse.aclnn.add %x1, %x2, %alpha_val : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>

    // RmsNorm operation: RmsNorm(add_res, gamma, epsilon)
    %y, %rstd = muse.aclnn.rms_norm %add_res, %gamma {epsilon = 1.000000e-05 : f64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)

    // After fusion:
    // - muse.aclnn.add and muse.aclnn.rms_norm should be replaced by muse.aclnn.add_rms_norm
    // - muse.aclnn.add_rms_norm returns (y, rstd, x) where x is the fused add result
    // CHECK-NOT: muse.aclnn.add
    // CHECK-NOT: muse.aclnn.rms_norm
    // CHECK: %[[Y:.*]], %[[RSTD:.*]], %[[X:.*]] = muse.aclnn.add_rms_norm %[[X1]], %[[X2]], %[[GAMMA]] {epsilon = 1.000000e-05 : f64}
    // CHECK: return %[[Y]], %[[RSTD]], %[[X]]

    return %y, %rstd, %add_res : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_no_fusion_alpha_not_one
  func.func @test_no_fusion_alpha_not_one(%x1: tensor<2x4xf32>, %x2: tensor<2x4xf32>, %gamma: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
    // alpha = 0.5 (Not 1.0, should not fuse)
    %alpha_val = arith.constant dense<0.5> : tensor<f64>
    %add_res = muse.aclnn.add %x1, %x2, %alpha_val : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %y, %rstd = muse.aclnn.rms_norm %add_res, %gamma {epsilon = 1.000000e-05 : f64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)

    // Should not fuse because alpha != 1.0
    // CHECK: muse.aclnn.add
    // CHECK: muse.aclnn.rms_norm
    // CHECK-NOT: muse.aclnn.add_rms_norm

    return %y, %rstd, %add_res : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_no_fusion_element_type_mismatch
  func.func @test_no_fusion_element_type_mismatch(%x1: tensor<2x4xf32>, %x2: tensor<2x4xf32>, %gamma: tensor<2x4xf16>) -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
    // Element type mismatch: x is f32 but gamma is f16
    %alpha_val = arith.constant dense<1.0> : tensor<f64>
    %add_res = muse.aclnn.add %x1, %x2, %alpha_val : (tensor<2x4xf32>, tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %y, %rstd = muse.aclnn.rms_norm %add_res, %gamma {epsilon = 1.000000e-05 : f64} : (tensor<2x4xf32>, tensor<2x4xf16>) -> (tensor<2x4xf32>, tensor<2x4xf32>)

    // Should not fuse because element types don't match
    // CHECK: muse.aclnn.add
    // CHECK: muse.aclnn.rms_norm
    // CHECK-NOT: muse.aclnn.add_rms_norm

    return %y, %rstd, %add_res : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_no_fusion_shape_mismatch
  // CHECK-SAME: (%[[X1:.*]]: tensor<2x4xf32>, %[[X2:.*]]: tensor<3x4xf32>, %[[GAMMA:.*]]: tensor<2x4xf32>)
  func.func @test_no_fusion_shape_mismatch(%x1: tensor<2x4xf32>, %x2: tensor<3x4xf32>, %gamma: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
    // Shape mismatch: x1 is 2x4 but x2 is 3x4 (different first dimension)
    %alpha_val = arith.constant dense<1.0> : tensor<f64>
    // Note: muse.add may still work with broadcasting, but fusion should fail
    // because Pass checks for exact shape match: x1Type.getShape() != x2Type.getShape()
    %add_res = muse.aclnn.add %x1, %x2, %alpha_val : (tensor<2x4xf32>, tensor<3x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %y, %rstd = muse.aclnn.rms_norm %add_res, %gamma {epsilon = 1.000000e-05 : f64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)

    // Should not fuse because x1 and x2 have different shapes
    // CHECK: muse.aclnn.add
    // CHECK: muse.aclnn.rms_norm
    // CHECK-NOT: muse.aclnn.add_rms_norm

    return %y, %rstd, %add_res : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
  }
}
