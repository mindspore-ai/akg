// RUN: mfusion-opt %s --fuse-addrmsnorm | FileCheck %s

module {
  // CHECK-LABEL: func @test_add_rms_norm_fusion
  // CHECK-SAME: (%[[X1:.*]]: tensor<2x4xf32>, %[[X2:.*]]: tensor<2x4xf32>, %[[GAMMA:.*]]: tensor<2x4xf32>, %{{.*}}: i1)
  func.func @test_add_rms_norm_fusion(%x1: tensor<2x4xf32>, %x2: tensor<2x4xf32>, %gamma: tensor<2x4xf32>, %unused: i1)
      -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
    // Positive path: add + rms_norm -> add_rms_norm. Extra unused i1 arg should not affect fusion.
    %add_res = mfuse.add %x1, %x2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    %y, %rstd = mfuse.aclnn.rms_norm %add_res, %gamma {epsilon = 1.000000e-05 : f64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)

    // CHECK: %[[Y:.*]], %[[RSTD:.*]], %[[X:.*]] = mfuse.aclnn.add_rms_norm %[[X1]], %[[X2]], %[[GAMMA]] {epsilon = 1.000000e-05 : f64}
    // CHECK: return %[[Y]], %[[RSTD]], %[[X]]

    return %y, %rstd, %add_res : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_no_fusion_element_type_mismatch
  func.func @test_no_fusion_element_type_mismatch(%x1: tensor<2x4xf32>, %x2: tensor<2x4xf32>, %gamma: tensor<2x4xf16>) -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
    // Element type mismatch: x is f32 but gamma is f16
    %add_res = mfuse.add %x1, %x2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %y, %rstd = mfuse.aclnn.rms_norm %add_res, %gamma {epsilon = 1.000000e-05 : f64} : (tensor<2x4xf32>, tensor<2x4xf16>) -> (tensor<2x4xf32>, tensor<2x4xf32>)

    // CHECK: mfuse.add
    // CHECK: mfuse.aclnn.rms_norm
    // CHECK-NOT: mfuse.aclnn.add_rms_norm

    return %y, %rstd, %add_res : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_no_fusion_shape_mismatch
  // CHECK-SAME: (%[[X1:.*]]: tensor<2x4xf32>, %[[X2:.*]]: tensor<3x4xf32>, %[[GAMMA:.*]]: tensor<2x4xf32>)
  func.func @test_no_fusion_shape_mismatch(%x1: tensor<2x4xf32>, %x2: tensor<3x4xf32>, %gamma: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>) {
    // Shape mismatch: x1 is 2x4 but x2 is 3x4; pass requires identical x1/x2 shapes.
    %add_res = mfuse.add %x1, %x2 : (tensor<2x4xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    %y, %rstd = mfuse.aclnn.rms_norm %add_res, %gamma {epsilon = 1.000000e-05 : f64} : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)

    // CHECK: mfuse.add
    // CHECK: mfuse.aclnn.rms_norm
    // CHECK-NOT: mfuse.aclnn.add_rms_norm

    return %y, %rstd, %add_res : tensor<2x4xf32>, tensor<2x4xf32>, tensor<2x4xf32>
  }
}
