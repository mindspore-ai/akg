// RUN: mfusion-opt %s --fuse-gelu | FileCheck %s

module {
  // CHECK-LABEL: func @test_gelu_fusion
  // CHECK-SAME: (%[[X:.*]]: tensor<2x4xf32>)
  func.func @test_gelu_fusion(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c3 = arith.constant dense<3.000000e+00> : tensor<f32>
    %pow3 = mfuse.pow %x, %c3 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %c004 = arith.constant dense<4.471500e-02> : tensor<f32>
    %mul_c = mfuse.mul %pow3, %c004 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %add1 = mfuse.add %x, %mul_c : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c_sqrt = arith.constant dense<7.978846e-01> : tensor<f32>
    %mul_s = mfuse.mul %add1, %c_sqrt : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %tanh = mfuse.aclnn.tanh %mul_s : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %ones = arith.constant dense<1.0> : tensor<f32>
    %add2 = mfuse.add %ones, %tanh : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c05 = arith.constant dense<5.000000e-01> : tensor<f32>
    %half = mfuse.mul %x, %c05 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %result = mfuse.mul %half, %add2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // After fusion: GELU pattern should be replaced by mfuse.aclnn.gelu
    // CHECK-NOT: mfuse.pow
    // CHECK-NOT: mfuse.mul
    // CHECK-NOT: mfuse.add
    // CHECK-NOT: mfuse.aclnn.tanh
    // CHECK-NOT: mfuse.constant_tensor
    // CHECK: %[[GELU:.*]] = mfuse.aclnn.gelu %[[X]] {approximate = "tanh"}
    // CHECK: return %[[GELU]]

    return %result : tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_no_fusion_wrong_exponent
  func.func @test_no_fusion_wrong_exponent(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c2 = arith.constant dense<2.000000e+00> : tensor<f32>
    %pow2 = mfuse.pow %x, %c2 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %c004 = arith.constant dense<4.471500e-02> : tensor<f32>
    %mul_c = mfuse.mul %pow2, %c004 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %add1 = mfuse.add %x, %mul_c : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c_sqrt = arith.constant dense<7.978846e-01> : tensor<f32>
    %mul_s = mfuse.mul %add1, %c_sqrt : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %tanh = mfuse.aclnn.tanh %mul_s : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %ones = arith.constant dense<1.0> : tensor<f32>
    %add2 = mfuse.add %ones, %tanh : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c05 = arith.constant dense<5.000000e-01> : tensor<f32>
    %half = mfuse.mul %x, %c05 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %result = mfuse.mul %half, %add2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // Should not fuse: exponent is 2.0, not 3.0
    // CHECK: mfuse.pow
    // CHECK: mfuse.mul
    // CHECK-NOT: mfuse.aclnn.gelu

    return %result : tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_no_fusion_wrong_coeff
  func.func @test_no_fusion_wrong_coeff(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c3 = arith.constant dense<3.000000e+00> : tensor<f32>
    %pow3 = mfuse.pow %x, %c3 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %c05_coeff = arith.constant dense<5.000000e-02> : tensor<f32>
    %mul_c = mfuse.mul %pow3, %c05_coeff : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %add1 = mfuse.add %x, %mul_c : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c_sqrt = arith.constant dense<7.978846e-01> : tensor<f32>
    %mul_s = mfuse.mul %add1, %c_sqrt : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %tanh = mfuse.aclnn.tanh %mul_s : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %ones = arith.constant dense<1.0> : tensor<f32>
    %add2 = mfuse.add %ones, %tanh : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c05 = arith.constant dense<5.000000e-01> : tensor<f32>
    %half = mfuse.mul %x, %c05 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %result = mfuse.mul %half, %add2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // Should not fuse: 0.044715 replaced by 0.05
    // CHECK: mfuse.pow
    // CHECK: mfuse.mul
    // CHECK-NOT: mfuse.aclnn.gelu

    return %result : tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_gelu_fusion_commutative
  // Test that MulOp commutativity is handled correctly (0.5 * x vs x * 0.5)
  func.func @test_gelu_fusion_commutative(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c3 = arith.constant dense<3.000000e+00> : tensor<f32>
    %pow3 = mfuse.pow %x, %c3 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %c004 = arith.constant dense<4.471500e-02> : tensor<f32>
    %mul_c = mfuse.mul %c004, %pow3 : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %add1 = mfuse.add %x, %mul_c : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c_sqrt = arith.constant dense<7.978846e-01> : tensor<f32>
    %mul_s = mfuse.mul %c_sqrt, %add1 : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %tanh = mfuse.aclnn.tanh %mul_s : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %ones = arith.constant dense<1.0> : tensor<f32>
    %add2 = mfuse.add %ones, %tanh : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c05 = arith.constant dense<5.000000e-01> : tensor<f32>
    %half = mfuse.mul %c05, %x : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %result = mfuse.mul %half, %add2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // Should fuse: commutativity handled correctly
    // CHECK-NOT: mfuse.pow
    // CHECK-NOT: mfuse.aclnn.tanh
    // CHECK: %[[GELU:.*]] = mfuse.aclnn.gelu %[[X]] {approximate = "tanh"}
    // CHECK: return %[[GELU]]

    return %result : tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_gelu_fusion_add_commutative
  // Test that AddOp commutativity is handled correctly (Add(x, mulC) vs Add(mulC, x), Add(ones, tanh) vs Add(tanh, ones))
  func.func @test_gelu_fusion_add_commutative(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c3 = arith.constant dense<3.000000e+00> : tensor<f32>
    %pow3 = mfuse.pow %x, %c3 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %c004 = arith.constant dense<4.471500e-02> : tensor<f32>
    %mul_c = mfuse.mul %pow3, %c004 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    // Add(mulC, x) - reversed order
    %add1 = mfuse.add %mul_c, %x : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c_sqrt = arith.constant dense<7.978846e-01> : tensor<f32>
    %mul_s = mfuse.mul %add1, %c_sqrt : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %tanh = mfuse.aclnn.tanh %mul_s : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %ones = arith.constant dense<1.0> : tensor<f32>
    // Add(tanh, ones) - reversed order
    %add2 = mfuse.add %tanh, %ones : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %c05 = arith.constant dense<5.000000e-01> : tensor<f32>
    %half = mfuse.mul %x, %c05 : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %result = mfuse.mul %half, %add2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // Should fuse: Add commutativity handled correctly
    // CHECK-NOT: mfuse.pow
    // CHECK-NOT: mfuse.aclnn.tanh
    // CHECK: %[[GELU:.*]] = mfuse.aclnn.gelu %[[X]] {approximate = "tanh"}
    // CHECK: return %[[GELU]]

    return %result : tensor<2x4xf32>
  }
}
