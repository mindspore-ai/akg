// RUN: mfusion-opt %s --fuse-gelu | FileCheck %s

module {
  // CHECK-LABEL: func @test_gelu_fusion
  // CHECK-SAME: (%[[X:.*]]: tensor<2x4xf32>)
  func.func @test_gelu_fusion(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c3 = arith.constant dense<3.000000e+00> : tensor<f64>
    %pow3 = muse.pow %x, %c3 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %c004 = arith.constant dense<4.471500e-02> : tensor<f64>
    %mul_c = muse.mul %pow3, %c004 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %add1 = muse.add %x, %mul_c : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c_sqrt = arith.constant dense<7.978846e-01> : tensor<f64>
    %mul_s = muse.mul %add1, %c_sqrt : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %tanh = muse.aclnn.tanh %mul_s : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %ones = arith.constant dense<1.0> : tensor<f32>
    %add2 = muse.add %ones, %tanh : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c05 = arith.constant dense<5.000000e-01> : tensor<f64>
    %half = muse.mul %x, %c05 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %result = muse.mul %half, %add2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // After fusion: GELU pattern should be replaced by muse.aclnn.gelu
    // CHECK-NOT: muse.pow
    // CHECK-NOT: muse.mul
    // CHECK-NOT: muse.add
    // CHECK-NOT: muse.aclnn.tanh
    // CHECK-NOT: muse.constant_tensor
    // CHECK: %[[GELU:.*]] = muse.aclnn.gelu %[[X]]
    // CHECK: return %[[GELU]]

    return %result : tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_no_fusion_wrong_exponent
  func.func @test_no_fusion_wrong_exponent(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c2 = arith.constant dense<2.000000e+00> : tensor<f64>
    %pow2 = muse.pow %x, %c2 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %c004 = arith.constant dense<4.471500e-02> : tensor<f64>
    %mul_c = muse.mul %pow2, %c004 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %add1 = muse.add %x, %mul_c : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c_sqrt = arith.constant dense<7.978846e-01> : tensor<f64>
    %mul_s = muse.mul %add1, %c_sqrt : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %tanh = muse.aclnn.tanh %mul_s : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %ones = arith.constant dense<1.0> : tensor<f32>
    %add2 = muse.add %ones, %tanh : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c05 = arith.constant dense<5.000000e-01> : tensor<f64>
    %half = muse.mul %x, %c05 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %result = muse.mul %half, %add2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // Should not fuse: exponent is 2.0, not 3.0
    // CHECK: muse.pow
    // CHECK: muse.mul
    // CHECK-NOT: muse.aclnn.gelu

    return %result : tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_no_fusion_wrong_coeff
  func.func @test_no_fusion_wrong_coeff(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c3 = arith.constant dense<3.000000e+00> : tensor<f64>
    %pow3 = muse.pow %x, %c3 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %c05_coeff = arith.constant dense<5.000000e-02> : tensor<f64>
    %mul_c = muse.mul %pow3, %c05_coeff : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %add1 = muse.add %x, %mul_c : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c_sqrt = arith.constant dense<7.978846e-01> : tensor<f64>
    %mul_s = muse.mul %add1, %c_sqrt : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %tanh = muse.aclnn.tanh %mul_s : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %ones = arith.constant dense<1.0> : tensor<f32>
    %add2 = muse.add %ones, %tanh : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c05 = arith.constant dense<5.000000e-01> : tensor<f64>
    %half = muse.mul %x, %c05 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %result = muse.mul %half, %add2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // Should not fuse: 0.044715 replaced by 0.05
    // CHECK: muse.pow
    // CHECK: muse.mul
    // CHECK-NOT: muse.aclnn.gelu

    return %result : tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_gelu_fusion_commutative
  // Test that MulOp commutativity is handled correctly (0.5 * x vs x * 0.5)
  func.func @test_gelu_fusion_commutative(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c3 = arith.constant dense<3.000000e+00> : tensor<f64>
    %pow3 = muse.pow %x, %c3 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %c004 = arith.constant dense<4.471500e-02> : tensor<f64>
    %mul_c = muse.mul %c004, %pow3 : (tensor<f64>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %add1 = muse.add %x, %mul_c : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c_sqrt = arith.constant dense<7.978846e-01> : tensor<f64>
    %mul_s = muse.mul %c_sqrt, %add1 : (tensor<f64>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %tanh = muse.aclnn.tanh %mul_s : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %ones = arith.constant dense<1.0> : tensor<f32>
    %add2 = muse.add %ones, %tanh : (tensor<f32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c05 = arith.constant dense<5.000000e-01> : tensor<f64>
    %half = muse.mul %c05, %x : (tensor<f64>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %result = muse.mul %half, %add2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // Should fuse: commutativity handled correctly
    // CHECK-NOT: muse.pow
    // CHECK-NOT: muse.aclnn.tanh
    // CHECK: %[[GELU:.*]] = muse.aclnn.gelu %[[X]]
    // CHECK: return %[[GELU]]

    return %result : tensor<2x4xf32>
  }

  // CHECK-LABEL: func @test_gelu_fusion_add_commutative
  // Test that AddOp commutativity is handled correctly (Add(x, mulC) vs Add(mulC, x), Add(ones, tanh) vs Add(tanh, ones))
  func.func @test_gelu_fusion_add_commutative(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %c3 = arith.constant dense<3.000000e+00> : tensor<f64>
    %pow3 = muse.pow %x, %c3 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %c004 = arith.constant dense<4.471500e-02> : tensor<f64>
    %mul_c = muse.mul %pow3, %c004 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    // Add(mulC, x) - reversed order
    %add1 = muse.add %mul_c, %x : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %c_sqrt = arith.constant dense<7.978846e-01> : tensor<f64>
    %mul_s = muse.mul %add1, %c_sqrt : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %tanh = muse.aclnn.tanh %mul_s : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %ones = arith.constant dense<1.0> : tensor<f32>
    // Add(tanh, ones) - reversed order
    %add2 = muse.add %tanh, %ones : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x4xf32>
    %c05 = arith.constant dense<5.000000e-01> : tensor<f64>
    %half = muse.mul %x, %c05 : (tensor<2x4xf32>, tensor<f64>) -> tensor<2x4xf32>
    %result = muse.mul %half, %add2 : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>

    // Should fuse: Add commutativity handled correctly
    // CHECK-NOT: muse.pow
    // CHECK-NOT: muse.aclnn.tanh
    // CHECK: %[[GELU:.*]] = muse.aclnn.gelu %[[X]]
    // CHECK: return %[[GELU]]

    return %result : tensor<2x4xf32>
  }
}
