// RUN: mfusion-opt %s --fuse-rms-norm | FileCheck %s

module {
  // CHECK-LABEL: func @test_rms_norm_decomposed_fusion
  // Decomposed RmsNorm: add(mean, eps) -> rsqrt -> mul(x, rsqrt) -> mul(gamma)
  // Should fuse to aclnn.rms_norm(x, gamma, epsilon)
  func.func @test_rms_norm_decomposed_fusion(%x: tensor<2x4xf32>, %gamma: tensor<2x4xf32>, %mean: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %eps = arith.constant dense<1.000000e-05> : tensor<2x4xf32>
    %add = mfuse.add %mean, %eps : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %rsqrt = mfuse.rsqrt %add : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %norm = mfuse.mul %x, %rsqrt : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %out = mfuse.mul %gamma, %norm : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }
  // CHECK-NOT: mfuse.rsqrt
  // CHECK-NOT: mfuse.mul
  // CHECK: mfuse.aclnn.rms_norm {{.*}}, {{.*}} {epsilon = {{.*}}}
  // CHECK: return

  // CHECK-LABEL: func @test_rms_norm_rsqrt_returned
  // Training graph: rsqrt is returned for backward pass (rsqrt has 2 uses).
  // Fusion should still apply; all rsqrt uses replaced by rstdOut.
  func.func @test_rms_norm_rsqrt_returned(%x: tensor<2x4xf32>, %gamma: tensor<2x4xf32>, %mean: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
    %eps = arith.constant dense<1.000000e-05> : tensor<2x4xf32>
    %add = mfuse.add %mean, %eps : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %rsqrt = mfuse.rsqrt %add : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %norm = mfuse.mul %x, %rsqrt : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %out = mfuse.mul %gamma, %norm : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %out, %rsqrt : tensor<2x4xf32>, tensor<2x4xf32>
  }
  // CHECK: mfuse.aclnn.rms_norm
  // CHECK-NOT: mfuse.rsqrt

  // CHECK-LABEL: func @test_no_fusion_multiple_uses
  // First mul result has multiple uses - should not fuse
  func.func @test_no_fusion_multiple_uses(%x: tensor<2x4xf32>, %gamma: tensor<2x4xf32>, %mean: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
    %eps = arith.constant dense<1.0e-05> : tensor<2x4xf32>
    %add = mfuse.add %mean, %eps : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %rsqrt = mfuse.rsqrt %add : (tensor<2x4xf32>) -> tensor<2x4xf32>
    %norm = mfuse.mul %x, %rsqrt : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    %out = mfuse.mul %gamma, %norm : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %out, %norm : tensor<2x4xf32>, tensor<2x4xf32>
  }
  // When fusion does not apply (multiple uses), rsqrt and mul remain (order: rsqrt then mul in IR)
  // CHECK: mfuse.rsqrt
  // CHECK: mfuse.mul
  // CHECK-NOT: mfuse.aclnn.rms_norm
}

