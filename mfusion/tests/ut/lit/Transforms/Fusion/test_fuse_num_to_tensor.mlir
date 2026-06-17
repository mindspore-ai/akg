// RUN: mfusion-opt %s --fuse-num-to-tensor | FileCheck %s

module {
  // CHECK-LABEL: func @test_fuse_num_to_tensor_cast
  // CHECK-NOT: mfuse.num_to_tensor
  // CHECK-NOT: mfuse.cast
  // CHECK: %[[RESULT:.*]] = mfuse.full
  // CHECK: return %[[RESULT]]
  func.func @test_fuse_num_to_tensor_cast() -> tensor<f16> {
    %value = mfuse.constant dense<42.0> : tensor<f64, {is_scalar = ""}>
    %num_to_tensor = mfuse.num_to_tensor %value : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
    %cast = mfuse.cast %num_to_tensor : (tensor<f32>) -> tensor<f16>
    return %cast : tensor<f16>
  }

  // CHECK-LABEL: func @test_fuse_num_to_tensor
  // CHECK-NOT: mfuse.num_to_tensor
  // CHECK: %[[RESULT:.*]] = mfuse.full
  // CHECK: return %[[RESULT]]
  func.func @test_fuse_num_to_tensor() -> tensor<f32> {
    %value = mfuse.constant dense<42.0> : tensor<f64, {is_scalar = ""} >
    %num_to_tensor = mfuse.num_to_tensor %value : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
    return %num_to_tensor : tensor<f32>
  }
}
