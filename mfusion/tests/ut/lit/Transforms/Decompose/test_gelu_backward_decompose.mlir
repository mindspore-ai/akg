// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=gelu_backward" -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @test_gelu_backward(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // geluBackwardOp
  %0 = mfuse.aclnn.gelu_backward %arg0, %arg1 {approximate = "tanh"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.gelu_backward
  // CHECK: mfuse.mul
}

// decompose geluBackwardOp with dynamic shape torch tensor input
func.func @test_gelu_backward_dynamic_shape(%arg0: !torch.int, %arg1: !torch.vtensor<[2,?],f32>, %arg2: !torch.vtensor<[2,?],f32>) -> !torch.vtensor<[2,?],f32> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
  %0 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,?],f32> to tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %1 = builtin.unrealized_conversion_cast %arg2 : !torch.vtensor<[2,?],f32> to tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %2 = mfuse.aclnn.gelu_backward %0, %1 {approximate = "tanh"} : (tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>> to !torch.vtensor<[2,?],f32>
  return %3 : !torch.vtensor<[2,?],f32>

  // CHECK-NOT: mfuse.aclnn.gelu_backward
  // CHECK: mfuse.mul {{.*}} -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
}
