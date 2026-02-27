// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=tanh" -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @test_tanh(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.aclnn.tanh %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.tanh
  // CHECK: mfuse.add
}

func.func @test_tanh_no_decompose(%arg0: tensor<4x4xf64>) -> tensor<4x4xf64> {
  %0 = mfuse.aclnn.tanh %arg0 : (tensor<4x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
  // CHECK: mfuse.aclnn.tanh
}

// decompose tanhOp with dynamic shape torch tensor input
func.func @test_tanh_dynamic_shape(%arg0: !torch.int, %arg1: !torch.vtensor<[2,?],f32>) -> !torch.vtensor<[2,?],f32> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
  %0 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,?],f32> to tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %1 = mfuse.aclnn.tanh %0 : (tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>> to !torch.vtensor<[2,?],f32>
  return %2 : !torch.vtensor<[2,?],f32>

  // CHECK-LABEL: func.func @test_tanh_dynamic_shape
  // CHECK-NOT: mfuse.aclnn.tanh
  // CHECK: mfuse.mul {{.*}} -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  // CHECK: mfuse.add {{.*}} -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  // CHECK: mfuse.add {{.*}} -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  // CHECK: mfuse.div {{.*}} -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
}
