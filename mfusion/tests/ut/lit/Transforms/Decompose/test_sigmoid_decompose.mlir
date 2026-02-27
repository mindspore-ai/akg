// RUN: mfusion-opt %s -decompose="pattern-type=AFTER_MANUAL_FUSION op-list=sigmoid" -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

func.func @test_sigmoid(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.aclnn.sigmoid %arg0 : (tensor<4x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
  // CHECK-NOT: mfuse.aclnn.sigmoid
  // CHECK: mfuse.exp
}

// decompose sigmoidOp with dynamic shape torch tensor input
func.func @test_sigmoid_dynamic_shape(%arg0: !torch.int, %arg1: !torch.vtensor<[2,?],f32>) -> !torch.vtensor<[2,?],f32> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
  %0 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,?],f32> to tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %1 = mfuse.aclnn.sigmoid %0 : (tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>> to !torch.vtensor<[2,?],f32>
  return %2 : !torch.vtensor<[2,?],f32>

  // CHECK-LABEL: func.func @test_sigmoid_dynamic_shape
  // CHECK-NOT: mfuse.aclnn.sigmoid
  // CHECK: mfuse.exp {{.*}} -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
}
