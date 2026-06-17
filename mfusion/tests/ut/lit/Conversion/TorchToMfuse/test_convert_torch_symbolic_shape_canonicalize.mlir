// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --canonicalize | FileCheck %s
// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --convert-mfuse-to-torch --reconcile-unrealized-casts --canonicalize | FileCheck %s --check-prefix=ROUNDTRIP

module {
  // CHECK-LABEL: func.func @constant_bind_shape
  // CHECK-SAME: %arg0: !torch.vtensor<[1,1,6,37],f32>
  // CHECK-SAME: %arg1: !torch.vtensor<[1,1,6,37],f32>
  // CHECK-SAME: -> !torch.vtensor<[1,1,6,37],f32>
  // CHECK-NOT: torch.bind_symbolic_shape
  // CHECK-DAG: %[[C0:.*]] = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[1,1,6,37],f32> to tensor<1x1x6x37xf32>
  // CHECK-DAG: %[[C1:.*]] = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[1,1,6,37],f32> to tensor<1x1x6x37xf32>
  // CHECK: mfuse.mul %[[C0]], %[[C1]] : (tensor<1x1x6x37xf32>, tensor<1x1x6x37xf32>) -> tensor<1x1x6x37xf32>
  // CHECK-NOT: tensor<1x1x?x37xf32, #mfuse.symshape<["1", "1", "6", "37"]>>
  // ROUNDTRIP-LABEL: func.func @constant_bind_shape
  // ROUNDTRIP-SAME: %arg0: !torch.vtensor<[1,1,6,37],f32>
  // ROUNDTRIP-SAME: %arg1: !torch.vtensor<[1,1,6,37],f32>
  // ROUNDTRIP-SAME: -> !torch.vtensor<[1,1,6,37],f32>
  // ROUNDTRIP-NOT: builtin.unrealized_conversion_cast
  // ROUNDTRIP: torch.aten.mul.Tensor
  // ROUNDTRIP-NOT: builtin.unrealized_conversion_cast
  func.func @constant_bind_shape(%arg0: !torch.vtensor<[1,1,?,37],f32>, %arg1: !torch.vtensor<[1,1,?,37],f32>)
      -> !torch.vtensor<[1,1,?,37],f32> attributes {torch.assume_strict_symbolic_shapes} {
    torch.bind_symbolic_shape %arg0, [], affine_map<() -> (1, 1, 6, 37)> : !torch.vtensor<[1,1,?,37],f32>
    torch.bind_symbolic_shape %arg1, [], affine_map<() -> (1, 1, 6, 37)> : !torch.vtensor<[1,1,?,37],f32>
    %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[1,1,?,37],f32>, !torch.vtensor<[1,1,?,37],f32> -> !torch.vtensor<[1,1,?,37],f32>
    torch.bind_symbolic_shape %0, [], affine_map<() -> (1, 1, 6, 37)> : !torch.vtensor<[1,1,?,37],f32>
    return %0 : !torch.vtensor<[1,1,?,37],f32>
  }

  // CHECK-LABEL: func.func @constant_result_with_symbolic_int
  // CHECK-SAME: %arg0: !torch.vtensor<[1,1,6,37],f32>
  // CHECK-SAME: %arg1: !torch.vtensor<[1,1,6,37],f32>
  // CHECK-SAME: -> !torch.vtensor<[1,1,6,37],f32>
  // CHECK: mfuse.syminfo = {s_single = #mfuse.syminfo<range=[6, 6]>}
  // CHECK-NOT: torch.bind_symbolic_shape
  // CHECK-DAG: %[[CS0:.*]] = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[1,1,6,37],f32> to tensor<1x1x6x37xf32>
  // CHECK-DAG: %[[CS1:.*]] = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[1,1,6,37],f32> to tensor<1x1x6x37xf32>
  // CHECK: mfuse.mul %[[CS0]], %[[CS1]] : (tensor<1x1x6x37xf32>, tensor<1x1x6x37xf32>) -> tensor<1x1x6x37xf32>
  // CHECK-NOT: tensor<1x1x?x37xf32, #mfuse.symshape<["1", "1", "6", "37"]>>
  // ROUNDTRIP-LABEL: func.func @constant_result_with_symbolic_int
  // ROUNDTRIP-SAME: %arg0: !torch.vtensor<[1,1,6,37],f32>
  // ROUNDTRIP-SAME: %arg1: !torch.vtensor<[1,1,6,37],f32>
  // ROUNDTRIP-SAME: -> !torch.vtensor<[1,1,6,37],f32>
  // ROUNDTRIP-NOT: builtin.unrealized_conversion_cast
  // ROUNDTRIP: torch.aten.mul.Tensor
  // ROUNDTRIP-NOT: builtin.unrealized_conversion_cast
  func.func @constant_result_with_symbolic_int(%arg0: !torch.vtensor<[1,1,?,37],f32>, %arg1: !torch.vtensor<[1,1,?,37],f32>)
      -> !torch.vtensor<[1,1,?,37],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %s_single = torch.symbolic_int "s_single" {min_val = 6, max_val = 6} : !torch.int
    torch.bind_symbolic_shape %arg0, [%s_single], affine_map<()[s0] -> (1, 1, 6, 37)> : !torch.vtensor<[1,1,?,37],f32>
    torch.bind_symbolic_shape %arg1, [%s_single], affine_map<()[s0] -> (1, 1, 6, 37)> : !torch.vtensor<[1,1,?,37],f32>
    %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[1,1,?,37],f32>, !torch.vtensor<[1,1,?,37],f32> -> !torch.vtensor<[1,1,?,37],f32>
    torch.bind_symbolic_shape %0, [%s_single], affine_map<()[s0] -> (1, 1, 6, 37)> : !torch.vtensor<[1,1,?,37],f32>
    return %0 : !torch.vtensor<[1,1,?,37],f32>
  }

  // CHECK-LABEL: func.func @mixed_dynamic_bind_shape
  // CHECK: mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>}
  // CHECK-NOT: torch.bind_symbolic_shape
  // CHECK-DAG: %[[D0:.*]] = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[1,?,37],f32> to tensor<1x?x37xf32, #mfuse.symshape<["1", "s0", "37"]>>
  // CHECK-DAG: %[[D1:.*]] = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[1,?,37],f32> to tensor<1x?x37xf32, #mfuse.symshape<["1", "s0", "37"]>>
  // CHECK: mfuse.mul %[[D0]], %[[D1]] : (tensor<1x?x37xf32, #mfuse.symshape<["1", "s0", "37"]>>, tensor<1x?x37xf32, #mfuse.symshape<["1", "s0", "37"]>>) -> tensor<1x?x37xf32, #mfuse.symshape<["1", "s0", "37"]>>
  // ROUNDTRIP-LABEL: func.func @mixed_dynamic_bind_shape
  // ROUNDTRIP-SAME: %arg0: !torch.vtensor<[1,?,37],f32>
  // ROUNDTRIP-SAME: %arg1: !torch.vtensor<[1,?,37],f32>
  // ROUNDTRIP-NOT: builtin.unrealized_conversion_cast
  // ROUNDTRIP: torch.aten.mul.Tensor
  // ROUNDTRIP-NOT: builtin.unrealized_conversion_cast
  func.func @mixed_dynamic_bind_shape(%arg0: !torch.vtensor<[1,?,37],f32>, %arg1: !torch.vtensor<[1,?,37],f32>)
      -> !torch.vtensor<[1,?,37],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %s0 = torch.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg0, [%s0], affine_map<()[s0] -> (1, s0, 37)> : !torch.vtensor<[1,?,37],f32>
    torch.bind_symbolic_shape %arg1, [%s0], affine_map<()[s0] -> (1, s0, 37)> : !torch.vtensor<[1,?,37],f32>
    %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[1,?,37],f32>, !torch.vtensor<[1,?,37],f32> -> !torch.vtensor<[1,?,37],f32>
    torch.bind_symbolic_shape %0, [%s0], affine_map<()[s0] -> (1, s0, 37)> : !torch.vtensor<[1,?,37],f32>
    return %0 : !torch.vtensor<[1,?,37],f32>
  }

}
