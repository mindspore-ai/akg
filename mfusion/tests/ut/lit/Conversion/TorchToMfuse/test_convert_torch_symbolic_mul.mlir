// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse | FileCheck %s

module {
  // CHECK-LABEL: func.func @main
  // CHECK-SAME: %arg0: !torch.int
  // CHECK-SAME: %arg1: !torch.vtensor<[2,?],f32>
  // CHECK-SAME: %arg2: !torch.vtensor<[2,?],f32>
  // CHECK: mfuse.syminfo = {s10 = #mfuse.syminfo<range=[2, inf]>}
  // CHECK-NOT: torch.bind_symbolic_shape
  // CHECK: %[[CAST0:.*]] = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,?],f32> to tensor<2x?xf32, #mfuse.symshape<["2", "s10"]>>
  // CHECK: %[[CAST1:.*]] = builtin.unrealized_conversion_cast %arg2 : !torch.vtensor<[2,?],f32> to tensor<2x?xf32, #mfuse.symshape<["2", "s10"]>>
  // CHECK: %[[SYM:.*]] = torch.symbolic_int "s10" {min_val = 2, max_val = 9223372036854775807} : !torch.int
  // CHECK: %[[MUL:.*]] = mfuse.mul %[[CAST1]], %[[CAST0]] : (tensor<2x?xf32, #mfuse.symshape<["2", "s10"]>>, tensor<2x?xf32, #mfuse.symshape<["2", "s10"]>>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s10"]>>
  // CHECK: %[[OUT:.*]] = builtin.unrealized_conversion_cast %[[MUL]] : tensor<2x?xf32, #mfuse.symshape<["2", "s10"]>> to !torch.vtensor<[2,?],f32>
  // CHECK: return %[[OUT]] : !torch.vtensor<[2,?],f32>
  func.func @main(%arg0: !torch.int, %arg1: !torch.vtensor<[2,?],f32>, %arg2: !torch.vtensor<[2,?],f32>) -> !torch.vtensor<[2,?],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.symbolic_int "s10" {min_val = 2, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg1, [%0], affine_map<()[s10] -> (2, s10)> : !torch.vtensor<[2,?],f32>
    torch.bind_symbolic_shape %arg2, [%0], affine_map<()[s10] -> (2, s10)> : !torch.vtensor<[2,?],f32>
    %1 = torch.aten.mul.Tensor %arg2, %arg1 : !torch.vtensor<[2,?],f32>, !torch.vtensor<[2,?],f32> -> !torch.vtensor<[2,?],f32>
    torch.bind_symbolic_shape %1, [%0], affine_map<()[s10] -> (2, s10)> : !torch.vtensor<[2,?],f32>
    return %1 : !torch.vtensor<[2,?],f32>
  }
}
