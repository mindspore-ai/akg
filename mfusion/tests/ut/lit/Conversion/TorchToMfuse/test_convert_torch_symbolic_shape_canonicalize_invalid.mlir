// RUN: not mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse 2>&1 | FileCheck %s --check-prefix=CONFLICT-CONST

module {
  func.func @conflicting_static_constant(%arg0: !torch.vtensor<[1,1,5,37],f32>, %arg1: !torch.vtensor<[1,1,5,37],f32>)
      -> !torch.vtensor<[1,1,5,37],f32> attributes {torch.assume_strict_symbolic_shapes} {
    torch.bind_symbolic_shape %arg0, [], affine_map<() -> (1, 1, 6, 37)> : !torch.vtensor<[1,1,5,37],f32>
    %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[1,1,5,37],f32>, !torch.vtensor<[1,1,5,37],f32> -> !torch.vtensor<[1,1,5,37],f32>
    return %0 : !torch.vtensor<[1,1,5,37],f32>
  }
}

// CONFLICT-CONST: static tensor dimension 2 is 5 but bind_symbolic_shape expression is 6
