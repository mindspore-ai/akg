// RUN: not mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse 2>&1 | FileCheck %s --check-prefix=CONFLICT-SYMBOL

module {
  func.func @static_dim_bound_to_symbol(%arg0: !torch.vtensor<[1,6,37],f32>, %arg1: !torch.vtensor<[1,6,37],f32>)
      -> !torch.vtensor<[1,6,37],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %s0 = torch.symbolic_int "s0" {min_val = 2, max_val = 9223372036854775807} : !torch.int
    torch.bind_symbolic_shape %arg0, [%s0], affine_map<()[s0] -> (1, s0, 37)> : !torch.vtensor<[1,6,37],f32>
    %0 = torch.aten.mul.Tensor %arg0, %arg1 : !torch.vtensor<[1,6,37],f32>, !torch.vtensor<[1,6,37],f32> -> !torch.vtensor<[1,6,37],f32>
    return %0 : !torch.vtensor<[1,6,37],f32>
  }
}

// CONFLICT-SYMBOL: static tensor dimension 1 is 6 but bind_symbolic_shape expression is non-constant: s0
