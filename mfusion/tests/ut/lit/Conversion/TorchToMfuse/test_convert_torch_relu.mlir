// RUN: mfusion-opt %s --convert-torch-to-mfuse --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @convert_relu_basic
// CHECK: mfuse.relu
// CHECK-NOT: torch.aten.relu
func.func @convert_relu_basic(%input: !torch.vtensor<[2,3],f32>) -> !torch.vtensor<[2,3],f32> {
  %relu = torch.aten.relu %input : !torch.vtensor<[2,3],f32> -> !torch.vtensor<[2,3],f32>
  return %relu : !torch.vtensor<[2,3],f32>
}

