// RUN: mfusion-opt %s --convert-torch-to-mfuse --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @convert_prim_num_to_tensor_scalar_no_const_fold
// CHECK: torch.prim.NumToTensor.Scalar
// CHECK-NOT: torch.vtensor.literal
func.func @convert_prim_num_to_tensor_scalar_no_const_fold() -> !torch.vtensor<[],si64> {
  %c2 = torch.constant.int 2
  %0 = torch.prim.NumToTensor.Scalar %c2 : !torch.int -> !torch.vtensor<[],si64>
  return %0 : !torch.vtensor<[],si64>
}

