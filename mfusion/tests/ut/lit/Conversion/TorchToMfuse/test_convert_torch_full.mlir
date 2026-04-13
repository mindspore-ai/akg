// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @convert_full_with_explicit_attrs
// CHECK: %[[FULL0:.*]] = mfuse.full
// CHECK-SAME: {device = "npu", dtype = 6 : i64, layout = 0 : i64, pin_memory = true}
// CHECK-NOT: torch.aten.full
func.func @convert_full_with_explicit_attrs() -> !torch.vtensor<[2,3],f32> {
  %c2 = torch.constant.int 2
  %c3 = torch.constant.int 3
  %size = torch.prim.ListConstruct %c2, %c3 : (!torch.int, !torch.int) -> !torch.list<int>
  %fill = torch.constant.float 3.500000e+00
  %dtype = torch.constant.int 6
  %layout = torch.constant.int 0
  %device = torch.constant.device "npu"
  %pin = torch.constant.bool true
  %full = torch.aten.full %size, %fill, %dtype, %layout, %device, %pin
      : !torch.list<int>, !torch.float, !torch.int, !torch.int, !torch.Device, !torch.bool
      -> !torch.vtensor<[2,3],f32>
  return %full : !torch.vtensor<[2,3],f32>
}

// CHECK-LABEL: func.func @convert_full_with_defaults
// CHECK: %[[FULL1:.*]] = mfuse.full
// CHECK-NOT: torch.aten.full
func.func @convert_full_with_defaults() -> !torch.vtensor<[4,5],si64> {
  %c4 = torch.constant.int 4
  %c5 = torch.constant.int 5
  %size = torch.prim.ListConstruct %c4, %c5 : (!torch.int, !torch.int) -> !torch.list<int>
  %fill = torch.constant.int 7
  %none = torch.constant.none
  %full = torch.aten.full %size, %fill, %none, %none, %none, %none
      : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none
      -> !torch.vtensor<[4,5],si64>
  return %full : !torch.vtensor<[4,5],si64>
}

// CHECK-LABEL: func.func @convert_full_with_variable_scalar
// CHECK: %[[FULL1:.*]] = mfuse.full
// CHECK-NOT: torch.aten.full
func.func @convert_full_with_variable_scalar(%arg0: !torch.int) -> !torch.vtensor<[4,5],si64> {
  %c4 = torch.constant.int 4
  %c5 = torch.constant.int 5
  %size = torch.prim.ListConstruct %c4, %c5 : (!torch.int, !torch.int) -> !torch.list<int>
  %none = torch.constant.none
  %full = torch.aten.full %size, %arg0, %none, %none, %none, %none
      : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none
      -> !torch.vtensor<[4,5],si64>
  return %full : !torch.vtensor<[4,5],si64>
}
