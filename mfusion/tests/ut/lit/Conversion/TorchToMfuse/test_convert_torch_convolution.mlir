// RUN: mfusion-opt %s --convert-torch-to-mfuse --convert-torch-symbol-to-mfuse --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @conv_stride2_no_bias
// CHECK: mfuse.aclnn.conv2d
// CHECK-DAG: stride = [2, 2]
// CHECK-NOT: torch.aten.convolution
func.func @conv_stride2_no_bias(%input: !torch.vtensor<[1,1,8,8],f32>, %weight: !torch.vtensor<[1,1,3,3],f32>)
    -> !torch.vtensor<[1,1,3,3],f32> {
  %int2 = torch.constant.int 2
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %stride = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %outpad = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %groups = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.aten.convolution %input, %weight, %none, %stride, %padding, %dilation, %false, %outpad, %groups
    : !torch.vtensor<[1,1,8,8],f32>, !torch.vtensor<[1,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>,
      !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,3,3],f32>
  return %0 : !torch.vtensor<[1,1,3,3],f32>
}

// CHECK-LABEL: func.func @conv_narrow_still_meta_conv2d
// CHECK: mfuse.aclnn.conv2d
// CHECK-NOT: stride = [2, 2]
func.func @conv_narrow_still_meta_conv2d(%input: !torch.vtensor<[1,1,8,8],f32>, %weight: !torch.vtensor<[1,1,3,3],f32>)
    -> !torch.vtensor<[1,1,6,6],f32> {
  %int1 = torch.constant.int 1
  %int0 = torch.constant.int 0
  %stride = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %padding = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %dilation = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
  %outpad = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
  %false = torch.constant.bool false
  %groups = torch.constant.int 1
  %none = torch.constant.none
  %0 = torch.aten.convolution %input, %weight, %none, %stride, %padding, %dilation, %false, %outpad, %groups
    : !torch.vtensor<[1,1,8,8],f32>, !torch.vtensor<[1,1,3,3],f32>, !torch.none, !torch.list<int>, !torch.list<int>,
      !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,1,6,6],f32>
  return %0 : !torch.vtensor<[1,1,6,6],f32>
}
