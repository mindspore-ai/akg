// RUN: mfusion-opt %s --torch-fuse-rope | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK: torch.operator "torch.npu.npu_rotary_mul"
// CHECK-NOT: torch.aten.add.Tensor
// CHECK-NOT: torch.aten.mul.Tensor

func.func @main(%x: !torch.vtensor<[1,8,512,128],bf16>,
                %cos: !torch.vtensor<[1,1,512,128],bf16>,
                %sin: !torch.vtensor<[1,1,512,128],bf16>) -> !torch.vtensor<[1,8,512,128],bf16> {
  %int3 = torch.constant.int 3
  %int0 = torch.constant.int 0
  %int64 = torch.constant.int 64
  %int_max = torch.constant.int 9223372036854775807
  %int1 = torch.constant.int 1
  %int_neg1 = torch.constant.int -1

  %x_left = torch.aten.slice.Tensor %x, %int3, %int0, %int64, %int1
    : !torch.vtensor<[1,8,512,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[1,8,512,64],bf16>
  %x_right = torch.aten.slice.Tensor %x, %int3, %int64, %int_max, %int1
    : !torch.vtensor<[1,8,512,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[1,8,512,64],bf16>
  %neg = torch.aten.neg %x_right
    : !torch.vtensor<[1,8,512,64],bf16> -> !torch.vtensor<[1,8,512,64],bf16>
  %list = torch.prim.ListConstruct %neg, %x_left
    : (!torch.vtensor<[1,8,512,64],bf16>, !torch.vtensor<[1,8,512,64],bf16>) -> !torch.list<vtensor>
  %rot = torch.aten.cat %list, %int_neg1
    : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,8,512,128],bf16>

  %cos_mul = torch.aten.mul.Tensor %x, %cos
    : !torch.vtensor<[1,8,512,128],bf16>, !torch.vtensor<[1,1,512,128],bf16>
      -> !torch.vtensor<[1,8,512,128],bf16>
  %sin_mul = torch.aten.mul.Tensor %rot, %sin
    : !torch.vtensor<[1,8,512,128],bf16>, !torch.vtensor<[1,1,512,128],bf16>
      -> !torch.vtensor<[1,8,512,128],bf16>
  %out = torch.aten.add.Tensor %cos_mul, %sin_mul, %int1
    : !torch.vtensor<[1,8,512,128],bf16>, !torch.vtensor<[1,8,512,128],bf16>, !torch.int
      -> !torch.vtensor<[1,8,512,128],bf16>
  return %out : !torch.vtensor<[1,8,512,128],bf16>
}

