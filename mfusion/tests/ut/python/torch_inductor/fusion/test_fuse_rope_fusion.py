#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UT for Torch FuseRoPE: fuse RoPE pattern to torch.npu.npu_rotary_mul before Torch->Mfuse."""

import textwrap

from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker


MLIR_ROPE_BASIC = textwrap.dedent(
    """
    module {
      func.func @main(%x: tensor<1x8x512x128xbf16>,
                      %cos: tensor<1x1x512x128xbf16>,
                      %sin: tensor<1x1x512x128xbf16>) -> tensor<1x8x512x128xbf16> {
        %x_v = builtin.unrealized_conversion_cast %x
          : tensor<1x8x512x128xbf16> to !torch.vtensor<[1,8,512,128],bf16>
        %cos_v = builtin.unrealized_conversion_cast %cos
          : tensor<1x1x512x128xbf16> to !torch.vtensor<[1,1,512,128],bf16>
        %sin_v = builtin.unrealized_conversion_cast %sin
          : tensor<1x1x512x128xbf16> to !torch.vtensor<[1,1,512,128],bf16>

        %int3 = torch.constant.int 3
        %int0 = torch.constant.int 0
        %int64 = torch.constant.int 64
        %int_max = torch.constant.int 9223372036854775807
        %int1 = torch.constant.int 1
        %int_neg1 = torch.constant.int -1

        %x_left = torch.aten.slice.Tensor %x_v, %int3, %int0, %int64, %int1
          : !torch.vtensor<[1,8,512,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int
            -> !torch.vtensor<[1,8,512,64],bf16>
        %x_right = torch.aten.slice.Tensor %x_v, %int3, %int64, %int_max, %int1
          : !torch.vtensor<[1,8,512,128],bf16>, !torch.int, !torch.int, !torch.int, !torch.int
            -> !torch.vtensor<[1,8,512,64],bf16>

        %neg_v = torch.aten.neg %x_right
          : !torch.vtensor<[1,8,512,64],bf16> -> !torch.vtensor<[1,8,512,64],bf16>

        %list = torch.prim.ListConstruct %neg_v, %x_left
          : (!torch.vtensor<[1,8,512,64],bf16>, !torch.vtensor<[1,8,512,64],bf16>) -> !torch.list<vtensor>
        %rot_v = torch.aten.cat %list, %int_neg1
          : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,8,512,128],bf16>

        %cos_mul = torch.aten.mul.Tensor %x_v, %cos_v
          : !torch.vtensor<[1,8,512,128],bf16>, !torch.vtensor<[1,1,512,128],bf16>
            -> !torch.vtensor<[1,8,512,128],bf16>
        %sin_mul = torch.aten.mul.Tensor %rot_v, %sin_v
          : !torch.vtensor<[1,8,512,128],bf16>, !torch.vtensor<[1,1,512,128],bf16>
            -> !torch.vtensor<[1,8,512,128],bf16>
        %out_v = torch.aten.add.Tensor %cos_mul, %sin_mul, %int1
          : !torch.vtensor<[1,8,512,128],bf16>, !torch.vtensor<[1,8,512,128],bf16>, !torch.int
            -> !torch.vtensor<[1,8,512,128],bf16>

        %out = builtin.unrealized_conversion_cast %out_v
          : !torch.vtensor<[1,8,512,128],bf16> to tensor<1x8x512x128xbf16>
        return %out : tensor<1x8x512x128xbf16>
      }
    }
    """
)


def test_torch_fuse_rope_basic():
    result = fuse_and_optimize(MLIR_ROPE_BASIC)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains('torch.operator "torch.npu.npu_rotary_mul"'), (
        checker.error or "Expected torch.npu.npu_rotary_mul after torch-fuse-rope"
    )

