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

"""UT for TorchFuseRoPE: bf16 skips ACL fusion; other dtypes fuse to npu_rotary_mul."""
# pylint: disable=protected-access

from __future__ import annotations

import textwrap

import pytest
import torch

from mfusion.torch import inductor as inductor_mod
from mfusion.torch._pipeline import PipelineRunner
from mfusion.torch.inductor import fuse_and_optimize
from ut_utils.mlir_checker import MlirChecker

ROPE_SHAPES = (
    (1, 8, 512, 128),
    (1, 32, 512, 128),
)

_TORCH_FUSION_PIPELINE = "builtin.module(canonicalize,torch-fusion,canonicalize)"


def _run_torch_fusion_only(mlir_text: str) -> str:
    """Run only the torch-fusion stage (canonicalize + torch-fusion + canonicalize)."""
    runner = PipelineRunner.from_torch_dialect_str(mlir_text)
    inductor_mod._run_composite_fusion_stage(
        runner,
        "Torch Fusion",
        _TORCH_FUSION_PIPELINE,
        inductor_mod._TORCH_FUSION_INTERNAL_PASSES,
        pre_canonicalize=True,
    )
    return str(runner.module)


def rope_decomposed_bf16(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rot = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rot * sin


def _make_rope_inputs(shape: tuple[int, ...], seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, dtype=torch.bfloat16, generator=gen)
    cos = torch.randn((1, 1, shape[-2], shape[-1]), dtype=torch.bfloat16, generator=gen)
    sin = torch.randn((1, 1, shape[-2], shape[-1]), dtype=torch.bfloat16, generator=gen)
    return x, cos, sin


def _rope_mlir_bf16_torch(h: int, seq: int, dim: int) -> str:
    return textwrap.dedent(
        f"""
        module {{
          func.func @main(%x: !torch.vtensor<[1,{h},{seq},{dim}],bf16>,
                          %cos: !torch.vtensor<[1,1,{seq},{dim}],bf16>,
                          %sin: !torch.vtensor<[1,1,{seq},{dim}],bf16>)
              -> !torch.vtensor<[1,{h},{seq},{dim}],bf16> {{
            %int3 = torch.constant.int 3
            %int0 = torch.constant.int 0
            %int64 = torch.constant.int 64
            %int_max = torch.constant.int 9223372036854775807
            %int1 = torch.constant.int 1
            %int_neg1 = torch.constant.int -1
            %x_left = torch.aten.slice.Tensor %x, %int3, %int0, %int64, %int1
              : !torch.vtensor<[1,{h},{seq},{dim}],bf16>, !torch.int, !torch.int, !torch.int, !torch.int
                -> !torch.vtensor<[1,{h},{seq},{dim // 2}],bf16>
            %x_right = torch.aten.slice.Tensor %x, %int3, %int64, %int_max, %int1
              : !torch.vtensor<[1,{h},{seq},{dim}],bf16>, !torch.int, !torch.int, !torch.int, !torch.int
                -> !torch.vtensor<[1,{h},{seq},{dim // 2}],bf16>
            %neg = torch.aten.neg %x_right : !torch.vtensor<[1,{h},{seq},{dim // 2}],bf16>
                -> !torch.vtensor<[1,{h},{seq},{dim // 2}],bf16>
            %list = torch.prim.ListConstruct %neg, %x_left
                : (!torch.vtensor<[1,{h},{seq},{dim // 2}],bf16>,
                   !torch.vtensor<[1,{h},{seq},{dim // 2}],bf16>) -> !torch.list<vtensor>
            %rot = torch.aten.cat %list, %int_neg1 : !torch.list<vtensor>, !torch.int
                -> !torch.vtensor<[1,{h},{seq},{dim}],bf16>
            %cos_mul = torch.aten.mul.Tensor %x, %cos
              : !torch.vtensor<[1,{h},{seq},{dim}],bf16>, !torch.vtensor<[1,1,{seq},{dim}],bf16>
                -> !torch.vtensor<[1,{h},{seq},{dim}],bf16>
            %sin_mul = torch.aten.mul.Tensor %rot, %sin
              : !torch.vtensor<[1,{h},{seq},{dim}],bf16>, !torch.vtensor<[1,1,{seq},{dim}],bf16>
                -> !torch.vtensor<[1,{h},{seq},{dim}],bf16>
            %out = torch.aten.add.Tensor %cos_mul, %sin_mul, %int1
              : !torch.vtensor<[1,{h},{seq},{dim}],bf16>, !torch.vtensor<[1,{h},{seq},{dim}],bf16>, !torch.int
                -> !torch.vtensor<[1,{h},{seq},{dim}],bf16>
            return %out : !torch.vtensor<[1,{h},{seq},{dim}],bf16>
          }}
        }}
        """
    )


def _assert_bf16_not_fused(out: str) -> None:
    assert "npu_rotary_mul" not in out, "bf16 RoPE must not fuse to npu_rotary_mul"
    assert "torch.aten.add.Tensor" in out
    assert "torch.aten.mul.Tensor" in out


MLIR_ROPE_BF16_PIPELINE = textwrap.dedent(
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


MLIR_ROPE_F16 = textwrap.dedent(
    """
    module {
      func.func @main(%x: !torch.vtensor<[1,8,512,128],f16>,
                      %cos: !torch.vtensor<[1,1,512,128],f16>,
                      %sin: !torch.vtensor<[1,1,512,128],f16>) -> !torch.vtensor<[1,8,512,128],f16> {
        %int3 = torch.constant.int 3
        %int0 = torch.constant.int 0
        %int64 = torch.constant.int 64
        %int_max = torch.constant.int 9223372036854775807
        %int1 = torch.constant.int 1
        %int_neg1 = torch.constant.int -1
        %x_left = torch.aten.slice.Tensor %x, %int3, %int0, %int64, %int1
          : !torch.vtensor<[1,8,512,128],f16>, !torch.int, !torch.int, !torch.int, !torch.int
            -> !torch.vtensor<[1,8,512,64],f16>
        %x_right = torch.aten.slice.Tensor %x, %int3, %int64, %int_max, %int1
          : !torch.vtensor<[1,8,512,128],f16>, !torch.int, !torch.int, !torch.int, !torch.int
            -> !torch.vtensor<[1,8,512,64],f16>
        %neg = torch.aten.neg %x_right : !torch.vtensor<[1,8,512,64],f16> -> !torch.vtensor<[1,8,512,64],f16>
        %list = torch.prim.ListConstruct %neg, %x_left
          : (!torch.vtensor<[1,8,512,64],f16>, !torch.vtensor<[1,8,512,64],f16>) -> !torch.list<vtensor>
        %rot = torch.aten.cat %list, %int_neg1 : !torch.list<vtensor>, !torch.int -> !torch.vtensor<[1,8,512,128],f16>
        %cos_mul = torch.aten.mul.Tensor %x, %cos
          : !torch.vtensor<[1,8,512,128],f16>, !torch.vtensor<[1,1,512,128],f16> -> !torch.vtensor<[1,8,512,128],f16>
        %sin_mul = torch.aten.mul.Tensor %rot, %sin
          : !torch.vtensor<[1,8,512,128],f16>, !torch.vtensor<[1,1,512,128],f16> -> !torch.vtensor<[1,8,512,128],f16>
        %out = torch.aten.add.Tensor %cos_mul, %sin_mul, %int1
          : !torch.vtensor<[1,8,512,128],f16>, !torch.vtensor<[1,8,512,128],f16>, !torch.int
            -> !torch.vtensor<[1,8,512,128],f16>
        return %out : !torch.vtensor<[1,8,512,128],f16>
      }
    }
    """
)


# --- compile path (mfusion / inductor) ---


def test_torch_fuse_rope_bf16_full_pipeline():
    result = fuse_and_optimize(MLIR_ROPE_BF16_PIPELINE)
    checker = MlirChecker.parse_torch_module(result)
    assert not checker.check_text_contains("npu_rotary_mul"), (
        checker.error or "bf16 full pipeline must not use npu_rotary_mul"
    )
    assert checker.check_text_contains("torch.aten.slice.Tensor"), (
        checker.error or "bf16 should keep decomposed slice"
    )


def test_torch_fuse_rope_bf16_torch_fusion_stage():
    result = _run_torch_fusion_only(MLIR_ROPE_BF16_PIPELINE)
    _assert_bf16_not_fused(result)


def test_torch_fuse_rope_f16_fuses_to_npu_rotary():
    result = _run_torch_fusion_only(MLIR_ROPE_F16)
    checker = MlirChecker.parse_torch_module(result)
    assert checker.check_text_contains('torch.operator "torch.npu.npu_rotary_mul"'), (
        checker.error or "f16 should fuse to npu_rotary_mul"
    )
    assert not checker.check_text_contains("torch.aten.add.Tensor"), (
        checker.error or "fused f16 graph should not keep aten.add"
    )


@pytest.mark.parametrize("shape", ROPE_SHAPES)
def test_torch_fusion_bf16_mlir_no_npu_rotary_mul(shape: tuple[int, ...]) -> None:
    _, h, seq, dim = shape
    out = _run_torch_fusion_only(_rope_mlir_bf16_torch(h, seq, dim))
    _assert_bf16_not_fused(out)


# --- NPU numeric guard (why bf16 skips fusion) ---


@pytest.mark.parametrize("shape", ROPE_SHAPES)
def test_npu_rotary_mul_acl_vs_decomposed(shape: tuple[int, ...]) -> None:
    """Sentinel: ACL kernel differs from eager bf16; revisit TorchFuseRoPE if this fails."""
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("NPU not available")

    x, cos, sin = _make_rope_inputs(shape, seed=42)
    ref = rope_decomposed_bf16(x, cos, sin)
    acl = torch.ops.npu.npu_rotary_mul(x.npu(), cos.npu(), sin.npu()).cpu()  # type: ignore[attr-defined]

    max_diff = (ref.float() - acl.float()).abs().max().item()
    assert max_diff > 0.001, (
        f"ACL matches decomposed bf16 (max_diff={max_diff}); revisit bf16 skip if ACL was fixed"
    )


@pytest.mark.parametrize("shape", ROPE_SHAPES)
def test_rope_decomposed_is_deterministic(shape: tuple[int, ...]) -> None:
    x, cos, sin = _make_rope_inputs(shape, seed=42)
    assert torch.equal(rope_decomposed_bf16(x, cos, sin), rope_decomposed_bf16(x, cos, sin))
