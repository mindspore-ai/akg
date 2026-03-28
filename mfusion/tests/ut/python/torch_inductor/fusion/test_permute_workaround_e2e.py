# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end regression around permute / MatMul paths after MFusion workarounds.

DVM / NPU Inductor does not implement `aten.permute.default`; surfaced `torch.aten.permute`
in IR that goes through the DVM path can fail compilation.

- **`test_e2e_unaligned_mm_*`**: input is plain `torch.aten.mm` only (no layout op). Asserts
  the full `fuse_and_optimize` output contains **no** `aten.permute` (strict DVM-oriented guard).

- **`test_e2e_transpose_int_then_mm_*`**: input uses `torch.aten.transpose.int` + `mm`.
  MFusion lowers transpose to `mfuse.permute`; **convert-mfuse-to-torch currently reprints
  that as `torch.aten.permute`**, so we do **not** assert absence of permute here. This test
  only guards that the full pipeline completes and still exposes matmul semantics.

Related workaround context: no `ConvertAtenPermute` for direct `aten.permute`, `mfuse.permute`
not DVM-clusterable, no default `fuse-matmul-transpose-weight` in `mfuse-fusion`.
"""

import textwrap

from mfusion.torch.inductor import fuse_and_optimize

from ut_utils.mlir_checker import MlirChecker


def _assert_no_aten_permute_in_torch_output(ir: str) -> None:
    """Guard: MLIR must not contain aten.permute (Torch Inductor: aten.permute.default) on the DVM path."""
    if "aten.permute" in ir:
        raise AssertionError(
            "fuse_and_optimize output must not contain aten.permute "
            "(DVM: aten.permute.default not implemented). IR dump:\n" + ir
        )


# K=100: for f32, 100*4=400 bytes is not 512-byte aligned; historically tied to matmul/transpose paths.
MLIR_MM_UNALIGNED_K = textwrap.dedent(
    r"""
    module {
      func.func @main(%arg0: !torch.vtensor<[2,100],f32>, %arg1: !torch.vtensor<[100,8],f32>) -> !torch.vtensor<[2,8],f32> attributes {torch.assume_strict_symbolic_shapes} {
        %0 = torch.aten.mm %arg0, %arg1 : !torch.vtensor<[2,100],f32>, !torch.vtensor<[100,8],f32> -> !torch.vtensor<[2,8],f32>
        return %0 : !torch.vtensor<[2,8],f32>
      }
    }
    """
)

# Same shape as transpose+mm in test_batch_matmul_fusion.py: layout via transpose.int, not `torch.aten.permute`.
MLIR_TRANSPOSE_INT_THEN_MM = textwrap.dedent(
    r"""
    module {
      func.func @main(%arg0: !torch.vtensor<[2,4],f32>, %arg1: !torch.vtensor<[2,8],f32>) -> !torch.vtensor<[4,8],f32> attributes {torch.assume_strict_symbolic_shapes} {
        %int0 = torch.constant.int 0
        %int1 = torch.constant.int 1
        %0 = torch.aten.transpose.int %arg0, %int0, %int1 : !torch.vtensor<[2,4],f32>, !torch.int, !torch.int -> !torch.vtensor<[4,2],f32>
        %1 = torch.aten.mm %0, %arg1 : !torch.vtensor<[4,2],f32>, !torch.vtensor<[2,8],f32> -> !torch.vtensor<[4,8],f32>
        return %1 : !torch.vtensor<[4,8],f32>
      }
    }
    """
)


def test_e2e_unaligned_mm_full_pipeline_after_permute_workaround():
    """Pure mm with K not 512-byte aligned: full pipeline succeeds; no aten.permute in output."""
    result = fuse_and_optimize(MLIR_MM_UNALIGNED_K)
    assert result and "module" in result
    _assert_no_aten_permute_in_torch_output(result)
    checker = MlirChecker.parse_torch_module(result)
    assert checker is not None, "pipeline output should parse as a Torch module"
    lowered = result.lower()
    assert "matmul" in lowered or "mm" in lowered, (
        "expected matmul/mm semantics in output, got:\n" + result
    )


def test_e2e_transpose_int_then_mm_full_pipeline_after_permute_workaround():
    """transpose.int + mm: full pipeline completes; Mfuse->Torch may emit torch.aten.permute."""
    result = fuse_and_optimize(MLIR_TRANSPOSE_INT_THEN_MM)
    assert result and "module" in result
    checker = MlirChecker.parse_torch_module(result)
    assert checker is not None, "pipeline output should parse as a Torch module"
    assert checker.check_has_op("torch.aten.mm") or checker.check_has_op("torch.aten.matmul"), (
        checker.error or "expected torch.aten.mm or torch.aten.matmul in output"
    )
