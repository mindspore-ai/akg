#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamic Tune ST.

The file is intentionally a pytest test module, not a CLI wrapper. It covers:
1. convert: postprocess a raw case into ModelNew contract output
2. run: execute/tune/verify an already converted case
3. convert and run: postprocess then execute the produced implementation
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

ST_DYNAMIC_TUNE_ROOT = Path(__file__).resolve().parent
ORIGINAL_CASES_ROOT = ST_DYNAMIC_TUNE_ROOT / "cases" / "original"
CONVERTED_CASES_ROOT = ST_DYNAMIC_TUNE_ROOT / "cases" / "converted"
AKG_AGENTS_ROOT = Path(__file__).resolve().parents[4]
PYTHON_ROOT = AKG_AGENTS_ROOT / "python"
for path in (str(PYTHON_ROOT), str(AKG_AGENTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from akg_agents.op.dynamic_tune.cases import postprocess_case, run_case
from akg_agents.op.dynamic_tune.cases.case import _CaseSpec
from akg_agents.op.dynamic_tune.cases.contract import _ModelNewContractValidator


CASE_NAMES = tuple(
    path.name for path in sorted(ORIGINAL_CASES_ROOT.iterdir()) if (path / "sample.json").is_file()
)
# Default op-st: level0 + torch/triton/ascend/arch (see other op/st tests). Heavy cases use level2 only.
ST_RUN_DEFAULT_CASE_NAMES = ("matmul", "relu", "rms_norm")
_OTHER_RUN_CASE_NAMES = tuple(n for n in CASE_NAMES if n not in ST_RUN_DEFAULT_CASE_NAMES)
assert set(ST_RUN_DEFAULT_CASE_NAMES) <= set(CASE_NAMES), (
    f"ST_RUN_DEFAULT_CASE_NAMES must ⊆ CASE_NAMES; missing "
    f"{set(ST_RUN_DEFAULT_CASE_NAMES) - set(CASE_NAMES)}"
)
DYNAMIC_TUNE_IMPL_CODE_ENV = "DYNAMIC_TUNE_IMPL_CODE"


def _case_dir(case_name: str) -> Path:
    return ORIGINAL_CASES_ROOT / case_name


def _converted_impl_path(case_name: str) -> Path:
    return CONVERTED_CASES_ROOT / case_name / "impl.py"


def _assert_modelnew_contract(impl_path: Path) -> None:
    errors = _ModelNewContractValidator.validate_code(
        impl_path.read_text(encoding="utf-8"),
        output_path=str(impl_path),
    )
    assert not errors, "后处理产物未通过 contract 校验:\n- " + "\n- ".join(errors)


def _require_opencode() -> None:
    if shutil.which("opencode") is None:
        pytest.skip("opencode executable is unavailable; skipping dynamic_tune convert ST")


def _require_npu() -> str:
    try:
        import torch  # type: ignore
        import torch_npu  # noqa: F401
    except ImportError:
        pytest.skip("torch_npu is unavailable; skipping dynamic_tune run ST")
    if not torch.npu.is_available():  # type: ignore[attr-defined]
        pytest.skip("NPU is unavailable; skipping dynamic_tune run ST")
    return f"npu:{int(os.environ.get('DEVICE_ID', '0'))}"


def _convert_case(case_dir: Path, output_path: Path) -> Path:
    case_spec = _CaseSpec.from_case_dir(case_dir)
    postprocess_case(
        case_name=case_spec.name,
        raw_impl_path=case_spec.impl_path,
        base_path=case_spec.base_path,
        sample_path=case_spec.sample_path,
        output_path=output_path,
    )
    _assert_modelnew_contract(output_path)
    return output_path


def _run_converted_case(*, case_dir: Path, impl_path: Path, device: str, work_root: Path):
    return run_case(
        case_dir=case_dir,
        impl_path=impl_path,
        npu_device=device,
        cache_dir=work_root / "cache",
        work_dir=work_root / "work",
        artifacts_root=work_root / "artifacts",
        artifact_name=f"{case_dir.name}_pytest",
    )


def _run_checked_in_case(case_name: str, tmp_path: Path) -> None:
    device = _require_npu()
    impl_path = _converted_impl_path(case_name)
    _assert_modelnew_contract(impl_path)
    summary = _run_converted_case(
        case_dir=_case_dir(case_name),
        impl_path=impl_path,
        device=device,
        work_root=tmp_path / "run",
    )
    assert summary["verify_passed"] is True


@pytest.mark.level2
@pytest.mark.st
@pytest.mark.torch
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_convert(case_name: str, tmp_path: Path) -> None:
    """Convert a raw case, or validate an explicit converted output from env."""

    impl_from_env = os.environ.get(DYNAMIC_TUNE_IMPL_CODE_ENV)
    if impl_from_env:
        _assert_modelnew_contract(Path(impl_from_env).expanduser().resolve())
        return

    _require_opencode()
    output_path = tmp_path / "converted" / case_name / "impl.py"
    _convert_case(_case_dir(case_name), output_path)


@pytest.mark.level0
@pytest.mark.st
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.parametrize("case_name", ST_RUN_DEFAULT_CASE_NAMES)
def test_run(case_name: str, tmp_path: Path) -> None:
    """Run/tune/verify checked-in converted cases selected for default op-st."""

    _run_checked_in_case(case_name, tmp_path)


@pytest.mark.level2
@pytest.mark.st
@pytest.mark.torch
@pytest.mark.parametrize("case_name", _OTHER_RUN_CASE_NAMES)
def test_run_level2(case_name: str, tmp_path: Path) -> None:
    """Other cases: only when level2 is included in -m."""

    _run_checked_in_case(case_name, tmp_path)


@pytest.mark.level2
@pytest.mark.st
@pytest.mark.torch
@pytest.mark.parametrize("case_name", CASE_NAMES)
def test_convert_and_run(case_name: str, tmp_path: Path) -> None:
    """Postprocess a raw case, then run/tune/verify that generated output."""

    _require_opencode()
    device = _require_npu()
    output_path = tmp_path / "converted" / case_name / "impl.py"
    _convert_case(_case_dir(case_name), output_path)
    summary = _run_converted_case(
        case_dir=_case_dir(case_name),
        impl_path=output_path,
        device=device,
        work_root=tmp_path / "convert_and_run",
    )
    assert summary["verify_passed"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
