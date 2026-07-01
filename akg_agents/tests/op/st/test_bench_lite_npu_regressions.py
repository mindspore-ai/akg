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
"""NPU regression tests for benchmark_lite secondary correctness checks."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch


pytestmark = [
    pytest.mark.level0,
    pytest.mark.torch,
    pytest.mark.triton,
    pytest.mark.ascend,
    pytest.mark.ascend910b4,
]


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_BENCH_PATH = REPO_ROOT / "benchmark/akg_kernels_bench_lite/tools/run_bench.py"
BENCH_LITE_COMMON_PATH = REPO_ROOT / "examples/kernel_related/bench_lite_common.py"


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_npu():
    pytest.importorskip("torch_npu")
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None or not npu_mod.is_available():
        pytest.skip("torch.npu is not available")
    npu_mod.set_device(int(os.environ.get("DEVICE_ID", "0")))


def _write_stateful_case(tmp_path: Path) -> tuple[Path, Path]:
    ref_path = tmp_path / "stateful_ref.py"
    sol_path = tmp_path / "stateful_sol.py"
    ref_path.write_text(
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        out = x * 2.0
        x.add_(10.0)
        return out


def get_inputs():
    return [torch.randn(8, dtype=torch.float32)]


def get_init_inputs():
    return []
""",
        encoding="utf-8",
    )
    sol_path.write_text(
        """
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, x):
        out = x * 2.0
        x.add_(10.0)
        return out
""",
        encoding="utf-8",
    )
    return ref_path, sol_path


def _write_nan_case(tmp_path: Path) -> tuple[Path, Path]:
    ref_path = tmp_path / "nan_ref.py"
    sol_path = tmp_path / "nan_sol.py"
    ref_path.write_text(
        """
import torch
import torch.nn as nn


class Model(nn.Module):
    def forward(self, x):
        return x * 2.0


def get_inputs():
    return [torch.randn(8, dtype=torch.float32)]


def get_init_inputs():
    return []
""",
        encoding="utf-8",
    )
    sol_path.write_text(
        """
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def forward(self, x):
        return torch.full_like(x, float("nan"))
""",
        encoding="utf-8",
    )
    return ref_path, sol_path


def _run_tools_case(run_bench, ref_path: Path, sol_path: Path) -> dict:
    return run_bench.run_single_case(
        ref_path=ref_path,
        sol_path=sol_path,
        tier="t3",
        rtol=1e-2,
        atol=1e-2,
        warmup_runs=0,
        iterations=1,
        num_trials=1,
        timeout=30,
    )


def _run_common_case(common, ref_path: Path, sol_path: Path) -> dict:
    return common._eval_single_case_inner(
        ref_path=ref_path,
        sol_path=sol_path,
        tier="t3",
        rtol=1e-2,
        atol=1e-2,
        warmup_runs=0,
        iterations=1,
        num_trials=1,
        backend="npu",
    )


def test_bench_lite_stateful_inputs_are_isolated_on_npu(tmp_path, monkeypatch):
    _require_npu()
    run_bench = _load_module(RUN_BENCH_PATH, "bench_lite_run_bench_npu_stateful")
    common = _load_module(BENCH_LITE_COMMON_PATH, "bench_lite_common_npu_stateful")
    monkeypatch.setattr(run_bench, "_get_device", lambda: "npu")

    ref_path, sol_path = _write_stateful_case(tmp_path)

    tools_result = _run_tools_case(run_bench, ref_path, sol_path)
    assert tools_result["status"] == "pass"
    assert tools_result["correctness"] is True
    assert tools_result["max_abs_diff"] == 0.0
    assert tools_result["max_rel_diff"] == 0.0

    common_result = _run_common_case(common, ref_path, sol_path)
    assert common_result["status"] == "pass"
    assert common_result["correctness"] is True
    assert common_result["max_abs_diff"] == 0.0
    assert common_result["max_rel_diff"] == 0.0


def test_bench_lite_nan_outputs_fail_closed_on_npu(tmp_path, monkeypatch):
    _require_npu()
    run_bench = _load_module(RUN_BENCH_PATH, "bench_lite_run_bench_npu_nan")
    common = _load_module(BENCH_LITE_COMMON_PATH, "bench_lite_common_npu_nan")
    monkeypatch.setattr(run_bench, "_get_device", lambda: "npu")

    ref_path, sol_path = _write_nan_case(tmp_path)

    tools_result = _run_tools_case(run_bench, ref_path, sol_path)
    assert tools_result["status"] == "fail"
    assert tools_result["correctness"] is False
    assert "NaN/Inf" in tools_result["correctness_detail"]

    common_result = _run_common_case(common, ref_path, sol_path)
    assert common_result["status"] == "fail"
    assert common_result["correctness"] is False
    assert "NaN/Inf" in common_result["correctness_detail"]
