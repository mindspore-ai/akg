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
"""Regression tests for benchmark_lite submission runner correctness checks."""

import importlib.util
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_BENCH_PATH = REPO_ROOT / "benchmark/akg_kernels_bench_lite/tools/run_bench.py"
BENCH_LITE_COMMON_PATH = REPO_ROOT / "examples/kernel_related/bench_lite_common.py"


def _load_run_bench():
    spec = importlib.util.spec_from_file_location("bench_lite_run_bench_test", RUN_BENCH_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_bench_lite_common():
    spec = importlib.util.spec_from_file_location(
        "bench_lite_common_test", BENCH_LITE_COMMON_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_correctness_rejects_solution_nan():
    run_bench = _load_run_bench()

    result = run_bench.check_correctness(
        torch.zeros(2),
        torch.tensor([float("nan"), 0.0]),
        rtol=1e-2,
        atol=1e-2,
    )

    assert result["correct"] is False
    assert math.isinf(result["max_abs_diff"])
    assert math.isinf(result["max_rel_diff"])
    assert "solution 包含 NaN/Inf" in result["detail"]


def test_common_check_correctness_rejects_solution_nan():
    common = _load_bench_lite_common()

    result = common._check_correctness(
        torch.zeros(2),
        torch.tensor([float("nan"), 0.0]),
        rtol=1e-2,
        atol=1e-2,
    )

    assert result["correct"] is False
    assert math.isinf(result["max_abs_diff"])
    assert math.isinf(result["max_rel_diff"])
    assert "solution contains NaN/Inf" in result["detail"]


def test_run_single_case_replays_seed_for_stateful_inputs(tmp_path, monkeypatch):
    run_bench = _load_run_bench()
    monkeypatch.setattr(run_bench, "_get_device", lambda: "cpu")

    ref_path = tmp_path / "stateful.py"
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

    result = run_bench.run_single_case(
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

    assert result["status"] == "pass"
    assert result["correctness"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["max_rel_diff"] == 0.0


def test_common_eval_single_case_replays_seed_for_stateful_inputs(tmp_path, monkeypatch):
    common = _load_bench_lite_common()
    monkeypatch.setattr(common, "_get_torch_device", lambda _backend=None: "cpu")

    ref_path = tmp_path / "stateful.py"
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

    result = common._eval_single_case_inner(
        ref_path=ref_path,
        sol_path=sol_path,
        tier="t3",
        rtol=1e-2,
        atol=1e-2,
        warmup_runs=0,
        iterations=1,
        num_trials=1,
        backend="cpu",
    )

    assert result["status"] == "pass"
    assert result["correctness"] is True
    assert result["max_abs_diff"] == 0.0
    assert result["max_rel_diff"] == 0.0
