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
import types
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


def _write_cached_output_case(tmp_path):
    ref_path = tmp_path / "cached_output.py"
    sol_path = tmp_path / "cached_output_sol.py"

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
    def __init__(self):
        super().__init__()
        self._out = None

    def forward(self, x):
        if self._out is None:
            self._out = x * 2.0
        return self._out
""",
        encoding="utf-8",
    )
    return ref_path, sol_path


def test_check_correctness_rejects_solution_nan():
    run_bench = _load_run_bench()

    result = run_bench.check_correctness(
        torch.zeros(2),
        torch.tensor([float("nan"), 0.0]),
    )

    assert result["correct"] is False
    assert math.isinf(result["max_abs_diff"])
    assert math.isinf(result["max_rel_diff"])
    assert "solution contains NaN/Inf" in result["detail"]


def test_common_check_correctness_rejects_solution_nan():
    common = _load_bench_lite_common()

    result = common._check_correctness(
        torch.zeros(2),
        torch.tensor([float("nan"), 0.0]),
    )

    assert result["correct"] is False
    assert math.isinf(result["max_abs_diff"])
    assert math.isinf(result["max_rel_diff"])
    assert "solution contains NaN/Inf" in result["detail"]


def test_bench_lite_paths_use_elementwise_allclose_semantics():
    run_bench = _load_run_bench()
    common = _load_bench_lite_common()

    near_zero_ref = torch.tensor([5.117739e-08], dtype=torch.float32)
    near_zero_sol = torch.tensor([8.663523e-08], dtype=torch.float32)

    # The first element needs atol and the second needs rtol. Both should pass
    # even though the global max absolute and relative diagnostics exceed 1e-2.
    mixed_ref = torch.tensor([0.0, 1000.0], dtype=torch.float32)
    mixed_sol = torch.tensor([0.005, 1005.0], dtype=torch.float32)

    cases = (
        (near_zero_ref, near_zero_sol, True),
        (mixed_ref, mixed_sol, True),
        (torch.tensor([0.0]), torch.tensor([0.011]), False),
        (torch.tensor([1000.0]), torch.tensor([1011.0]), False),
    )
    for ref, sol, expected in cases:
        standalone = run_bench.check_correctness(ref, sol, rtol=1e-2, atol=1e-2)
        generated = common._check_correctness(ref, sol, rtol=1e-2, atol=1e-2)
        assert standalone["correct"] is expected
        assert generated["correct"] is expected
        assert standalone["correct"] is generated["correct"]

    near_zero_result = run_bench.check_correctness(near_zero_ref, near_zero_sol)
    assert near_zero_result["correct"] is True
    assert near_zero_result["max_rel_diff"] > 0.5

    mixed_result = run_bench.check_correctness(mixed_ref, mixed_sol)
    assert mixed_result["correct"] is True
    assert mixed_result["max_abs_diff"] > 1e-2
    assert mixed_result["max_rel_diff"] > 1e-2


def test_bench_lite_paths_reject_matching_non_finite_outputs():
    run_bench = _load_run_bench()
    common = _load_bench_lite_common()

    for value in (float("nan"), float("inf"), float("-inf")):
        ref = torch.tensor([value], dtype=torch.float32)
        sol = ref.clone()
        standalone = run_bench.check_correctness(ref, sol)
        generated = common._check_correctness(ref, sol)
        assert standalone["correct"] is False
        assert generated["correct"] is False
        assert "reference contains NaN/Inf" in standalone["detail"]
        assert "reference contains NaN/Inf" in generated["detail"]


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


def test_run_single_case_rejects_cached_output_across_random_trials(tmp_path, monkeypatch):
    run_bench = _load_run_bench()
    monkeypatch.setattr(run_bench, "_get_device", lambda: "cpu")
    ref_path, sol_path = _write_cached_output_case(tmp_path)

    result = run_bench.run_single_case(
        ref_path=ref_path,
        sol_path=sol_path,
        tier="t1",
        rtol=1e-2,
        atol=1e-2,
        warmup_runs=0,
        iterations=1,
        num_trials=1,
        timeout=30,
    )

    assert result["status"] == "fail"
    assert result["correctness"] is False
    assert "trial=2/3" in result["correctness_detail"]


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


def test_common_eval_single_case_rejects_cached_output_across_random_trials(tmp_path, monkeypatch):
    common = _load_bench_lite_common()
    monkeypatch.setattr(common, "_get_torch_device", lambda _backend=None: "cpu")
    ref_path, sol_path = _write_cached_output_case(tmp_path)

    result = common._eval_single_case_inner(
        ref_path=ref_path,
        sol_path=sol_path,
        tier="t1",
        rtol=1e-2,
        atol=1e-2,
        warmup_runs=0,
        iterations=1,
        num_trials=1,
        backend="cpu",
    )

    assert result["status"] == "fail"
    assert result["correctness"] is False
    assert "trial=2/3" in result["correctness_detail"]


def test_submission_runner_uses_profiler_npu_for_npu_latency(monkeypatch):
    run_bench = _load_run_bench()
    monkeypatch.setattr(run_bench, "_get_device", lambda: "npu")

    calls = []
    samples_us = iter([120.0, 100.0, 110.0])
    profiler_module = types.ModuleType("akg_agents.op.verifier.profiler")

    def fake_profiler_npu(fn, **kwargs):
        calls.append(kwargs)
        assert fn() == 3
        return next(samples_us)

    profiler_module.profiler_npu = fake_profiler_npu
    monkeypatch.setitem(sys.modules, "akg_agents.op.verifier.profiler", profiler_module)

    result = run_bench.measure_latency(
        lambda value: value + 1,
        [2],
        warmup_runs=7,
        iterations=11,
        num_trials=3,
    )

    assert result["timing_method"] == "profiler_npu"
    assert math.isclose(result["median_ms"], 0.11)
    assert len(calls) == 3
    assert all(call["warmup"] == 7 for call in calls)
    assert all(call["active"] == 11 for call in calls)
    assert all(call["clear_l2_cache"] is True for call in calls)
    assert all(call["dsl"] == "triton_ascend" for call in calls)


def test_common_runner_uses_profiler_npu_for_npu_latency(monkeypatch):
    common = _load_bench_lite_common()
    monkeypatch.setattr(common, "_get_torch_device", lambda _backend=None: "npu")

    def fail_if_cuda_timing_is_probed(_device):
        raise AssertionError("CUDA timing must not be probed on NPU")

    monkeypatch.setattr(common, "_use_cuda_events", fail_if_cuda_timing_is_probed)

    calls = []
    samples_us = iter([220.0, 180.0, 200.0])
    profiler_module = types.ModuleType("akg_agents.op.verifier.profiler")

    def fake_profiler_npu(fn, **kwargs):
        calls.append(kwargs)
        assert fn() == 5
        return next(samples_us)

    profiler_module.profiler_npu = fake_profiler_npu
    monkeypatch.setitem(sys.modules, "akg_agents.op.verifier.profiler", profiler_module)

    result = common._measure_latency(
        lambda value: value + 1,
        [4],
        warmup_runs=5,
        iterations=13,
        num_trials=3,
        backend="npu",
    )

    assert result["timing_method"] == "profiler_npu"
    assert math.isclose(result["median_ms"], 0.2)
    assert len(calls) == 3
    assert all(call["warmup"] == 5 for call in calls)
    assert all(call["active"] == 13 for call in calls)
    assert all(call["clear_l2_cache"] is True for call in calls)
    assert all(call["dsl"] == "triton_ascend" for call in calls)
