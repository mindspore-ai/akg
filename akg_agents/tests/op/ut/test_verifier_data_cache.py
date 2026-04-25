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

"""Unit tests for verifier-side persistent data cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from akg_agents.op.verifier.data_cache import (
    VerifierDataCacheConfig,
    build_baseline_cache_key,
    build_reference_cache_key,
    extract_baseline_time_us,
    read_baseline_result_from_cache,
    read_reference_data_from_cache,
    write_baseline_result_to_cache,
    write_reference_data_to_cache,
)
from akg_agents.op.verifier.kernel_verifier import KernelVerifier


FRAMEWORK_CODE = """
import torch


class Model:
    def __init__(self):
        pass

    def __call__(self, x):
        return x


def get_inputs():
    return [torch.tensor([1.0])]


def get_init_inputs():
    return []
"""

IMPL_CODE = """
class ModelNew:
    def __init__(self):
        pass

    def __call__(self, x):
        return x
"""


class DummyProfileWorker:
    def __init__(self):
        self.profile_settings = None

    async def profile(self, package_data, task_id, op_name, profile_settings):
        self.profile_settings = dict(profile_settings)
        return {
            "gen_time": 5.0,
            "base_time": profile_settings.get("override_base_time_us"),
            "speedup": 2.0,
            "roofline_time": None,
            "roofline_speedup": 0.0,
            "roofline": None,
            "artifacts": {},
        }


def _make_config(tmp_path: Path) -> dict:
    return {
        "log_dir": str(tmp_path / "logs"),
        "data_cache": {
            "enabled": True,
            "cache_dir": str(tmp_path / "verifier_cache"),
        },
    }


def _make_verifier(tmp_path: Path, worker=None) -> KernelVerifier:
    return KernelVerifier(
        op_name="relu",
        framework_code=FRAMEWORK_CODE,
        task_id="ut",
        framework="torch",
        dsl="triton_ascend",
        backend="ascend",
        arch="ascend910b4",
        impl_func_name="ModelNew",
        config=_make_config(tmp_path),
        worker=worker,
    )


def test_reference_cache_roundtrip(tmp_path):
    cfg = VerifierDataCacheConfig(enabled=True, cache_dir=str(tmp_path / "cache"))
    cache_key = build_reference_cache_key(
        op_name="relu",
        framework_code=FRAMEWORK_CODE,
        framework="torch",
        backend="ascend",
        arch="ascend910b4",
        bench_type="kernelbench",
    )

    reference_bytes = b"cached-reference-payload"
    write_reference_data_to_cache(
        cfg,
        op_name="relu",
        cache_key=cache_key,
        reference_data=reference_bytes,
        metadata={"save_inputs": True},
    )

    loaded = read_reference_data_from_cache(cfg, op_name="relu", cache_key=cache_key)
    assert loaded == reference_bytes


def test_baseline_cache_roundtrip(tmp_path):
    cfg = VerifierDataCacheConfig(enabled=True, cache_dir=str(tmp_path / "cache"))
    cache_key = build_baseline_cache_key(
        op_name="relu",
        framework_code=FRAMEWORK_CODE,
        framework="torch",
        backend="ascend",
        arch="ascend910b4",
        bench_type="kernelbench",
        warmup_times=5,
        run_times=50,
    )

    write_baseline_result_to_cache(
        cfg,
        op_name="relu",
        cache_key=cache_key,
        result_data={"avg_time_us": 12.5, "method": "unit_test"},
    )

    payload = read_baseline_result_from_cache(cfg, op_name="relu", cache_key=cache_key)
    assert payload is not None
    assert extract_baseline_time_us(payload) == 12.5
    assert payload["method"] == "unit_test"


@pytest.mark.asyncio
async def test_run_uses_cached_reference_data(monkeypatch, tmp_path):
    verifier = _make_verifier(tmp_path, worker=object())
    cfg = verifier._get_data_cache_config()
    cache_key = verifier._get_reference_cache_key()
    cached_reference = b"cached-pt-content"
    write_reference_data_to_cache(
        cfg,
        op_name=verifier.op_name,
        cache_key=cache_key,
        reference_data=cached_reference,
        metadata={"save_inputs": True},
    )

    captured = {}

    def fake_gen_verify_project(impl_code, verify_dir, device_id=0):
        captured["reference_data"] = verifier.config.get("reference_data")
        captured["use_reference_data"] = verifier.config.get("use_reference_data")
        captured["use_reference_inputs"] = verifier.config.get("use_reference_inputs")

    async def fake_run_verify(verify_dir, timeout=300, device_id=0):
        return True, "ok"

    monkeypatch.setattr(verifier, "gen_verify_project", fake_gen_verify_project)
    monkeypatch.setattr(verifier, "run_verify", fake_run_verify)

    success, log = await verifier.run({"coder_code": IMPL_CODE}, device_id=0)
    assert success is True
    assert log == "ok"
    assert captured["reference_data"] == cached_reference
    assert captured["use_reference_data"] is True
    assert captured["use_reference_inputs"] is True


@pytest.mark.asyncio
async def test_run_profile_uses_cached_baseline(monkeypatch, tmp_path):
    worker = DummyProfileWorker()
    verifier = _make_verifier(tmp_path, worker=worker)
    cfg = verifier._get_data_cache_config()
    cache_key = verifier._get_baseline_cache_key(5, 50)
    write_baseline_result_to_cache(
        cfg,
        op_name=verifier.op_name,
        cache_key=cache_key,
        result_data={"avg_time_us": 17.5, "method": "cached"},
    )

    verify_dir = (
        Path(verifier.log_dir).expanduser()
        / verifier.op_name
        / f"Iteration{verifier.task_id}_Step00_verify"
    )
    verify_dir.mkdir(parents=True, exist_ok=True)
    (verify_dir / f"{verifier.op_name}_{verifier.framework}.py").write_text(
        FRAMEWORK_CODE,
        encoding="utf-8",
    )
    (verify_dir / f"{verifier.op_name}_{verifier.dsl}_impl.py").write_text(
        IMPL_CODE,
        encoding="utf-8",
    )

    captured = {}

    def fake_gen_profile_project(verify_dir, device_id=0, warmup_times=5, run_times=50, skip_base=False):
        captured["skip_base"] = skip_base

    monkeypatch.setattr(verifier, "gen_profile_project", fake_gen_profile_project)
    monkeypatch.setattr(verifier, "_pack_directory", lambda verify_dir: b"pkg")

    result = await verifier.run_profile(
        {"coder_code": IMPL_CODE},
        current_step=0,
        device_id=0,
        profile_settings={"warmup_times": 5, "run_times": 50},
    )

    assert captured["skip_base"] is True
    assert worker.profile_settings is not None
    assert worker.profile_settings["override_base_time_us"] == 17.5
    assert worker.profile_settings["skip_base_profile"] is True
    assert result["base_time"] == 17.5
