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

import asyncio
import io
from pathlib import Path

import pytest
import torch

from akg_agents.op.verifier.data_cache import (
    VerifierDataCacheConfig,
    build_baseline_cache_key,
    build_reference_cache_key,
    extract_baseline_time_us,
    load_verifier_data_cache_config,
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


def _make_reference_payload_bytes() -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "op_name": "relu",
            "seed": 0,
            "save_inputs": True,
            "inputs": [torch.tensor([1.0])],
            "init_inputs": [],
            "outputs": [torch.tensor([1.0])],
        },
        buffer,
    )
    return buffer.getvalue()


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


class DummyReferenceWorker:
    async def generate_reference(self, package_data, task_id, op_name, timeout):
        raise AssertionError("generate_reference should be stubbed by the test")

    async def verify(self, package_data, task_id, op_name, timeout):
        return True, "ok", {}


def _make_config(tmp_path: Path) -> dict:
    return {
        "log_dir": str(tmp_path / "logs"),
        "data_cache": {
            "enabled": True,
            "cache_dir": str(tmp_path / "verifier_cache"),
        },
    }


def _make_verifier(tmp_path: Path, worker=None, task_id: str = "ut") -> KernelVerifier:
    return KernelVerifier(
        op_name="relu",
        framework_code=FRAMEWORK_CODE,
        task_id=task_id,
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

    reference_bytes = _make_reference_payload_bytes()
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
        dsl="triton_ascend",
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


def test_baseline_cache_key_includes_dsl():
    common_kwargs = {
        "op_name": "relu",
        "framework_code": FRAMEWORK_CODE,
        "framework": "torch",
        "backend": "ascend",
        "arch": "ascend910b4",
        "bench_type": "kernelbench",
        "warmup_times": 5,
        "run_times": 50,
    }

    triton_key = build_baseline_cache_key(**common_kwargs, dsl="triton_ascend")
    torch_key = build_baseline_cache_key(**common_kwargs, dsl="torch")
    assert triton_key != torch_key


def test_reference_cache_key_includes_task_id():
    common_kwargs = {
        "op_name": "relu",
        "framework_code": FRAMEWORK_CODE,
        "framework": "torch",
        "backend": "ascend",
        "arch": "ascend910b4",
        "bench_type": "kernelbench",
    }

    task_a_key = build_reference_cache_key(**common_kwargs, task_id="task-a")
    task_b_key = build_reference_cache_key(**common_kwargs, task_id="task-b")
    assert task_a_key != task_b_key


def test_verifier_cache_keys_include_task_id(tmp_path):
    verifier_a = _make_verifier(tmp_path, task_id="task-a")
    verifier_b = _make_verifier(tmp_path, task_id="task-b")

    assert verifier_a._get_reference_cache_key() != verifier_b._get_reference_cache_key()
    assert verifier_a._get_baseline_cache_key(5, 50) != verifier_b._get_baseline_cache_key(5, 50)


def test_detect_dynamic_shape_requires_function_definition(tmp_path):
    verifier = _make_verifier(tmp_path)
    verifier.framework_code += "\n# get_inputs_dyn_list mentioned in a comment should not enable dynamic shape\n"
    assert verifier._detect_dynamic_shape() is False

    verifier.framework_code += "\n\ndef get_inputs_dyn_list():\n    return []\n"
    assert verifier._detect_dynamic_shape() is True


def test_cached_reference_data_loads_with_weights_only(monkeypatch, tmp_path):
    verifier = _make_verifier(tmp_path)
    captured = {}

    def fake_torch_load(file_obj, **kwargs):
        captured.update(kwargs)
        return {"save_inputs": True, "inputs": [], "outputs": []}

    monkeypatch.setattr(torch, "load", fake_torch_load)

    assert verifier._is_valid_cached_reference_data(b"payload") is True
    assert captured["map_location"] == "cpu"
    assert captured["weights_only"] is True


def test_cache_dir_is_expanded_from_config(monkeypatch, tmp_path):
    home_dir = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home_dir))

    cfg = load_verifier_data_cache_config(
        {
            "data_cache": {
                "enabled": True,
                "cache_dir": "~/akg_cache",
            }
        }
    )

    assert cfg.cache_dir == str(home_dir / "akg_cache")


@pytest.mark.asyncio
async def test_run_uses_cached_reference_data(monkeypatch, tmp_path):
    verifier = _make_verifier(tmp_path, worker=DummyReferenceWorker())
    cfg = verifier._get_data_cache_config()
    cache_key = verifier._get_reference_cache_key()
    cached_reference = _make_reference_payload_bytes()
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


@pytest.mark.asyncio
async def test_run_regenerates_corrupted_reference_cache(monkeypatch, tmp_path):
    verifier = _make_verifier(tmp_path, worker=DummyReferenceWorker())
    cfg = verifier._get_data_cache_config()
    cache_key = verifier._get_reference_cache_key()
    write_reference_data_to_cache(
        cfg,
        op_name=verifier.op_name,
        cache_key=cache_key,
        reference_data=b"not-a-valid-torch-save",
        metadata={"save_inputs": True},
    )

    regenerated_reference = _make_reference_payload_bytes()

    async def fake_generate_reference_data(task_desc, timeout=120, save_inputs=False, device_id=None):
        return True, "regenerated", regenerated_reference

    monkeypatch.setattr(verifier, "generate_reference_data", fake_generate_reference_data)

    loaded = await verifier._prepare_cached_reference_data(device_id=0)
    assert loaded == regenerated_reference
    assert read_reference_data_from_cache(cfg, op_name=verifier.op_name, cache_key=cache_key) == regenerated_reference


@pytest.mark.asyncio
async def test_run_serializes_concurrent_reference_cache_generation(monkeypatch, tmp_path):
    verifier_a = _make_verifier(tmp_path, worker=DummyReferenceWorker(), task_id="shared")
    verifier_b = _make_verifier(tmp_path, worker=DummyReferenceWorker(), task_id="shared")
    regenerated_reference = _make_reference_payload_bytes()
    generate_calls = 0

    async def fake_generate_reference_data(task_desc, timeout=120, save_inputs=False, device_id=None):
        nonlocal generate_calls
        generate_calls += 1
        await asyncio.sleep(0.05)
        return True, "regenerated", regenerated_reference

    monkeypatch.setattr(verifier_a, "generate_reference_data", fake_generate_reference_data)
    monkeypatch.setattr(verifier_b, "generate_reference_data", fake_generate_reference_data)

    loaded_a, loaded_b = await asyncio.gather(
        verifier_a._prepare_cached_reference_data(device_id=0),
        verifier_b._prepare_cached_reference_data(device_id=0),
    )

    assert loaded_a == regenerated_reference
    assert loaded_b == regenerated_reference
    assert generate_calls == 1


@pytest.mark.asyncio
async def test_run_skips_reference_cache_for_dynamic_shape(tmp_path):
    verifier = _make_verifier(tmp_path, worker=DummyReferenceWorker())
    verifier.framework_code += "\n\ndef get_inputs_dyn_list():\n    return []\n"

    loaded = await verifier._prepare_cached_reference_data(device_id=0)
    assert loaded is None
