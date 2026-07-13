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
import json
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import torch

from akg_agents.core.worker.remote_worker import RemoteWorker
from akg_agents.core.worker.interface import empty_profile_result
from akg_agents.op.verifier.data_cache import (
    VerifierDataCacheConfig,
    build_baseline_cache_key,
    build_reference_cache_key,
    build_sol_problem_cache_identity,
    build_workflow_data_cache_key_id,
    extract_baseline_time_us,
    get_verifier_data_cache_key_id,
    load_verifier_data_cache_config,
    read_baseline_result_from_cache,
    read_reference_data_from_cache,
    set_verifier_data_cache_key_id,
    set_workflow_data_cache_key_id,
    verifier_data_cache_lock,
    write_baseline_result_to_cache,
    write_reference_data_to_cache,
)
from akg_agents.op.verifier.baseline_profiler import (
    _run_cached_baseline_profile,
    profile_baseline_once,
)
from akg_agents.core.worker.manager import WorkerManager
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

SOL_DEFINITION = {
    "name": "000_relu",
    "description": "ReLU SOL case",
    "axes": {"n": {"type": "var"}},
    "inputs": {"x": {"shape": ["n"], "dtype": "float32"}},
    "outputs": {"out": {"shape": ["n"], "dtype": "float32"}},
}

SOL_REFERENCE = "import torch\n\n\ndef run(x):\n    return torch.relu(x)\n"


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

    async def acquire_device(self, task_id="unknown", timeout=None):
        return 0, 1  # (device_id, lease_id)

    async def release_device(self, device_id, lease_id, task_id="unknown"):
        pass

    async def profile(self, package_data, task_id, op_name, profile_settings):
        self.profile_settings = dict(profile_settings)
        section = profile_settings.get("override_base_section") or {}
        return {
            "gen_time": 5.0,
            "base_time": section.get("avg_us"),
            "speedup": 2.0,
            "roofline_time": None,
            "roofline_speedup": 0.0,
            "roofline": None,
            "artifacts": {},
        }


class FailedProfileWorker(DummyProfileWorker):
    async def profile(self, package_data, task_id, op_name, profile_settings):
        return empty_profile_result(error="network down")


class LeaseTrackingWorker:
    def __init__(self):
        self.events = []

    @asynccontextmanager
    async def device_lease(self, task_id="unknown", timeout=None):
        self.events.append(("acquire", task_id))
        try:
            yield 7
        finally:
            self.events.append(("release", task_id))


class DummyReferenceWorker:
    async def acquire_device(self, task_id="unknown", timeout=None):
        return 0, 1  # (device_id, lease_id)

    async def release_device(self, device_id, lease_id, task_id="unknown"):
        pass

    async def generate_reference(self, package_data, task_id, op_name, timeout):
        raise AssertionError("generate_reference should be stubbed by the test")

    async def verify(self, package_data, task_id, op_name, timeout):
        return True, "ok", {}


class SimulatedRemoteWorker(RemoteWorker):
    def __init__(self, reference_bytes: bytes):
        super().__init__("http://unit-test-worker")
        self.reference_bytes = reference_bytes
        self.generate_reference_calls = 0
        self.verify_calls = 0
        self.acquired_devices = []
        self.released_devices = []

    async def acquire_device(self, task_id: str = "unknown", timeout: float = None):
        self.acquired_devices.append(task_id)
        return 3, 1  # (device_id, lease_id)

    async def release_device(self, device_id: int, lease_id: int, task_id: str = "unknown"):
        self.released_devices.append((device_id, task_id))

    async def generate_reference(self, package_data, task_id, op_name, timeout):
        self.generate_reference_calls += 1
        assert isinstance(package_data, (bytes, bytearray))
        return True, "remote generated", self.reference_bytes

    async def verify(self, package_data, task_id, op_name, timeout):
        self.verify_calls += 1
        assert isinstance(package_data, (bytes, bytearray))
        with tarfile.open(fileobj=io.BytesIO(package_data), mode="r") as tar_file:
            names = set(tar_file.getnames())
        assert "relu_reference.pt" in names
        return True, "remote ok", {}


def _make_config(tmp_path: Path) -> dict:
    return {
        "log_dir": str(tmp_path / "logs"),
        "data_cache": {
            "enabled": True,
            "cache_dir": str(tmp_path / "verifier_cache"),
        },
    }


def _write_sol_problem_dir(
    tmp_path: Path,
    *,
    case_name: str = "sol_case",
    workload_uuid: str = "case-1",
    n: int = 128,
) -> Path:
    case_dir = tmp_path / case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    workload = {
        "uuid": workload_uuid,
        "axes": {"n": n},
        "inputs": {"x": {"type": "random"}},
        "tolerance": {"max_atol": 1e-5, "max_rtol": 1e-5},
    }
    (case_dir / "definition.json").write_text(
        json.dumps(SOL_DEFINITION, sort_keys=True),
        encoding="utf-8",
    )
    (case_dir / "workload.jsonl").write_text(
        json.dumps(workload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (case_dir / "reference.py").write_text(SOL_REFERENCE, encoding="utf-8")
    return case_dir


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


def _make_sol_config(tmp_path: Path, sol_problem_dir: Path = None) -> dict:
    config = _make_config(tmp_path)
    if sol_problem_dir is None:
        sol_problem_dir = _write_sol_problem_dir(tmp_path)
    config.update(
        {
            "bench_type": "sol",
            "sol_problem_dir": str(sol_problem_dir),
        }
    )
    return config


def _make_sol_verifier(
    tmp_path: Path,
    worker=None,
    task_id: str = "ut",
    sol_problem_dir: Path = None,
) -> KernelVerifier:
    return KernelVerifier(
        op_name="relu",
        framework_code="",
        task_id=task_id,
        framework="torch",
        dsl="triton_cuda",
        backend="cuda",
        arch="a100",
        impl_func_name="ModelNew",
        config=_make_sol_config(tmp_path, sol_problem_dir),
        worker=worker,
        bench_type="sol",
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


def test_sol_baseline_cache_key_includes_problem_content(tmp_path):
    case_a = _write_sol_problem_dir(tmp_path, case_name="sol_case_a", workload_uuid="case-1", n=128)
    case_b = _write_sol_problem_dir(tmp_path, case_name="sol_case_b", workload_uuid="case-2", n=256)

    verifier_a = _make_sol_verifier(tmp_path, task_id="same-task", sol_problem_dir=case_a)
    verifier_b = _make_sol_verifier(tmp_path, task_id="same-task", sol_problem_dir=case_b)

    assert verifier_a._get_baseline_cache_key(5, 50) != verifier_b._get_baseline_cache_key(5, 50)


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


def test_verifier_cache_keys_reuse_same_task_id(tmp_path):
    verifier_a = _make_verifier(tmp_path, task_id="cache-demo")
    verifier_b = _make_verifier(tmp_path, task_id="cache-demo")

    assert verifier_a._get_reference_cache_key() == verifier_b._get_reference_cache_key()
    assert verifier_a._get_baseline_cache_key(5, 50) == verifier_b._get_baseline_cache_key(5, 50)


def test_cache_key_id_allows_cross_task_reuse(tmp_path):
    config_a = _make_config(tmp_path)
    config_b = _make_config(tmp_path)
    set_verifier_data_cache_key_id(config_a, "stable-op-task")
    set_verifier_data_cache_key_id(config_b, "stable-op-task")

    verifier_a = KernelVerifier(
        op_name="relu",
        framework_code=FRAMEWORK_CODE,
        task_id="task-a",
        framework="torch",
        dsl="triton_ascend",
        backend="ascend",
        arch="ascend910b4",
        impl_func_name="ModelNew",
        config=config_a,
    )
    verifier_b = KernelVerifier(
        op_name="relu",
        framework_code=FRAMEWORK_CODE,
        task_id="task-b",
        framework="torch",
        dsl="triton_ascend",
        backend="ascend",
        arch="ascend910b4",
        impl_func_name="ModelNew",
        config=config_b,
    )

    assert verifier_a._get_reference_cache_key() == verifier_b._get_reference_cache_key()
    assert verifier_a._get_baseline_cache_key(5, 50) == verifier_b._get_baseline_cache_key(5, 50)


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


def test_data_cache_config_handles_non_dict_section():
    cfg = load_verifier_data_cache_config({"data_cache": "invalid"})

    assert cfg.enabled is False
    assert get_verifier_data_cache_key_id({"data_cache": "invalid"}, "fallback") == "fallback"

    config = {"data_cache": "invalid"}
    set_verifier_data_cache_key_id(config, "stable-key")
    assert config["data_cache"] == {"cache_key_id": "stable-key"}


def test_workflow_data_cache_key_id_helper_preserves_user_value():
    expected_key_id = build_workflow_data_cache_key_id(
        op_name="relu",
        framework="torch",
        dsl="triton_ascend",
        backend="ascend",
        arch="ascend910b4",
        bench_type="kernelbench",
    )
    config = {"data_cache": {"cache_key_id": "user-key"}}

    set_workflow_data_cache_key_id(
        config,
        op_name="relu",
        framework="torch",
        dsl="triton_ascend",
        backend="ascend",
        arch="ascend910b4",
        bench_type="kernelbench",
    )
    assert config["data_cache"]["cache_key_id"] == "user-key"

    set_workflow_data_cache_key_id(
        config,
        op_name="relu",
        framework="torch",
        dsl="triton_ascend",
        backend="ascend",
        arch="ascend910b4",
        bench_type="kernelbench",
        overwrite=True,
    )
    assert config["data_cache"]["cache_key_id"] == expected_key_id


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
    assert worker.profile_settings["override_base_section"]["avg_us"] == 17.5
    assert worker.profile_settings["skip_base_profile"] is True
    assert result["base_time"] == 17.5


@pytest.mark.asyncio
async def test_run_profile_preserves_worker_failure(monkeypatch, tmp_path):
    verifier = _make_verifier(tmp_path, worker=FailedProfileWorker())
    verify_dir = (
        Path(verifier.log_dir).expanduser()
        / verifier.op_name
        / f"Iteration{verifier.task_id}_Step0_verify"
    )
    verify_dir.mkdir(parents=True, exist_ok=True)
    (verify_dir / f"{verifier.op_name}_{verifier.framework}.py").write_text(
        FRAMEWORK_CODE, encoding="utf-8")
    (verify_dir / f"{verifier.op_name}_{verifier.dsl}_impl.py").write_text(
        IMPL_CODE, encoding="utf-8")
    monkeypatch.setattr(verifier, "gen_profile_project", lambda *args, **kwargs: None)
    monkeypatch.setattr(verifier, "_pack_directory", lambda verify_dir: b"pkg")

    result = await verifier.run_profile(
        {"coder_code": IMPL_CODE}, current_step=0, device_id=0)

    assert result["error"] == "network down"
    assert result["gen_time"] is None
    assert result["base_time"] is None


@pytest.mark.asyncio
async def test_run_profile_uses_cached_sol_baseline(monkeypatch, tmp_path):
    worker = DummyProfileWorker()
    verifier = _make_sol_verifier(tmp_path, worker=worker)
    cfg = verifier._get_data_cache_config()
    cache_key = verifier._get_baseline_cache_key(5, 50)
    assert cache_key
    write_baseline_result_to_cache(
        cfg,
        op_name=verifier.op_name,
        cache_key=cache_key,
        result_data={"avg_time_us": 19.5, "method": "cached_sol"},
    )

    verify_dir = (
        Path(verifier.log_dir).expanduser()
        / verifier.op_name
        / f"Iteration{verifier.task_id}_Step00_verify"
    )
    verify_dir.mkdir(parents=True, exist_ok=True)
    (verify_dir / "definition.json").write_text("{}", encoding="utf-8")
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
    assert worker.profile_settings["override_base_section"]["avg_us"] == 19.5
    assert worker.profile_settings["skip_base_profile"] is True
    assert result["base_time"] == 19.5


@pytest.mark.asyncio
async def test_baseline_preprofile_rereads_cache_after_lock(tmp_path):
    config = _make_config(tmp_path)
    cfg = load_verifier_data_cache_config(config)
    cache_key = build_baseline_cache_key(
        op_name="relu",
        framework_code=FRAMEWORK_CODE,
        framework="torch",
        backend="ascend",
        arch="ascend910b4",
        bench_type="kernelbench",
        warmup_times=3,
        run_times=7,
        dsl="triton_ascend",
        task_id="baseline_profile",
    )

    async with verifier_data_cache_lock(
        cfg,
        namespace="baseline",
        op_name="relu",
        cache_key=cache_key,
    ):
        task = asyncio.create_task(
            profile_baseline_once(
                "relu",
                FRAMEWORK_CODE,
                "triton_ascend",
                "torch",
                "ascend",
                "ascend910b4",
                config,
                warmup_times=3,
                run_times=7,
            )
        )
        await asyncio.sleep(0.05)
        write_baseline_result_to_cache(
            cfg,
            op_name="relu",
            cache_key=cache_key,
            result_data={"avg_time_us": 23.0, "method": "locked_test"},
        )

    assert await task == 23.0


@pytest.mark.asyncio
async def test_cached_baseline_releases_device_and_manager(monkeypatch, tmp_path):
    import akg_agents.core.worker.manager as manager_module

    manager = WorkerManager()
    worker = LeaseTrackingWorker()
    await manager.register(worker, "ascend", "ascend910b4")
    monkeypatch.setattr(manager_module, "_GLOBAL_MANAGER", manager)
    captured = {}

    async def prepare(worker_arg, verifier, profile_dir, device_id,
                      warmup_times, run_times, timeout):
        captured["worker"] = worker_arg
        captured["device_id"] = device_id
        return {"success": True, "time_us": 12.5, "log": ""}

    result = await _run_cached_baseline_profile(
        "relu", "triton_ascend", "torch", "ascend", "ascend910b4",
        {"log_dir": str(tmp_path / "logs"),
         "data_cache": {"enabled": False}},
        1, 2, 30,
        bench_type="sol", bench_label="SOL", cache_framework_code="",
        prepare_fn=prepare, cache_method="test", times_label="case",
    )

    assert result == 12.5
    assert captured == {"worker": worker, "device_id": 7}
    assert worker.events == [
        ("acquire", "baseline_profile"),
        ("release", "baseline_profile"),
    ]
    assert (await manager.get_status())[0]["load"] == 0


@pytest.mark.asyncio
async def test_sol_baseline_preprofile_rereads_cache_after_lock(tmp_path):
    sol_problem_dir = _write_sol_problem_dir(tmp_path)
    config = _make_sol_config(tmp_path, sol_problem_dir)
    cfg = load_verifier_data_cache_config(config)
    sol_identity = build_sol_problem_cache_identity(str(sol_problem_dir))
    cache_key = build_baseline_cache_key(
        op_name="relu",
        framework_code=sol_identity,
        framework="torch",
        backend="cuda",
        arch="a100",
        bench_type="sol",
        warmup_times=3,
        run_times=7,
        dsl="triton_cuda",
        task_id="baseline_profile",
    )

    async with verifier_data_cache_lock(
        cfg,
        namespace="baseline",
        op_name="relu",
        cache_key=cache_key,
    ):
        task = asyncio.create_task(
            profile_baseline_once(
                "relu",
                "",
                "triton_cuda",
                "torch",
                "cuda",
                "a100",
                config,
                warmup_times=3,
                run_times=7,
            )
        )
        await asyncio.sleep(0.05)
        write_baseline_result_to_cache(
            cfg,
            op_name="relu",
            cache_key=cache_key,
            result_data={"avg_time_us": 31.0, "method": "locked_sol_test"},
        )

    assert await task == 31.0


@pytest.mark.asyncio
async def test_remote_worker_reference_cache_roundtrip(tmp_path):
    reference_bytes = _make_reference_payload_bytes()
    worker_a = SimulatedRemoteWorker(reference_bytes)
    verifier_a = _make_verifier(tmp_path, worker=worker_a, task_id="remote-cache")

    success, log = await verifier_a.run({"coder_code": IMPL_CODE}, device_id=-1)
    assert success is True
    assert log == "remote ok"
    assert worker_a.generate_reference_calls == 1
    assert worker_a.verify_calls == 1
    assert worker_a.acquired_devices == ["remote-cache"]
    assert worker_a.released_devices == [(3, "remote-cache")]

    worker_b = SimulatedRemoteWorker(b"should-not-be-used")
    verifier_b = _make_verifier(tmp_path, worker=worker_b, task_id="remote-cache")

    success, log = await verifier_b.run({"coder_code": IMPL_CODE}, device_id=-1)
    assert success is True
    assert log == "remote ok"
    assert worker_b.generate_reference_calls == 0
    assert worker_b.verify_calls == 1


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


@pytest.mark.asyncio
async def test_managed_reference_data_is_cleared_when_cache_key_changes(tmp_path):
    verifier = _make_verifier(tmp_path, worker=None, task_id="task-a")
    cached_reference = _make_reference_payload_bytes()

    verifier._apply_cached_reference_data(cached_reference)
    assert verifier.config["use_reference_data"] is True
    assert verifier.config["reference_data"] == cached_reference
    assert verifier.config["_data_cache_reference_key"] == verifier._get_reference_cache_key()

    verifier.task_id = "task-b"
    loaded = await verifier._prepare_cached_reference_data(device_id=0)

    assert loaded is None
    assert "reference_data" not in verifier.config
    assert "use_reference_data" not in verifier.config
    assert "use_reference_inputs" not in verifier.config
    assert "_data_cache_reference_key" not in verifier.config
