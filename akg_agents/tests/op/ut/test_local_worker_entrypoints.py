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

"""End-to-end behavior tests for the refactored LocalWorker entry points.

These exercise the shared `_extract_package` + `_run_script` plumbing and the
msprof/nsys merge with real mock subprocesses (no NPU device needed). They
pin the externally-observable contract — return shapes, success/failure,
timeout, and extract-error mapping — so the slimming refactor can't change it.
"""

import io
import json
import os
import tarfile
from types import SimpleNamespace

import pytest

from akg_agents.core.worker.local_worker import LocalWorker


def _pkg(files: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _worker():
    return LocalWorker(device_pool=object(), backend="cpu")


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_verify_success():
    pkg = _pkg({"verify_toy.py": "print('VERIFY_OK')\n"})
    ok, log, artifacts = await _worker().verify(pkg, "t", "toy")
    assert ok is True
    assert "VERIFY_OK" in log
    assert artifacts == {}


@pytest.mark.asyncio
async def test_verify_collects_json_artifacts():
    script = "open('out.json', 'w').write('{\"k\": 1}')\nprint('done')\n"
    pkg = _pkg({"verify_toy.py": script})
    ok, _log, artifacts = await _worker().verify(pkg, "t", "toy")
    assert ok is True
    assert artifacts.get("out.json") == '{"k": 1}'


@pytest.mark.asyncio
async def test_verify_script_failure_returns_log():
    pkg = _pkg({"verify_toy.py": "import sys; sys.stderr.write('boom'); sys.exit(3)\n"})
    ok, log, _ = await _worker().verify(pkg, "t", "toy")
    assert ok is False
    assert "boom" in log


@pytest.mark.asyncio
async def test_verify_missing_script():
    ok, log, _ = await _worker().verify(_pkg({"other.py": "x=1\n"}), "t", "toy")
    assert ok is False
    assert "not found" in log


@pytest.mark.asyncio
async def test_verify_extract_error_maps_to_failure():
    ok, log, artifacts = await _worker().verify(b"this is not a tar", "t", "toy")
    assert ok is False
    assert "Failed to extract package" in log
    assert artifacts == {}


@pytest.mark.asyncio
async def test_verify_rejects_archive_path_traversal(tmp_path):
    outside = tmp_path / "escaped.py"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        data = b"owned\n"
        info = tarfile.TarInfo(str(outside))
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    ok, log, artifacts = await _worker().verify(
        buf.getvalue(), "t", "toy")
    assert ok is False
    assert "Unsafe archive member rejected" in log
    assert artifacts == {}
    assert not outside.exists()


@pytest.mark.asyncio
async def test_verify_accepts_str_dir_path(tmp_path):
    (tmp_path / "verify_toy.py").write_text("print('FROM_DIR')\n", encoding="utf-8")
    ok, log, _ = await _worker().verify(str(tmp_path), "t", "toy")
    assert ok is True
    assert "FROM_DIR" in log


@pytest.mark.asyncio
async def test_verify_timeout_kills_and_flags(monkeypatch):
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "0.5")
    pkg = _pkg({"verify_toy.py": "import time; time.sleep(30)\n"})
    ok, log, _ = await _worker().verify(pkg, "t", "toy", timeout=1)
    assert ok is False
    assert "timed out" in log.lower()


# --------------------------------------------------------------------------
# generate_reference
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_reference_success():
    script = (
        "print('REFERENCE_GENERATION_SUCCESS')\n"
        "open('toy_reference.pt', 'wb').write(b'PTDATA')\n"
    )
    ok, _log, ref = await _worker().generate_reference(
        _pkg({"verify_toy.py": script}), "t", "toy")
    assert ok is True
    assert ref == b"PTDATA"


@pytest.mark.asyncio
async def test_generate_reference_missing_marker():
    # Exits 0 but never prints the success marker → treated as failure.
    ok, _log, ref = await _worker().generate_reference(
        _pkg({"verify_toy.py": "print('no marker here')\n"}), "t", "toy")
    assert ok is False
    assert ref == b""


@pytest.mark.asyncio
async def test_generate_reference_extract_error():
    ok, log, ref = await _worker().generate_reference(b"garbage", "t", "toy")
    assert ok is False
    assert "Failed to extract package" in log
    assert ref == b""


# --------------------------------------------------------------------------
# profile_single_task
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_profile_single_success():
    script = ("import json\n"
              "json.dump({'avg_time_us': 42.0}, open('profile_single_result.json', 'w'))\n")
    res = await _worker().profile_single_task(
        _pkg({"profile_single_toy.py": script}), "t", "toy", {})
    assert res["success"] is True
    assert res["time_us"] == 42.0


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_time", [0, -1, float("inf"), float("nan")])
async def test_profile_single_rejects_non_positive_or_non_finite_time(bad_time):
    payload = json.dumps({"avg_time_us": bad_time})
    script = f"open('profile_single_result.json', 'w').write({payload!r})\n"
    res = await _worker().profile_single_task(
        _pkg({"profile_single_toy.py": script}), "t", "toy", {})
    assert res["success"] is False
    assert res["time_us"] is None


@pytest.mark.asyncio
async def test_profile_single_extract_error():
    res = await _worker().profile_single_task(b"garbage", "t", "toy", {})
    assert res["success"] is False
    assert "Failed to extract package" in res["log"]


# --------------------------------------------------------------------------
# profile (full base + generation via the python-script path)
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_profile_full_base_and_generation(monkeypatch):
    monkeypatch.setattr(
        "akg_agents.op.verifier.adapters.factory.get_dsl_adapter",
        lambda _dsl: SimpleNamespace(profile_via_python_script=True),
    )
    monkeypatch.setattr(
        "akg_agents.core.worker.local_worker.compute_roofline_profile",
        lambda **_k: {"success": False, "skipped": True},
    )
    monkeypatch.setattr(
        "akg_agents.core.worker.local_worker.write_roofline_profile_result",
        lambda *_a, **_k: None,
    )
    base = ("import json\n"
            "json.dump({'avg_time_us': 10.0, 'per_case_us': [10.0]}, "
            "open('base_profile_result.json', 'w'))\n")
    gen = ("import json\n"
           "json.dump({'avg_time_us': 5.0, 'per_case_us': [5.0]}, "
           "open('generation_profile_result.json', 'w'))\n")
    pkg = _pkg({"profile_toy_base.py": base, "profile_toy_generation.py": gen})

    res = await _worker().profile(pkg, "t", "toy", {"dsl": "fake"})

    assert res["base_time"] == 10.0
    assert res["gen_time"] == 5.0
    assert res["per_shape_base_us"] == [10.0]
    assert res["per_shape_gen_us"] == [5.0]
    assert res["speedup"] == pytest.approx(2.0)  # geomean(base/gen)
    assert res.get("error") is None


@pytest.mark.asyncio
async def test_profile_extract_error_returns_error_shape():
    res = await _worker().profile(b"garbage", "t", "toy", {"dsl": "fake"})
    assert res["gen_time"] is None
    assert res["base_time"] is None
    assert res["speedup"] == 0.0
    assert res["per_shape_gen_us"] == []
    assert res["roofline"] is None
    assert "Failed to extract package" in res["error"]


# --------------------------------------------------------------------------
# _run_trace_profiling (the msprof / nsys merge)
# --------------------------------------------------------------------------

def test_run_trace_profiling_runs_both_and_tags_method(tmp_path):
    (tmp_path / "profile_toy_base.py").write_text("", encoding="utf-8")
    (tmp_path / "profile_toy_generation.py").write_text("", encoding="utf-8")
    seen = []

    def run(script):
        seen.append(os.path.basename(script))
        return True, "", script + ".trace"

    def analyze(_path, kind):
        return True, "", {"base": 8.0, "generation": 4.0}[kind]

    sections = _worker()._run_trace_profiling(
        str(tmp_path), "toy", "t", run, analyze, "msprof")

    assert seen == ["profile_toy_base.py", "profile_toy_generation.py"]
    assert sections["base"]["avg_us"] == 8.0
    assert sections["gen"]["avg_us"] == 4.0
    assert sections["base"]["method"] == "msprof"


def test_run_trace_profiling_skips_missing_and_survives_run_failure(tmp_path):
    # Only the generation script exists; base is skipped. run() fails for it.
    (tmp_path / "profile_toy_generation.py").write_text("", encoding="utf-8")

    def run(_script):
        return False, "profiler exploded", None

    def analyze(_path, _kind):
        raise AssertionError("analyze must not be called when run fails")

    sections = _worker()._run_trace_profiling(
        str(tmp_path), "toy", "t", run, analyze, "nsys")

    assert sections["base"] is None
    assert sections["gen"] is None
