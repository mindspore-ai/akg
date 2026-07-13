# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

import asyncio
import io
import tarfile
import threading
import time
from types import SimpleNamespace

import pytest

from akg_agents.core.worker.local_worker import LocalWorker
from akg_agents.op.verifier import profiler_utils
from akg_agents.op.verifier.profiler_utils import make_profile_section


def _profile_package(op_name: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in (
            f"profile_{op_name}_base.py",
            f"profile_{op_name}_generation.py",
        ):
            data = b"# placeholder\n"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.mark.asyncio
async def test_local_worker_profile_keeps_gen_on_base_fail(
    monkeypatch,
):
    op_name = "toy"

    def fake_get_dsl_adapter(_dsl):
        return SimpleNamespace(profile_via_python_script=True)

    async def fake_collect(verify_dir, op_name_arg, run_script, *,
                           task_id="0", override_base_section=None):
        return {
            "base": None,
            "gen": make_profile_section(
                7.5, per_case_us=[7.0, 8.0], method="unit"),
        }

    monkeypatch.setattr(
        "akg_agents.op.verifier.adapters.factory.get_dsl_adapter",
        fake_get_dsl_adapter,
    )
    monkeypatch.setattr(
        "akg_agents.core.worker.local_worker.run_profile_scripts_and_collect_results",
        fake_collect,
    )
    monkeypatch.setattr(
        "akg_agents.core.worker.local_worker.compute_roofline_profile",
        lambda **_kwargs: {"success": False, "skipped": True},
    )
    monkeypatch.setattr(
        "akg_agents.core.worker.local_worker.write_roofline_profile_result",
        lambda *_args, **_kwargs: None,
    )

    worker = LocalWorker(device_pool=object(), backend="cpu")
    result = await worker.profile(
        _profile_package(op_name),
        task_id="unit",
        op_name=op_name,
        profile_settings={
            "timeout": 1234,
            "warmup_times": 3,
            "run_times": 7,
            "dsl": "fake",
        },
    )

    assert result["error"] == "base profile failed"
    assert result["gen_time"] == 7.5
    assert result["base_time"] is None
    assert result["per_shape_gen_us"] == [7.0, 8.0]
    assert result["per_shape_base_us"] == []


@pytest.mark.asyncio
async def test_run_profile_script_timeout_kills_subprocess(tmp_path):
    """The structural guarantee: a profile script that overruns its timeout is
    killed (process group torn down), not left orphaned. The script records its
    own PID; after the timed-out call we assert that PID is gone."""
    import os
    import sys

    op_name = "toy"
    pid_file = tmp_path / "child.pid"
    script = tmp_path / f"profile_{op_name}_generation.py"
    script.write_text(
        "import os, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )

    worker = LocalWorker(device_pool=object(), backend="cpu")
    ok = await worker._run_profile_script_async(
        str(tmp_path), f"profile_{op_name}_generation.py",
        timeout=1, task_id="unit", label="generation_profile",
    )

    assert ok is False  # timed out → failure
    assert pid_file.exists(), "child never started"
    child_pid = int(pid_file.read_text())
    if os.name == "posix":
        # The child (and its process group) must be dead — no orphan.
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)


@pytest.mark.asyncio
async def test_external_profiler_cancel_waits_for_thread_cleanup():
    """A cancelled profile must finish executor cleanup before lease release."""
    started = threading.Event()
    cleaned = threading.Event()

    def fake_profiler(cancel_event):
        started.set()
        assert cancel_event.wait(timeout=5), "cancellation was not propagated"
        time.sleep(0.05)  # model process-tree drain / trace cleanup
        cleaned.set()
        return {"base": None, "gen": None}

    worker = LocalWorker(device_pool=object(), backend="ascend")
    task = asyncio.create_task(worker._run_external_profiler(fake_profiler))
    while not started.is_set():
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set(), "cleanup thread was still running after cancellation"


def test_run_nsys_uses_argv_and_profile_directory(monkeypatch, tmp_path):
    script = tmp_path / "profile_toy_generation.py"
    script.write_text("# fixture\n", encoding="utf-8")
    seen = {}

    def fake_capture(cmd, **kwargs):
        seen.update(cmd=cmd, kwargs=kwargs)
        (tmp_path / "nsys_report_profile_toy_generation.nsys-rep").touch()
        return 0, "", "", False

    monkeypatch.setattr(profiler_utils, "run_command_capture", fake_capture)
    ok, error, report = profiler_utils.run_nsys(str(script), timeout=7)

    assert ok is True and error == ""
    assert report == str(
        tmp_path / "nsys_report_profile_toy_generation.nsys-rep")
    assert seen["cmd"] == [
        "nsys", "profile", "--output=nsys_report_profile_toy_generation",
        "python", str(script),
    ]
    assert seen["kwargs"]["cwd"] == str(tmp_path)


def test_run_msprof_uses_shell_free_argv(monkeypatch, tmp_path):
    script = tmp_path / "profile toy.py"
    script.write_text("# fixture\n", encoding="utf-8")
    seen = {}

    def fake_capture(cmd, **kwargs):
        seen.update(cmd=cmd, kwargs=kwargs)
        return (
            0,
            "[INFO] Process profiling data complete. Data is saved in /tmp/prof\n",
            "",
            False,
        )

    monkeypatch.setattr(profiler_utils, "run_command_capture", fake_capture)
    ok, error, report = profiler_utils.run_msprof(str(script), timeout=7)

    assert ok is True and error == "" and report == "/tmp/prof"
    assert seen["cmd"] == ["msprof", f"--application=python {script}"]
    assert "shell" not in seen["kwargs"]
