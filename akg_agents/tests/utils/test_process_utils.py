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

"""Tests for subprocess cleanup helpers (process_utils + local_worker)."""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.normpath(os.path.join(_HERE, "..", "..", "python"))
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from akg_agents.core.worker import eval_config  # noqa: E402
from akg_agents.utils import process_utils as pu  # noqa: E402

POSIX = os.name == "posix"


# ---------- resolver / config layer ----------

def test_resolvers_default(monkeypatch):
    for var in ("AKG_EVAL_KILL_GRACE_S", "AKG_EVAL_KILL_DRAIN_S"):
        monkeypatch.delenv(var, raising=False)
    # Use a stale cwd that has no config.yaml so we hit dataclass default.
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    assert eval_config.resolve_kill_grace_s() == 5.0
    assert eval_config.resolve_kill_drain_s() == 2.0


def test_resolvers_env_override(monkeypatch):
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "1.5")
    monkeypatch.setenv("AKG_EVAL_KILL_DRAIN_S", "0.5")
    assert eval_config.resolve_kill_grace_s() == 1.5
    assert eval_config.resolve_kill_drain_s() == 0.5


def test_resolvers_zero_is_valid(monkeypatch):
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "0")
    assert eval_config.resolve_kill_grace_s() == 0.0


def test_resolvers_negative_fallback(monkeypatch):
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "-2")
    assert eval_config.resolve_kill_grace_s() != -2  # fallback to default


def test_resolvers_garbage_fallback(monkeypatch):
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "not-a-number")
    assert eval_config.resolve_kill_grace_s() == 5.0


# ---------- kill_process_tree ----------

def test_kill_already_exited_process_is_noop():
    """No-op when process is already dead; must not raise."""
    p = subprocess.Popen([sys.executable, "-c", "pass"], **pu.popen_process_group_kwargs())
    p.wait()
    assert p.returncode is not None
    pu.kill_process_tree(p, force=True)   # should silently skip


def test_kill_process_tree_no_pid():
    class _Stub:
        pid = None
        def poll(self): return None
    pu.kill_process_tree(_Stub(), force=True)  # must not raise


# ---------- run_command_capture ----------

def test_run_command_capture_success():
    rc, out, err, timed_out = pu.run_command_capture(
        [sys.executable, "-c", "print('hello'); import sys; sys.stderr.write('werr')"],
    )
    assert rc == 0
    assert "hello" in out
    assert "werr" in err
    assert timed_out is False


def test_run_command_capture_does_not_mutate_supplied_env():
    env = {"PATH": os.environ.get("PATH", "")}
    rc, _out, _err, timed_out = pu.run_command_capture(
        [sys.executable, "-c", "pass"], env=env,
    )
    assert rc == 0 and timed_out is False
    assert "PYTHONUNBUFFERED" not in env


@pytest.mark.skipif(not POSIX, reason="PWD is a POSIX process contract")
def test_run_command_capture_updates_pwd_for_cwd(tmp_path):
    rc, out, _err, timed_out = pu.run_command_capture(
        [sys.executable, "-c", "import os; print(os.environ['PWD'])"],
        cwd=str(tmp_path),
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert rc == 0 and timed_out is False
    assert out.strip() == str(tmp_path.resolve())


def test_run_command_capture_timeout(monkeypatch):
    # Force fast kill so the test stays quick.
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "0.5")
    monkeypatch.setenv("AKG_EVAL_KILL_DRAIN_S", "0.5")
    t0 = time.monotonic()
    rc, _out, _err, timed_out = pu.run_command_capture(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=0.5,
    )
    elapsed = time.monotonic() - t0
    assert timed_out is True
    # Must finish well under the 30s sleep (timeout + 2*grace + drain ≈ 2s).
    assert elapsed < 5, f"timeout cleanup took too long: {elapsed:.2f}s"


def test_run_command_capture_cooperative_cancel(monkeypatch):
    """An executor owner can cancel the real child, not only its Future."""
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "0.2")
    monkeypatch.setenv("AKG_EVAL_KILL_DRAIN_S", "0.5")
    cancel_event = threading.Event()
    timer = threading.Timer(0.2, cancel_event.set)
    timer.start()
    started = time.monotonic()
    try:
        _rc, _out, _err, cancelled = pu.run_command_capture(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=30,
            cancel_event=cancel_event,
        )
    finally:
        timer.cancel()
    assert cancelled is True
    assert time.monotonic() - started < 5


@pytest.mark.skipif(not POSIX, reason="worker crash registry is POSIX-only")
def test_normal_completion_clears_worker_process_registry(monkeypatch, tmp_path):
    registry = tmp_path / "groups.json"
    monkeypatch.setenv("AKG_WORKER_PROCESS_REGISTRY", str(registry))
    rc, _out, _err, timed_out = pu.run_command_capture(
        [sys.executable, "-c", "print('ok')"], timeout=5)
    assert rc == 0 and timed_out is False
    assert json.loads(registry.read_text())["groups"] == {}


@pytest.mark.skipif(not POSIX, reason="worker crash registry is POSIX-only")
def test_successor_reaps_registered_group_from_dead_daemon(
        monkeypatch, tmp_path):
    """A SIGKILLed daemon cannot run finally; its successor uses the
    persisted, PID-fingerprinted PGID to remove the orphan tree."""
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "0.2")
    registry = tmp_path / "groups.json"
    monkeypatch.setenv("AKG_WORKER_PROCESS_REGISTRY", str(registry))
    grandchild_pid = tmp_path / "grandchild.pid"
    helper = tmp_path / "registry_helper.py"
    helper.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(30)'])\n"
        f"open({str(grandchild_pid)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        [sys.executable, str(helper)], **pu.popen_process_group_kwargs())
    try:
        pu.register_process_group(process)
        deadline = time.monotonic() + 3
        while not grandchild_pid.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert grandchild_pid.exists()

        data = json.loads(registry.read_text())
        record = data["groups"][str(process.pid)]
        record["owner_pid"] = 999_999_999
        record["owner_start_token"] = "dead-owner"
        registry.write_text(json.dumps(data), encoding="utf-8")

        assert pu.reap_orphaned_process_groups() == [process.pid]
        process.wait(timeout=3)
        gc_pid = int(grandchild_pid.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(gc_pid, 0)
        assert json.loads(registry.read_text())["groups"] == {}
    finally:
        if process.poll() is None:
            pu.kill_process_tree(process, force=True)
            process.wait(timeout=3)


@pytest.mark.skipif(not POSIX, reason="grandchild reaping needs POSIX process groups")
def test_run_command_capture_kills_grandchild(monkeypatch, tmp_path):
    """Parent spawns a grandchild sleeping forever; on timeout, both must die."""
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "0.5")
    pid_file = tmp_path / "gc.pid"
    helper = tmp_path / "helper.py"
    helper.write_text(
        "import os, subprocess, sys, time\n"
        "sub = subprocess.Popen([sys.executable, '-c', 'import time\\nwhile True: time.sleep(0.5)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(sub.pid))\n"
        "time.sleep(30)\n"
    )
    t0 = time.monotonic()
    _rc, _o, _e, timed_out = pu.run_command_capture(
        [sys.executable, str(helper)], timeout=1.0,
    )
    assert timed_out is True
    assert time.monotonic() - t0 < 5
    assert pid_file.exists()
    gc_pid = int(pid_file.read_text().strip())
    # Give signals a moment to propagate up the tree.
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(gc_pid, 0)


@pytest.mark.skipif(not POSIX, reason="grandchild reaping needs POSIX process groups")
def test_timeout_kills_grandchild_after_group_leader_exits(monkeypatch, tmp_path):
    """Regression: an exited leader must not make group cleanup a no-op.

    The child exits immediately, but its grandchild inherits both capture
    pipes.  Popen.communicate therefore times out with returncode already 0;
    cleanup must still signal the original PGID and reap the grandchild.
    """
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "0.2")
    monkeypatch.setenv("AKG_EVAL_KILL_DRAIN_S", "0.5")
    pid_file = tmp_path / "exited_leader_gc.pid"
    helper = tmp_path / "exited_leader.py"
    helper.write_text(
        "import subprocess, sys\n"
        "sub = subprocess.Popen([sys.executable, '-c', "
        "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(sub.pid))\n",
        encoding="utf-8",
    )

    rc, _out, _err, timed_out = pu.run_command_capture(
        [sys.executable, str(helper)], timeout=0.5,
    )
    assert timed_out is True
    assert rc == 0  # the group leader really did exit normally
    gc_pid = int(pid_file.read_text().strip())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(gc_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"grandchild {gc_pid} survived timeout cleanup")


# ---------- process_utils.communicate_or_kill ----------

@pytest.mark.skipif(not POSIX, reason="asyncio.create_subprocess_exec + process groups")
def test_communicate_or_kill_timeout_returns_flag(monkeypatch):
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "0.5")
    from akg_agents.utils.process_utils import communicate_or_kill as _communicate_or_kill

    async def go():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            **pu.popen_process_group_kwargs(),
        )
        return await _communicate_or_kill(proc, timeout=0.5, task_id="t", action="probe")

    out, err, timed_out = asyncio.run(go())
    assert timed_out is True
    assert out == b"" and err == b""


@pytest.mark.skipif(not POSIX, reason="asyncio.create_subprocess_exec + process groups")
def test_communicate_or_kill_cancel_propagates(monkeypatch):
    """Outer asyncio cancel must (a) kill the child group, (b) re-raise CancelledError."""
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "0.5")
    from akg_agents.utils.process_utils import communicate_or_kill as _communicate_or_kill

    async def go():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            **pu.popen_process_group_kwargs(),
        )
        task = asyncio.create_task(
            _communicate_or_kill(proc, timeout=10, task_id="t", action="probe"))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert proc.returncode is not None  # child reaped

    asyncio.run(go())


@pytest.mark.skipif(not POSIX, reason="async process-group test needs POSIX")
def test_communicate_timeout_kills_group_after_leader_exits(monkeypatch, tmp_path):
    monkeypatch.setenv("AKG_EVAL_KILL_GRACE_S", "0.2")
    pid_file = tmp_path / "async_exited_leader_gc.pid"
    helper = tmp_path / "async_exited_leader.py"
    helper.write_text(
        "import subprocess, sys\n"
        "sub = subprocess.Popen([sys.executable, '-c', "
        "'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(sub.pid))\n",
        encoding="utf-8",
    )

    async def go():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(helper),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            **pu.popen_process_group_kwargs(),
        )
        result = await pu.communicate_or_kill(
            proc, timeout=0.5, task_id="t", action="exited-leader")
        return proc, result

    proc, (_out, _err, timed_out) = asyncio.run(go())
    assert timed_out is True
    assert proc.returncode == 0
    gc_pid = int(pid_file.read_text().strip())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(gc_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"grandchild {gc_pid} survived async timeout cleanup")


def test_communicate_or_kill_normal_completion(monkeypatch):
    from akg_agents.utils.process_utils import communicate_or_kill as _communicate_or_kill

    async def go():
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "print('ok')",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            **pu.popen_process_group_kwargs(),
        )
        return await _communicate_or_kill(proc, timeout=10, task_id="t", action="probe")

    out, _err, timed_out = asyncio.run(go())
    assert timed_out is False
    assert b"ok" in out
