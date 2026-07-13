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

"""Pins Claude batch wiring to the shared agent subprocess supervisor."""

import argparse
import io
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "batch"))

import phase_machine  # noqa: E402
import run as R  # noqa: E402
import task_handle  # noqa: E402


class _DummyCM:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _BrokenStdout:
    """Models an SSH/tee consumer that disappeared mid-batch."""

    def write(self, _text):
        raise BrokenPipeError("controlling SSH pipe closed")

    def flush(self):
        raise BrokenPipeError("controlling SSH pipe closed")


def main() -> int:
    failures = []
    captured = {}
    updates = []

    with tempfile.TemporaryDirectory() as tmp:
        task_dir = Path(tmp) / "myop_123_abc123"
        task_dir.mkdir()

        def fake_stream(cmd, cwd, started, timeout_s, log_fp, line_cb=None,
                        extra_env=None):
            captured.update(cmd=cmd, cwd=cwd, timeout_s=timeout_s,
                            extra_env=extra_env)
            return 0, False

        R._stream_subprocess = fake_stream
        R.mf.repo_root = lambda: REPO
        R.mf.now_iso = lambda: "T0"
        R.mf.update_case = lambda _bd, _op, **kw: updates.append(kw)
        R.mf.snapshot_task_dirs = lambda: set()
        R.mf.load_progress = lambda _bd: {
            "cases": {"myop": {"task_dir": str(task_dir)}}
        }
        R.mf.pick_new_task_dir = lambda *_a, **_k: None
        R.mf.read_phase = lambda _td: "FINISH"
        R.mf.read_task_state = lambda _td: {"best_metric": 1.0}
        phase_machine.clear_active_task = lambda **_kw: True
        task_handle.open_task = lambda *_a, **_kw: _DummyCM()

        args = argparse.Namespace(
            claude_bin="claude", model="", extra_claude_arg=[],
            max_rounds=5, eval_timeout=30, timeout_min=60,
        )
        case = {
            "op_name": "myop",
            "ref": "workspace/myop_ref.py",
            "kernel": "workspace/myop_kernel.py",
        }
        rc = R.run_one(REPO, case, args, "--devices 0", io.StringIO())

        # Losing the controlling SSH stdout must not prevent a completed
        # child task from reaching the durable batch manifest.
        updates.clear()
        broken_log = io.StringIO()
        old_stdout = R.sys.stdout
        try:
            R.sys.stdout = _BrokenStdout()
            broken_pipe_rc = R.run_one(
                REPO, case, args, "--devices 0", broken_log)
        finally:
            R.sys.stdout = old_stdout

    checks = [
        (captured.get("cmd", [None])[0] == "claude",
         "Claude command uses shared stream supervisor"),
        (captured.get("extra_env", {}).get("AR_BATCH_OP") == "myop",
         "shared supervisor receives batch identity env"),
        (captured.get("timeout_s") == 3600,
         "shared supervisor receives wall-clock budget"),
        (any(u.get("status") == "done" and u.get("rc") == 0
             for u in updates),
         "FINISH result recorded"),
        (rc == 0, "run_one returns success"),
        (broken_pipe_rc == 0,
         "closed SSH stdout does not abort result harvesting"),
        (any(u.get("status") == "done" and u.get("rc") == 0
             for u in updates),
         "FINISH result persists after console EPIPE"),
        ("launching: claude --print" in broken_log.getvalue()
         and "exited rc=0" in broken_log.getvalue(),
         "durable batch log survives console EPIPE"),
    ]

    stale_progress = {
        "cases": {
            "myop": {
                "status": "running",
                "task_dir": str(task_dir),
                "runner_pid": 999999,
                "note": "",
            }
        }
    }
    R._pid_alive = lambda _pid: False
    phase_machine.is_task_active = lambda _td: False
    demoted, harvested = R.recover_stale_running(stale_progress)
    harvested_case = stale_progress["cases"]["myop"]
    checks.extend([
        (demoted == 0 and harvested == 1,
         "stale recovery distinguishes completed task from incomplete orphan"),
        (harvested_case.get("status") == "done"
         and harvested_case.get("final_phase") == "FINISH"
         and harvested_case.get("result", {}).get("best_metric") == 1.0,
         "stale recovery harvests authoritative FINISH result"),
    ])
    for ok, label in checks:
        print(("[ok]   " if ok else "[FAIL] ") + label)
        if not ok:
            failures.append(label)
    if failures:
        print(f"\n{len(failures)} Claude wiring checks failed")
        return 1
    print(f"\nAll {len(checks)} Claude shared-supervisor checks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
