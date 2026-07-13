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

"""Wiring test for batch run.py --agent opencode.

Proves run_one_opencode builds the right `run_loop.py` command, binds the
task_dir from run_loop's `[run_loop] task_dir=` line, and records the
manifest case — all WITHOUT spawning opencode or needing NPU. The actual
opencode↔plugin↔decide loop is proven separately (tests/opencode_door +
the live run_loop tests); here we only pin the batch-side wiring so the
Claude path and the opencode path stay in lockstep on bookkeeping.

Usage:  python tests/batch/run_opencode_wiring_test.py
"""
import argparse
import importlib.util
import io
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                    # workspace_autoresearch/
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "batch"))

import run as R                            # noqa: E402
import phase_machine                       # noqa: E402
import task_handle                         # noqa: E402

_RL_SPEC = importlib.util.spec_from_file_location(
    "autoresearch_run_loop", REPO / ".opencode" / "run_loop.py")
RUN_LOOP = importlib.util.module_from_spec(_RL_SPEC)
_RL_SPEC.loader.exec_module(RUN_LOOP)


class _DummyCM:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _make_args(**over):
    base = dict(max_rounds=5, eval_timeout=30, model="", timeout_min=60,
                agent="opencode")
    base.update(over)
    return argparse.Namespace(**base)


def _run(case, hw_arg, *, stream_lines, rc=0, phase="FINISH",
         progress_task_dir=None):
    """Drive run_one_opencode with all process/IO/state calls stubbed.
    Returns (captured_cmd, update_calls, result_rc)."""
    captured = {"cmd": None, "extra_env": None}
    updates = []

    def fake_stream(cmd, cwd, started, timeout_s, log_fp, line_cb=None,
                    extra_env=None):
        captured["cmd"] = cmd
        captured["extra_env"] = extra_env
        for ln in stream_lines:
            if line_cb:
                line_cb(ln)
        return rc, False

    # Stub the agent-neutral primitives so nothing real is spawned/written.
    R._stream_subprocess = fake_stream
    R.mf.update_case = lambda batch_dir, op, **kw: updates.append(kw)
    R.mf.now_iso = lambda: "T0"
    R.mf.repo_root = lambda: REPO
    R.mf.load_progress = lambda batch_dir: {
        "cases": {"myop": {"task_dir": progress_task_dir}}
    }
    R.mf.read_phase = lambda td: phase
    R.mf.read_task_state = lambda td: {"best_metric": 1.0}
    phase_machine.clear_active_task = lambda **kw: True
    task_handle.open_task = lambda *a, **k: _DummyCM()

    rc_out = R.run_one_opencode(REPO, case, _make_args(), hw_arg,
                                io.StringIO(), prev_task_dir=None)
    return captured, updates, rc_out


def main() -> int:
    fails = []
    case = {"op_name": "myop", "ref": "workspace/myop_ref.py",
            "kernel": "workspace/myop_kernel.py"}

    # --- Case 1: run_loop reports a task_dir + FINISH → status done ---
    with tempfile.TemporaryDirectory() as tmp:
        task_dir = Path(tmp) / "myop_123_abc123"
        task_dir.mkdir()
        td = str(task_dir.resolve())
        captured, updates, rc = _run(
            case, "--devices 0",
            stream_lines=[f"[run_loop] task_dir={td}\n", "some log\n"],
            rc=0, phase="FINISH")
    cmd = captured["cmd"]
    extra_env = captured["extra_env"] or {}

    cmd_s = " ".join(cmd or [])
    checks = [
        (cmd is not None, "command was built"),
        (str(REPO / ".opencode" / "run_loop.py") in cmd_s, "invokes run_loop.py"),
        ("--ref workspace/myop_ref.py" in cmd_s, "passes --ref"),
        ("--kernel workspace/myop_kernel.py" in cmd_s, "passes --kernel"),
        ("--op-name myop" in cmd_s, "passes --op-name"),
        ("--max-rounds 5" in cmd_s, "passes --max-rounds"),
        ("--devices 0" in cmd_s, "forwards hw flag"),
        (extra_env.get("AR_BATCH_OP") == "myop", "passes batch op env"),
        (extra_env.get("AR_BATCH_DIR") == str(REPO.resolve()),
         "passes batch dir env"),
        ("--max-iters" not in cmd_s,
         "no --max-iters (bounded by decide(stop) + outer timeout, like claude)"),
        (any(u.get("task_dir") for u in updates), "bound task_dir into manifest"),
        (any(u.get("status") == "done" and u.get("final_phase") == "FINISH"
             for u in updates), "recorded status=done at FINISH"),
        (rc == 0, "returns rc=0 on done"),
    ]
    for ok, label in checks:
        if ok:
            print(f"[ok]   case1: {label}")
        else:
            fails.append(f"case1: {label}")
            print(f"[FAIL] case1: {label}")

    # --- Case 2: run_loop never reports a task_dir → error, rc!=0 ---
    captured2, updates2, rc2 = _run(case, "--worker-url h:1",
                                    stream_lines=["nothing useful\n"],
                                    rc=2, phase="PLAN")
    cmd2 = captured2["cmd"]
    c2 = [
        (rc2 == 2, "returns 2 when no task_dir bound"),
        (any(u.get("status") == "error" for u in updates2), "records error"),
        ("--worker-url h:1" in " ".join(cmd2 or []), "forwards worker-url hw flag"),
    ]
    for ok, label in c2:
        if ok:
            print(f"[ok]   case2: {label}")
        else:
            fails.append(f"case2: {label}")
            print(f"[FAIL] case2: {label}")

    # --- Case 3: no marker, scaffold already recorded task_dir in progress ---
    with tempfile.TemporaryDirectory() as tmp:
        td3 = Path(tmp) / "myop_123_abc123"
        td3.mkdir()
        _, updates3, rc3 = _run(
            case, "--devices 0",
            stream_lines=["ordinary log\n"], rc=0, phase="FINISH",
            progress_task_dir=str(td3))
    c3 = [
        (rc3 == 0, "returns rc=0 from progress task_dir fallback"),
        (any(u.get("task_dir") == str(td3.resolve()) for u in updates3),
         "uses scaffold-recorded task_dir"),
        (any(u.get("status") == "done" for u in updates3),
         "records done from progress task_dir fallback"),
    ]
    for ok, label in c3:
        if ok:
            print(f"[ok]   case3: {label}")
        else:
            fails.append(f"case3: {label}")
            print(f"[FAIL] case3: {label}")

    # --- Case 4: pin sessions across both supported OpenCode log layouts ---
    old_match = RUN_LOOP._SESSION_RE.search(
        "message=created id=ses_old123 slug=fixture")
    current_match = RUN_LOOP._SESSION_RE.search(
        "service=session id=ses_new456 slug=fixture created")
    c4 = [
        (old_match and old_match.group(1) == "ses_old123",
         "captures legacy session log"),
        (current_match and current_match.group(1) == "ses_new456",
         "captures OpenCode 1.14 session log"),
        ("ProviderModelNotFoundError" in RUN_LOOP._FATAL_PROVIDER_MARKERS,
         "classifies missing configured model as fatal"),
    ]
    for ok, label in c4:
        if ok:
            print(f"[ok]   case4: {label}")
        else:
            fails.append(f"case4: {label}")
            print(f"[FAIL] case4: {label}")

    if fails:
        print(f"\n{len(fails)} wiring check(s) failed:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"\nAll {len(checks) + len(c2) + len(c3) + len(c4)} "
          "opencode-batch wiring checks pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
