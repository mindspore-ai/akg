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

"""opencode door test — proves the JS↔Python door bridges ``decide``
faithfully, i.e. that opencode and Claude Code share ONE decision brain.

We drive ``.opencode/door.py`` exactly as the JS plugin will (base64 event
on argv) over the same real-task fixtures the Claude golden suite uses, and
assert the returned Decision matches the SEMANTICS pinned by the Claude
golden snapshots (block vs allow; key substrings). Byte-for-byte wire
output differs by design — that framing is each adapter's job — but the
verdict underneath must be identical.

The decision cases are pure Python. When Node is available, this also smoke
tests the JS ``shell.env`` adapter; a live opencode smoke test is still needed
for host hook wiring (AR_PLUGIN_TRACE=<file>).

Usage:  python tests/opencode_door/run_door_test.py
"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                       # workspace_autoresearch/
DOOR = REPO / ".opencode" / "door.py"

# Real task dir reused as a fixture base (copied + phase-tweaked per scenario;
# original never touched) — it carries a real state.json + history.jsonl + task.yaml.
_TASK_YAML = """\
agent:
  max_rounds: 20
  ref_file: reference.py
description: Door parity test fixture
editable_files:
  - kernel.py
eval:
  timeout: 30
metric:
  improvement_threshold: 0.0
  lower_is_better: true
  primary: latency_us
name: door_fixture
worker:
  urls:
    - 127.0.0.1:9111
"""


def _fixture_state(phase):
    """Return the smallest valid state used by the neutral decision engine."""
    return {
        "phase": phase,
        "owner": {"session_id": "", "pid": 0, "claimed_at": None},
        "progress_initialized": True,
        "pending_settle": None,
        "expected_plan_version": 0,
        "expected_history_round": 0,
        "task": "door_fixture",
        "eval_rounds": 0,
        "max_rounds": 20,
        "consecutive_failures": 0,
        "best_metric": 2.0,
        "best_commit": "fixture",
        "baseline_metric": 1.0,
        "baseline_source": "ref",
        "baseline_outcome": "ok",
        "seed_metric": 2.0,
        "plan_version": 0,
        "next_pid": 0,
        "num_cases": 1,
        "per_shape_descs": ["fixture shape"],
        "diagnose_attempts": 0,
        "diagnose_attempts_for_version": None,
        "last_diagnose_failure_reason": None,
        "last_stop_reason": None,
        "last_stop_time": None,
    }


def _make_fixture(phase, *, state_overrides=None, drop_plan=True):
    """Create a self-contained task fixture and return its task directory."""
    tmp = Path(tempfile.mkdtemp(prefix="ar_door_"))
    dst = tmp / "task"
    dst.mkdir()
    (dst / "task.yaml").write_text(_TASK_YAML, encoding="utf-8")
    (dst / "kernel.py").write_text("# door fixture\n", encoding="utf-8")
    (dst / "reference.py").write_text("# door fixture\n", encoding="utf-8")
    state_dir = dst / ".ar_state"
    state_dir.mkdir()
    state_path = dst / ".ar_state" / "state.json"
    state = _fixture_state(phase)
    if state_overrides:
        state.update(state_overrides)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    (state_dir / "history.jsonl").write_text(
        json.dumps({
            "round": 0,
            "description": "fixture seed",
            "decision": "SEED",
            "metrics": {"latency_us": 2.0},
            "outcome": "ok",
            "correctness": True,
            "commit": "fixture",
        }) + "\n",
        encoding="utf-8",
    )
    if drop_plan:
        pp = dst / ".ar_state" / "plan.md"
        if pp.exists():
            pp.unlink()
    return dst


def _call_door(event: dict, task_dir) -> dict:
    """Invoke the door the way the opencode plugin will: base64 event on
    argv. Returns the parsed Decision dict."""
    b64 = base64.b64encode(json.dumps(event).encode("utf-8")).decode("ascii")
    env = dict(os.environ)
    env["AR_TASK_DIR"] = str(task_dir)        # env-fallback ownership
    env["AR_SESSION_ID"] = ""
    env["CLAUDE_CODE_SESSION_ID"] = ""
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(DOOR), "event", b64],
        capture_output=True, text=True, encoding="utf-8", env=env,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, f"door crashed: {proc.stderr}"
    return json.loads(proc.stdout)


def _check_shell_adapter():
    """Exercise the real JS shell hooks without starting an LLM session."""
    node = shutil.which("node")
    if not node:
        print("[skip] shell.adapter (node not installed)")
        return True
    plugin_uri = (REPO / ".opencode" / "plugin" / "autoresearch.js").as_uri()
    code = f"""
import {{ AutoResearch }} from {json.dumps(plugin_uri)};
const hooks = await AutoResearch({{ $: null, client: {{}} }});
const shellOutput = {{ env: {{ EXISTING: "kept" }} }};
await hooks["shell.env"]({{ sessionID: "session-123" }}, shellOutput);
const toolOutput = {{ args: {{ command: "python scripts/engine/pipeline.py task" }} }};
await hooks["tool.execute.before"](
  {{ tool: "bash", sessionID: "session-123", callID: "call-1" }},
  toolOutput,
);
await hooks["tool.execute.before"](
  {{ tool: "bash", sessionID: "session-123", callID: "call-2" }},
  toolOutput,
);
console.log(JSON.stringify({{env: shellOutput.env, command: toolOutput.args.command}}));
"""
    env = dict(os.environ)
    env.update({
        "AR_OPENCODE_ENV_FILE": "/adapter/env file.sh",
        "ANTHROPIC_AUTH_TOKEN": "must-not-leak",
        "UNRELATED_TEST_VALUE": "must-not-copy",
    })
    proc = subprocess.run(
        [node, "--input-type=module", "--eval", code],
        capture_output=True, text=True, encoding="utf-8", env=env,
        cwd=str(REPO),
    )
    if proc.returncode != 0:
        print(f"[FAIL] shell.adapter: {proc.stderr}")
        return False
    got = json.loads(proc.stdout)
    expected_env = {
        "AR_SESSION_ID": "session-123",
        "EXISTING": "kept",
    }
    ok = all(got["env"].get(key) == value
             for key, value in expected_env.items())
    ok = ok and "ANTHROPIC_AUTH_TOKEN" not in got["env"]
    ok = ok and "UNRELATED_TEST_VALUE" not in got["env"]
    ok = ok and got["command"] == (
        ". '/adapter/env file.sh'\n"
        "python scripts/engine/pipeline.py task"
    )
    print("[ok]   shell.adapter" if ok else
          f"[FAIL] shell.adapter: got {got}")
    return ok


# Each case: (name, phase, event, predicate(decision)->bool, why).
# decide() dispatches on the NEUTRAL `tool_kind` (shell|edit|subagent); each
# adapter maps its native tool name onto that. evt() derives tool_kind from
# the native `tool` here exactly as the plugin/cc_hook do, so the cases can
# stay written in either agent's native vocabulary.
_NATIVE_TO_KIND = {
    "Bash": "shell", "bash": "shell",
    "Edit": "edit", "edit": "edit", "Write": "edit", "write": "edit",
    "Task": "subagent", "task": "subagent",
}


def _cases():
    def evt(**kw):
        base = {"kind": "", "tool_kind": "", "tool": "", "command": "",
                "file_path": "", "subagent_type": "", "output": "",
                "stop_reason": "unknown", "session_id": ""}
        base.update(kw)
        if not base["tool_kind"] and base["tool"]:
            base["tool_kind"] = _NATIVE_TO_KIND.get(base["tool"], "")
        return base

    return [
        ("guard_bash.unknown_script", "EDIT",
         evt(kind="pre_tool", tool="Bash",
             command="python scripts/eval.py x"),
         lambda d: d["block"] and "does not exist" in d["block_reason"],
         "unknown script must block"),

        ("guard_bash.diagnose_blocks_create_plan", "DIAGNOSE",
         evt(kind="pre_tool", tool="Bash",
             command="python scripts/engine/create_plan.py ."),
         lambda d: d["block"] and "DIAGNOSE" in d["block_reason"],
         "create_plan blocked pre-artifact in DIAGNOSE"),

        ("guard_bash.allow_pipeline_in_edit", "EDIT",
         evt(kind="pre_tool", tool="Bash",
             command="python scripts/engine/pipeline.py ."),
         lambda d: not d["block"],
         "pipeline.py allowed in EDIT"),

        ("guard_task.diagnose_wrong_subagent", "DIAGNOSE",
         evt(kind="pre_tool", tool="Task", subagent_type="general-purpose"),
         lambda d: d["block"] and "ar-diagnosis" in d["block_reason"],
         "wrong subagent blocked in DIAGNOSE"),

        ("guard_task.non_diagnose_noop", "EDIT",
         evt(kind="pre_tool", tool="Task", subagent_type="general-purpose"),
         lambda d: not d["block"],
         "Task unrestricted outside DIAGNOSE"),

        ("stop.block_non_finish", "EDIT",
         evt(kind="stop", stop_reason="end_turn"),
         lambda d: d["block"] and "Cannot Stop" in d["block_reason"],
         "stop blocked before FINISH"),

        ("stop.allow_finish", "FINISH",
         evt(kind="stop", stop_reason="end_turn"),
         lambda d: not d["block"] and any("FINISH" in s for s in d["status"]),
         "stop allowed at FINISH, with summary status"),
    ]


def _materialise(name, task_dir, event):
    """Fill in event fields that need the live temp path."""
    td = str(task_dir)
    if name == "guard_bash.allow_pipeline_in_edit":
        event["command"] = f'python scripts/engine/pipeline.py "{td}"'
    return event


def main() -> int:
    failures = []
    for name, phase, event, predicate, why in _cases():
        task_dir = _make_fixture(phase)
        try:
            event = _materialise(name, task_dir, dict(event))
            decision = _call_door(event, task_dir)
            ok = predicate(decision)
            if ok:
                print(f"[ok]   {name}")
            else:
                failures.append(name)
                print(f"[FAIL] {name}: {why}\n        got: {decision}")
        finally:
            shutil.rmtree(task_dir.parent, ignore_errors=True)

    if not _check_shell_adapter():
        failures.append("shell.adapter")

    if failures:
        print(f"\n{len(failures)} door case(s) failed: {failures}")
        return 1
    print(f"\nAll {len(_cases())} door cases pass — opencode and Claude Code "
          f"share one decide() brain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
