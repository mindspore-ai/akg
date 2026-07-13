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

"""Agent-neutral decision layer for the autoresearch phase machine.

This module is the SINGLE place that answers, for any ReAct agent harness
(Claude Code, opencode, …): "given a normalised lifecycle event and the
on-disk task state, what should the agent be told to do?" It sits one rung
above ``phase_machine`` and observes transactions exposed by
``workflow``/``task_handle``. Agent adapters live in
that agent's own mount point (``.claude/`` hooks, ``.opencode/`` plugin)
and do nothing but translate: native event → ``AgentEvent``, then
``Decision`` → that agent's native response mechanism.

Layering (no import cycle):
    adapters (.claude/hooks, .opencode)  ──►  decide  ──►  phase_machine
                                                     └────►  workflow / task_handle

Contract:
  * Input  — ``AgentEvent`` (normalised; no agent vocabulary).
  * Output — ``Decision`` (normalised channels: block / status / context /
    todos). The adapter frames each non-empty channel into wire bytes.
  * Business events commit their own phase before post-tool reporting.
    decide() only mutates hook-owned lifecycle metadata (activation,
    ownership heartbeat, DIAGNOSE attempts, stop trace).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Optional

# decide sits above phase_machine and may pull in workflow / task_handle /
# task_config / utils — all under scripts/, which phase_machine already
# puts on sys.path. Imports are kept lazy inside handlers where the
# originals were lazy (to dodge package-init cycles), eager otherwise.
from phase_machine import (
    BASELINE, EDIT, DIAGNOSE, REPLAN, FINISH,
    DIAGNOSE_ATTEMPTS_CAP,
    DIAGNOSE_NEED_DIAGNOSIS, DIAGNOSE_READY, DIAGNOSE_MANUAL_FALLBACK,
    read_phase, get_guidance, get_task_dir, set_task_dir, touch_heartbeat,
    load_progress, update_progress, load_state,
    check_bash, check_edit, parse_invoked_ar_script, parse_script_names,
    is_single_foreground_ar_invocation, diagnose_state,
    diagnose_artifact_path, diagnose_marker,
    edit_marker_path, state_record_path, history_path, plan_path,
    get_active_item, get_plan_items,
)

WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}

# Mirrors guard_bash._BLESSED_SCRIPTS / _LIBRARY_NOT_CLI verbatim.
_BLESSED_SCRIPTS = {
    "scaffold.py", "baseline.py", "dashboard.py",
    "create_plan.py", "pipeline.py", "resume.py",
    "parse_args.py",
}
_LIBRARY_NOT_CLI = {
    "eval_kernel.py": (
        "eval_kernel.py is a subprocess child of utils.eval_runner.local_eval, "
        "not a CLI. Run `python scripts/engine/baseline.py "
        "<task_dir>` instead — it drives eval_kernel.py for you via "
        "task_config.run_eval."
    ),
    "quick_check.py": (
        "quick_check.py is a subprocess child of pipeline.py "
        "(EDIT-phase smoke check), not a CLI. Run `python "
        "scripts/engine/pipeline.py <task_dir>` after your edit — it "
        "calls quick_check.py for you before running eval."
    ),
    "report.py": (
        "report.py is auto-invoked by pipeline.py at FINISH; you don't "
        "run it manually. The generated report lives at "
        "<task_dir>/.ar_state/report.md after the FINISH-phase pipeline."
    ),
}
_REQUIRED_SUBAGENT = "ar-diagnosis"


# ---------------------------------------------------------------------------
# Normalised I/O
# ---------------------------------------------------------------------------
@dataclass
class AgentEvent:
    """A lifecycle event from any agent harness, with all agent-specific
    vocabulary already stripped by the adapter.

    kind: "pre_tool" | "post_tool" | "stop"  — the lifecycle moment.
    tool_kind: the NEUTRAL tool taxonomy decide() dispatches on —
          "shell" | "edit" | "subagent" | "" . Each adapter maps ITS native
          tool names onto this (Claude: Bash->shell, Edit/Write->edit,
          Task->subagent; opencode: bash->shell, edit/write->edit,
          task->subagent). decide() never sees a native tool name, so no
          agent's vocabulary is privileged as "the default".
    tool: the raw native tool name (Claude "Bash", opencode "bash", …) — kept
          for human-readable logging only; NOT used for dispatch.
    """
    kind: str
    tool_kind: str = ""        # neutral: "shell" | "edit" | "subagent"
    tool: str = ""             # raw native name (logging only)
    command: str = ""          # pre/post shell
    file_path: str = ""        # pre/post edit
    subagent_type: str = ""    # pre subagent (the subagent NAME, e.g. ar-diagnosis)
    output: str = ""           # post shell (tool stdout)
    stop_reason: str = "unknown"
    session_id: str = ""       # informational; ownership still via env seam


@dataclass
class Decision:
    """What to tell the agent. Each field is an independent output channel;
    the adapter renders the non-empty ones into its native wire format.

    block        — refuse the action (pre_tool → typically a hard stop;
                   stop → keep the loop going). block_reason carries the text.
    status       — human-visible lines (Claude: stderr). Ordered.
    context      — model-visible instruction for the next turn, already
                   composed as plain text (Claude: PostToolUse
                   additionalContext). Mutually exclusive with todos.
    todos_header / todos — when todos_header is not None, the adapter should
                   mirror the live plan into the agent's todo UI (Claude:
                   TodoWrite). ``todos`` is the already-projected payload
                   list; the CC-specific envelope text lives in the adapter.
                   Agents without a todo UI drop this channel.
    """
    block: bool = False
    block_reason: str = ""
    status: list = field(default_factory=list)
    context: Optional[str] = None
    todos_header: Optional[str] = None
    todos: Optional[list] = None


class _Block(Exception):
    """Internal: a handler reached a block verdict. Caught by decide()."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _Sink:
    """Accumulates a handler's outputs so the logic can keep the original
    emit-style call sites while decide() returns them as a Decision."""
    def __init__(self):
        self.status: list = []
        self.context: Optional[str] = None
        self.todos_header: Optional[str] = None
        self.todos: Optional[list] = None

    # --- emit primitives, 1:1 with the old hooks.utils helpers ---
    def emit_status(self, msg: str):
        self.status.append(msg)

    def block(self, reason: str):
        raise _Block(reason)

    def block_with_guidance(self, task_dir: str, reason: str):
        raise _Block(f"[AR] {reason}. {get_guidance(task_dir)}")

    def emit_context(self, text: str):
        self.context = text

    def emit_todowrite(self, task_dir: str, header: str):
        """Project the current plan into the structured todo payload. The
        tool-specific (best-effort) mirror envelope is the adapter's job;
        here we only decide WHAT the live list is (agent-neutral)."""
        live = [it for it in get_plan_items(task_dir) if not it["done"]]
        todos = []
        for it in live:
            status = "in_progress" if it["active"] else "pending"
            todos.append({
                "content": f"[{it['id']}] {it['description'][:80]}",
                "activeForm": f"Working on {it['id']}: {it['description'][:60]}",
                "status": status,
            })
        self.todos_header = header
        self.todos = todos


# ---------------------------------------------------------------------------
# Shared helpers (ported from the hook modules)
# ---------------------------------------------------------------------------
def _activation_target(command: str) -> Optional[str]:
    if "AR_TASK_DIR=" not in command:
        return None
    try:
        tokens = shlex.split(command, posix=True, comments=False)
    except ValueError:
        return None
    for tok in tokens:
        if tok.startswith("AR_TASK_DIR="):
            return tok[len("AR_TASK_DIR="):] or None
    return None


def _task_dir_from_command(command: str) -> Optional[str]:
    try:
        tokens = shlex.split(command, posix=True, comments=False)
    except ValueError:
        return None
    for tok in tokens:
        if tok and os.path.isfile(os.path.join(tok, ".ar_state", "state.json")):
            return os.path.abspath(tok)
    return None


def _has_pending_settle(task_dir: str) -> bool:
    state = load_state(task_dir)
    return bool(state and state.get("pending_settle"))


def _task_dir_from_edit_target(file_path: str) -> Optional[str]:
    p = os.path.abspath(file_path)
    while True:
        parent = os.path.dirname(p)
        if parent == p:
            return None
        if os.path.isfile(os.path.join(parent, ".ar_state", "state.json")):
            return parent
        p = parent


def _rel_to_task(file_path: str, task_dir: str) -> Optional[str]:
    fp_native = os.path.normpath(os.path.abspath(file_path))
    td_native = os.path.normpath(os.path.abspath(task_dir))
    try:
        if os.path.commonpath([fp_native, td_native]) != td_native:
            return None
    except ValueError:
        return None
    return os.path.relpath(file_path, task_dir).replace("\\", "/")


def _is_stuck(phase: str, progress) -> bool:
    return phase == BASELINE and progress is None


def _baseline_message(outcome, new_phase, progress, guidance) -> str:
    if outcome != "ok":
        reason = ("seed kernel produced no timing"
                  if progress.seed_metric is None
                  else "seed kernel failed correctness / profile")
        return (f"[AR] Baseline failed: {reason}. Phase -> {new_phase}. Plan a "
                f"kernel fix/rewrite via the standard plan->edit loop. "
                f"{guidance}")
    return f"[AR] Baseline complete. Phase -> {new_phase}. {guidance}"


# ---------------------------------------------------------------------------
# pre_tool / Bash  (was guard_bash.main)
# ---------------------------------------------------------------------------
def _script_name_check(command: str, sink: _Sink):
    from utils.settings import hallucinated_scripts
    aliases = hallucinated_scripts()
    for script_path, script_name in parse_script_names(command):
        if script_name in aliases:
            real = aliases[script_name]
            sink.block(f"[AR] '{script_name}' does not exist. "
                       f"Use: python scripts/{real}")
        if script_name in _LIBRARY_NOT_CLI:
            sink.block(f"[AR] {_LIBRARY_NOT_CLI[script_name]}")
        if script_name not in _BLESSED_SCRIPTS:
            sink.block(f"[AR] Unknown script '{script_name}'. "
                       f"Valid scripts: {sorted(_BLESSED_SCRIPTS)}")


def _pre_bash(event: AgentEvent, sink: _Sink):
    task_dir = get_task_dir()
    if not task_dir:
        return
    touch_heartbeat(task_dir)

    command = event.command
    _script_name_check(command, sink)

    phase = read_phase(task_dir)
    invoked = parse_invoked_ar_script(command)

    if phase == DIAGNOSE and invoked == "create_plan.py":
        state = diagnose_state(task_dir)
        if state.action == DIAGNOSE_NEED_DIAGNOSIS:
            sink.block(
                f"[AR] create_plan.py blocked in DIAGNOSE: artifact "
                f"check failed ({state.artifact_reason}). Spawn the "
                f"ar-diagnosis subagent first; only after "
                f"the artifact validates may you run create_plan.py. "
                f"(Subagent attempts so far: {state.attempts}/"
                f"{DIAGNOSE_ATTEMPTS_CAP}; at the cap the gate is "
                f"relaxed and you may write the plan directly.)"
            )

    if phase == EDIT and invoked == "create_plan.py" \
            and _has_pending_settle(task_dir):
        ok, reason = is_single_foreground_ar_invocation(
            command, script="create_plan.py")
        if not ok:
            sink.block(
                f"[AR] Recovery path requires a single foreground "
                f"create_plan.py invocation while state.pending_settle "
                f"is set: {reason}. Re-issue without chaining; FD "
                f"redirects (`2>&1`, `> log.txt`) are fine."
            )
        ok, reason = check_bash(REPLAN, command)
        if not ok:
            sink.block_with_guidance(task_dir, reason)
        return

    ok, reason = check_bash(phase, command)
    if not ok:
        sink.block_with_guidance(task_dir, reason)


# ---------------------------------------------------------------------------
# pre_tool / Edit-family  (was guard_edit.main)
# ---------------------------------------------------------------------------
def _edit_phase_git_gate(task_dir: str, editable_files, sink: _Sink):
    try:
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=task_dir, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5,
        ).stdout.strip()
    except Exception:
        return

    marker = edit_marker_path(task_dir)
    if os.path.exists(marker):
        return

    for ef in editable_files:
        rel_in_repo = os.path.relpath(os.path.join(task_dir, ef), repo_root)
        try:
            diff = subprocess.run(
                ["git", "diff", "--name-only", "--", rel_in_repo],
                cwd=repo_root, capture_output=True, text=True, timeout=5,
            )
        except Exception:
            continue
        if diff.stdout.strip():
            sink.block(
                f"[AR] Uncommitted change in {ef!r} on entry to EDIT phase. "
                f"Likely an unfinalized previous round, but could also be "
                f"a seed commit that didn't land or an off-flow edit. "
                f"Run pipeline.py to settle the current diff into a round "
                f"before editing more: "
                f"python scripts/engine/pipeline.py \"{task_dir}\""
            )

    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        f.write("1")


def _pre_edit(event: AgentEvent, sink: _Sink):
    file_path = event.file_path
    if not file_path:
        return

    target_task = _task_dir_from_edit_target(file_path)

    if target_task:
        touch_heartbeat(target_task)
        rel = _rel_to_task(file_path, target_task)

        from task_config import load_task_config
        config = load_task_config(target_task)
        editable_files = list(config.editable_files) if config else []

        phase = read_phase(target_task)
        diag_action = (diagnose_state(target_task).action
                       if phase == DIAGNOSE else None)
        pending = (phase == EDIT and _has_pending_settle(target_task))
        ok, reason = check_edit(phase, rel, editable_files,
                                diagnose_action=diag_action,
                                pending_settle=pending)
        if not ok:
            sink.block_with_guidance(target_task, reason)

        if phase == EDIT and rel in set(editable_files):
            _edit_phase_git_gate(target_task, editable_files, sink)
        return

    owned = get_task_dir()
    if owned:
        sink.block_with_guidance(
            owned,
            f"Edit target {file_path!r} is outside the active task "
            f"directory ({owned}). The agent's scope is the task_dir "
            f"only — source files in workspace/, repo configs, hooks, "
            f"and anything else outside are off-limits. If you need to "
            f"change a source --ref or --kernel file, exit the task and "
            f"re-run /autoresearch with the corrected source."
        )


# ---------------------------------------------------------------------------
# pre_tool / subagent  (was guard_task.main; tool_kind == "subagent")
# ---------------------------------------------------------------------------
def _pre_task(event: AgentEvent, sink: _Sink):
    task_dir = get_task_dir()
    if not task_dir:
        return
    touch_heartbeat(task_dir)

    if read_phase(task_dir) != DIAGNOSE:
        return

    state = diagnose_state(task_dir)
    if state.action == DIAGNOSE_READY:
        sink.block(
            f"[AR] DIAGNOSE artifact already validated for plan_version="
            f"{state.plan_version}. Do not re-spawn the subagent; write "
            f".ar_state/plan_items.xml from the diagnosis and run "
            f"create_plan.py."
        )

    if state.action == DIAGNOSE_MANUAL_FALLBACK:
        sink.block(
            f"[AR] DIAGNOSE subagent already failed "
            f"{DIAGNOSE_ATTEMPTS_CAP} times for plan_version="
            f"{state.plan_version}. Switch to manual planning: read "
            f".ar_state/history.jsonl + plan.md, Write <items>...</items> "
            f"to .ar_state/plan_items.xml, then run create_plan.py. The "
            f"artifact gate on create_plan.py is relaxed in this state."
        )

    if event.subagent_type != _REQUIRED_SUBAGENT:
        sink.block(
            f"[AR] DIAGNOSE phase requires the '{_REQUIRED_SUBAGENT}' "
            f"subagent, not {event.subagent_type!r}. Re-spawn with the "
            f"subagent set exactly to '{_REQUIRED_SUBAGENT}' (your harness's "
            f"'{_REQUIRED_SUBAGENT}' agent definition) — it is the only "
            f"diagnostician the host accepts here."
        )


# ---------------------------------------------------------------------------
# post_tool / Bash  (was post_bash.main)
# ---------------------------------------------------------------------------
def _clean_stale_edit_marker(task_dir: str, sink: _Sink):
    marker = edit_marker_path(task_dir)
    if not os.path.exists(marker):
        return
    from utils.git_utils import is_working_tree_clean
    if is_working_tree_clean(task_dir):
        try:
            os.remove(marker)
            sink.emit_status("[AR] Cleaned stale edit marker (git is clean).")
        except OSError:
            pass


def _handle_activation(new_task_dir: str, sink: _Sink):
    new_task_dir = os.path.abspath(new_task_dir)
    if not os.path.isdir(new_task_dir):
        sink.emit_status(f"[AR] ERROR: task_dir not found: {new_task_dir}")
        return

    has_state = os.path.exists(state_record_path(new_task_dir))

    from task_handle import (
        open_task as _open_task, Role as _Role,
        TaskOwnershipError as _Ownership,
        TaskConsistencyError as _Consistency,
    )
    try:
        with _open_task(new_task_dir, role=_Role.AGENT) as t:
            _clean_stale_edit_marker(new_task_dir, sink)
            previous = t.phase
            phase = t.activate(fresh=not has_state)
            if not has_state:
                sink.emit_status(f"[AR] Fresh start. Phase -> BASELINE. "
                                 f"{get_guidance(new_task_dir)}")
                return

            if phase != previous:
                sink.emit_status(
                    f"[AR] Recovered executable phase {previous} -> {phase} "
                    f"from committed task state.")
            sink.emit_status(f"[AR] Resuming. Phase: {phase}.")
            _print_resume_context(new_task_dir, sink)
            sink.emit_status(get_guidance(new_task_dir))
    except _Consistency as e:
        sink.emit_status(f"[AR] {e}")
    except _Ownership as e:
        sink.emit_status(
            f"[AR] ERROR: refused to activate {new_task_dir} — {e} "
            f"Stop the other session before retrying."
        )


def _print_resume_context(task_dir: str, sink: _Sink):
    progress = load_progress(task_dir)
    if not progress:
        return
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", "?")
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    failures = progress.get("consecutive_failures", 0)
    plan_ver = progress.get("plan_version", 0)

    sink.emit_status(
        f"[AR] Resume context: Round {rounds}/{max_rounds} | "
        f"Best: {best} | Baseline: {baseline} | "
        f"Failures: {failures} | Plan v{plan_ver}"
    )

    hpath = history_path(task_dir)
    if os.path.exists(hpath):
        with open(hpath, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            sink.emit_status(f"[AR] Last {min(3, len(lines))} rounds:")
            for rec in lines[-3:]:
                rnd = rec.get("round")
                rnd = "?" if rnd is None else str(rnd)
                dec = rec.get("decision", "?")
                desc = rec.get("description", "")[:40]
                sink.emit_status(f"[AR]   R{rnd}: {dec} — {desc}")

    if os.path.exists(plan_path(task_dir)):
        active = get_active_item(task_dir)
        if active:
            sink.emit_status(
                f"[AR] Active item: {active['id']}: "
                f"{active['description'][:50]}")
        sink.emit_status("[AR] Read .ar_state/plan.md and "
                         ".ar_state/history.jsonl for full context.")


def _handle_post_create_plan(task_dir: str, sink: _Sink):
    """Report the transaction result; create_plan already committed phase."""
    from phase_machine import validate_plan
    state = load_state(task_dir) or {}
    phase = state.get("phase")
    if phase != EDIT or state.get("pending_settle") is not None:
        sink.emit_status(
            f"[AR] Plan transaction did not commit; phase={phase}, "
            f"pending_settle={bool(state.get('pending_settle'))}. "
            f"Read create_plan.py stderr and retry the same phase.")
        return
    ok, err = validate_plan(task_dir)
    if not ok:
        sink.emit_status(f"[AR] Plan transaction is inconsistent: {err}")
        return
    sink.emit_status(f"[AR] Plan committed. Phase -> EDIT. "
                     f"{get_guidance(task_dir)}")
    sink.emit_todowrite(task_dir, "[AR] Plan committed. Phase -> EDIT.")


def _post_bash(event: AgentEvent, sink: _Sink):
    command = event.command

    target = _activation_target(command)
    if target:
        _handle_activation(target, sink)
        return

    task_dir = _task_dir_from_command(command)
    if not task_dir:
        return
    set_task_dir(task_dir, force=False)
    touch_heartbeat(task_dir)

    phase = read_phase(task_dir)
    invoked = parse_invoked_ar_script(command)

    if invoked == "baseline.py":
        progress = load_progress(task_dir)
        if not progress:
            sink.emit_status(
                "[AR] Baseline pending: no valid PyTorch reference "
                "(env/ref/worker). Fix the cause and re-run "
                "baseline.py.")
        else:
            outcome = getattr(progress, "baseline_outcome", None)
            sink.emit_status(_baseline_message(outcome, phase, progress,
                                               get_guidance(task_dir)))

    elif invoked == "pipeline.py":
        new_phase = read_phase(task_dir)
        sink.emit_status(
            f"[AR] Pipeline complete. Phase -> {new_phase}. "
            f"{get_guidance(task_dir)}")
        sink.emit_todowrite(
            task_dir, f"[AR] Round settled. Phase -> {new_phase}.")

    elif invoked == "create_plan.py":
        _handle_post_create_plan(task_dir, sink)


# ---------------------------------------------------------------------------
# post_tool / Edit-family  (was post_edit.main)
# ---------------------------------------------------------------------------
def _safe_load_config(task_dir: str):
    try:
        from task_config import load_task_config
        return load_task_config(task_dir)
    except Exception:
        return None


def _post_edit(event: AgentEvent, sink: _Sink):
    task_dir = get_task_dir()
    if not task_dir:
        return
    touch_heartbeat(task_dir)

    file_path = event.file_path
    if not file_path:
        return

    phase = read_phase(task_dir)

    config = _safe_load_config(task_dir)
    is_editable = False
    if config:
        try:
            rel = os.path.relpath(file_path, task_dir).replace("\\", "/")
            is_editable = rel in set(config.editable_files)
        except ValueError:
            is_editable = False

    if is_editable and phase == EDIT:
        sink.emit_status(
            f"[AR] Code edited. Continue editing OR run: "
            f"python scripts/engine/pipeline.py \"{task_dir}\""
        )


# ---------------------------------------------------------------------------
# post_tool / subagent  (was post_task.main; tool_kind == "subagent")
# ---------------------------------------------------------------------------
def _emit_retry_context(task_dir: str, plan_version: int, reason: str,
                        attempts: int, sink: _Sink):
    artifact = diagnose_artifact_path(task_dir, plan_version)
    marker = diagnose_marker(plan_version)
    msg = (
        f"[AR Phase: DIAGNOSE retry {attempts}/{DIAGNOSE_ATTEMPTS_CAP}] "
        f"Subagent did not produce a valid artifact: {reason}\n"
        f"\n"
        f"Required action: re-spawn the ar-diagnosis subagent. "
        f"In your prompt, restate that the subagent's FINAL action must be "
        f"a Write call to:\n"
        f"  {artifact}\n"
        f"and that the file body must contain headings 'Root cause', "
        f"'Fix directions', 'What to avoid', and end with this exact marker "
        f"line (plan-version-specific, do not paraphrase):\n"
        f"  {marker}\n"
        f"\n"
        f"Do NOT call create_plan.py, do NOT edit task source files, "
        f"do NOT Stop. "
        f"Only the ar-diagnosis subagent is legal in DIAGNOSE until the "
        f"artifact validates."
    )
    sink.emit_context(msg)


def _emit_manual_planning_context(task_dir: str, plan_version: int,
                                  reason: str, sink: _Sink):
    msg = (
        f"[AR Phase: DIAGNOSE — manual planning fallback] Subagent failed "
        f"{DIAGNOSE_ATTEMPTS_CAP}x for plan_v={plan_version} "
        f"(last reason: {reason}). Further subagent calls are blocked. Build "
        f"plan_items.xml from history.jsonl + plan.md (same as PLAN/REPLAN "
        f"flow), then run create_plan.py. Artifact gate is relaxed."
    )
    sink.emit_context(msg)


def _post_task(event: AgentEvent, sink: _Sink):
    task_dir = get_task_dir()
    if not task_dir:
        return
    touch_heartbeat(task_dir)
    if read_phase(task_dir) != DIAGNOSE:
        return

    state = diagnose_state(task_dir)
    pv = state.plan_version

    if state.action == DIAGNOSE_READY:
        update_progress(task_dir, diagnose_attempts=0,
                        diagnose_attempts_for_version=pv)
        sink.emit_status(
            f"[AR] DIAGNOSE artifact validated for plan_version={pv}. "
            f"Proceed to create_plan.py with diagnose_v{pv}.md as input."
        )
        return

    new_attempts = state.attempts + 1
    update_progress(task_dir, diagnose_attempts=new_attempts,
                    diagnose_attempts_for_version=pv,
                    last_diagnose_failure_reason=state.artifact_reason)
    if new_attempts >= DIAGNOSE_ATTEMPTS_CAP:
        sink.emit_status(
            f"[AR] DIAGNOSE subagent exhausted {DIAGNOSE_ATTEMPTS_CAP} "
            f"attempts for plan_version={pv}; switching to manual "
            f"planning fallback. Last reason: {state.artifact_reason}"
        )
        _emit_manual_planning_context(task_dir, pv, state.artifact_reason, sink)
    else:
        _emit_retry_context(task_dir, pv, state.artifact_reason,
                            new_attempts, sink)


# ---------------------------------------------------------------------------
# stop  (was stop_save.main)
# ---------------------------------------------------------------------------
def _on_stop(event: AgentEvent, sink: _Sink):
    from datetime import datetime, timezone

    stop_reason = event.stop_reason

    task_dir = get_task_dir()
    if not task_dir:
        return

    progress = load_progress(task_dir)
    phase = read_phase(task_dir)
    stuck = _is_stuck(phase, progress)

    if phase != FINISH and not stuck:
        # Stop's block uses a different exit convention than pre_tool; the
        # adapter maps that. Here it's just a block verdict.
        sink.block(
            f"[AR] Cannot Stop at phase={phase}. Continue the loop:\n\n"
            f"{get_guidance(task_dir)}"
        )

    if stuck:
        sink.emit_status(
            "\n[AR] Task aborted at BASELINE: no valid PyTorch reference "
            "(env / ref / worker)."
        )
        sink.emit_status(
            f"[AR] Fix the cause and `/autoresearch --resume {task_dir}` "
            f"to retry baseline, or re-scaffold from a fixed --ref."
        )
        return

    if progress is None:
        return

    update_progress(
        task_dir,
        last_stop_reason=stop_reason,
        last_stop_time=datetime.now(timezone.utc).isoformat(),
    )

    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 0)
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")

    improv = ""
    if best is not None and baseline is not None and baseline != 0:
        pct = (baseline - best) / abs(baseline) * 100
        improv = f" ({pct:+.1f}%)"

    sink.emit_status(f"\n[AR] Session stopped at FINISH: {stop_reason}")
    sink.emit_status(f"[AR] {rounds}/{max_rounds} rounds | Best: {best}{improv}")
    sink.emit_status(f"[AR] Resume: /autoresearch --resume {task_dir}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def _dispatch(event: AgentEvent, sink: _Sink):
    if event.kind == "stop":
        _on_stop(event, sink)
        return
    if event.kind == "pre_tool":
        if event.tool_kind == "shell":
            _pre_bash(event, sink)
        elif event.tool_kind == "edit":
            _pre_edit(event, sink)
        elif event.tool_kind == "subagent":
            _pre_task(event, sink)
        return
    if event.kind == "post_tool":
        if event.tool_kind == "shell":
            _post_bash(event, sink)
        elif event.tool_kind == "edit":
            _post_edit(event, sink)
        elif event.tool_kind == "subagent":
            _post_task(event, sink)
        return


def decide(event: AgentEvent) -> Decision:
    """Agent-neutral entry point. Returns a Decision; never writes to any
    wire and never calls sys.exit. May mutate .ar_state as a side effect."""
    sink = _Sink()
    try:
        _dispatch(event, sink)
    except _Block as b:
        return Decision(block=True, block_reason=b.reason,
                        status=sink.status, context=sink.context,
                        todos_header=sink.todos_header, todos=sink.todos)
    return Decision(block=False, status=sink.status, context=sink.context,
                    todos_header=sink.todos_header, todos=sink.todos)
