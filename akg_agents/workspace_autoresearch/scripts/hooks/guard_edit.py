#!/usr/bin/env python3
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

"""
PreToolUse hook for Edit/Write — path-driven gate.

Two clean branches, no fallback chains:

  Branch 1: edit target lives inside an AR task_dir (has a
  ``.ar_state/state.json`` ancestor) → that task's ``phase`` +
  ``editable_files`` rules apply via ``phase_machine.check_edit``. Session
  ownership is irrelevant here — file path is a hard fact, state.owner is
  a soft hint that can drift out of sync with the live process.

  Branch 2: edit target is outside every AR task_dir. If the current
  session owns a task in flight, this is an off-flow edit ("the agent's
  scope is its task_dir") → block. If no session owns anything (pre-
  activation, or operator setup), allow.

Note: cross-task edits (session owns T1, edit lands inside T2) are
handled by Branch 1's per-task rules, NOT by the off-flow check —
T2's editable_files is the right gate for that case.

The EDIT-phase dirty-tree git gate runs alongside Branch 1 when the
edit lands on an editable file.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.utils import (read_hook_input, block_decision, block_with_guidance,
                         extract_target_path)
from phase_machine import (
    read_phase, get_task_dir, touch_heartbeat,
    edit_marker_path, check_edit, EDIT, DIAGNOSE,
    diagnose_state, load_state,
)


def _has_pending_settle(task_dir: str) -> bool:
    state = load_state(task_dir)
    return bool(state and state.get("pending_settle"))


def _rel_to_task(file_path: str, task_dir: str):
    """Return task-relative forward-slash path, or None if outside task_dir.

    Uses os.path.commonpath to test containment instead of `startswith`,
    which would false-match siblings whose name is a prefix of the active
    task's name (e.g. file in `ar_tasks/<op>` would be misidentified
    as inside the active task `ar_tasks/<op>_v2`). With unique
    timestamp+hash suffixes from scaffold this collision is rare in
    practice but the wrong primitive — the right test is genuine path
    ancestry, not string prefix.

    Containment check uses native-separator paths (commonpath returns
    os.sep-form regardless of input), THEN converts the relative result
    to forward-slash for downstream check_edit which compares against
    forward-slash patterns.
    """
    fp_native = os.path.normpath(os.path.abspath(file_path))
    td_native = os.path.normpath(os.path.abspath(task_dir))
    try:
        if os.path.commonpath([fp_native, td_native]) != td_native:
            return None
    except ValueError:
        # commonpath raises on cross-drive paths (Windows) and on
        # absolute-vs-relative mixes — both mean fp is outside task_dir.
        return None
    return os.path.relpath(file_path, task_dir).replace("\\", "/")


def _edit_phase_git_gate(task_dir: str, editable_files):
    """In EDIT phase, if any editable file has uncommitted changes AND the
    edit-started marker is absent, the tree is dirty without an in-progress
    round to attribute it to. Force Claude to run pipeline.py to finalize.

    Sets the marker on success so subsequent edits within the same round
    pass through.

    Note on the message wording: this gate fires for both uncommitted
    changes from a previous round AND for off-flow edits to editable
    files. Keep the message neutral so the LLM doesn't latch onto
    "previous round" as the only possible cause.
    """
    try:
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=task_dir, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=5,
        ).stdout.strip()
    except Exception:
        return  # git unavailable — skip gate rather than block

    marker = edit_marker_path(task_dir)
    if os.path.exists(marker):
        return  # already in an active round

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
            block_decision(
                f"[AR] Uncommitted change in {ef!r} on entry to EDIT phase. "
                f"Likely an unfinalized previous round, but could also be "
                f"a seed commit that didn't land or an off-flow edit. "
                f"Run pipeline.py to settle the current diff into a round "
                f"before editing more: "
                f"python scripts/engine/pipeline.py \"{task_dir}\""
            )

    # Start-of-round marker so re-editing the same file doesn't re-fire this gate
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w") as f:
        f.write("1")


_WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def _task_dir_from_edit_target(file_path: str):
    """Walk up from ``file_path`` looking for a ``.ar_state/state.json``
    sibling. The first ancestor that has one is the task this edit
    belongs to. Returns None if no ancestor matches (edit is outside
    every AR task)."""
    p = os.path.abspath(file_path)
    while True:
        parent = os.path.dirname(p)
        if parent == p:
            return None
        if os.path.isfile(os.path.join(parent, ".ar_state", "state.json")):
            return parent
        p = parent


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") not in _WRITE_TOOLS:
        sys.exit(0)

    file_path = extract_target_path(hook_input)
    if not file_path:
        sys.exit(0)

    target_task = _task_dir_from_edit_target(file_path)

    # Branch 1: edit lands inside an AR task → that task's editability
    # rules apply, regardless of who owns it.
    if target_task:
        touch_heartbeat(target_task)
        # rel is guaranteed non-None: _task_dir_from_edit_target returned
        # target_task BECAUSE file_path is inside it.
        rel = _rel_to_task(file_path, target_task)

        from task_config import load_task_config
        config = load_task_config(target_task)
        editable_files = list(config.editable_files) if config else []

        phase = read_phase(target_task)
        # plan_items.xml in DIAGNOSE is gated on the three-state action;
        # in EDIT it's gated on pending-settle recovery. Compute both so
        # check_edit can apply the rules.
        diag_action = (diagnose_state(target_task).action
                       if phase == DIAGNOSE else None)
        pending = (phase == EDIT and _has_pending_settle(target_task))
        ok, reason = check_edit(phase, rel, editable_files,
                                diagnose_action=diag_action,
                                pending_settle=pending)
        if not ok:
            block_with_guidance(target_task, reason)

        # Phase says OK. For EDIT writes to editable files, also check
        # the git state gate (dirty tree on entry to a new round).
        if phase == EDIT and rel in set(editable_files):
            _edit_phase_git_gate(target_task, editable_files)

        sys.exit(0)

    # Branch 2: edit is outside every AR task. Only meaningful when a
    # session has an active claim — then this is an off-flow edit (the
    # agent should stay within its task_dir). With no active claim
    # there's nothing to enforce: pre-activation scaffolding or
    # operator-side repo work is legitimate.
    owned = get_task_dir()
    if owned:
        block_with_guidance(
            owned,
            f"Edit target {file_path!r} is outside the active task "
            f"directory ({owned}). The agent's scope is the task_dir "
            f"only — source files in workspace/, repo configs, hooks, "
            f"and anything else outside are off-limits. If you need to "
            f"change a source --ref or --kernel file, exit the task and "
            f"re-run /autoresearch with the corrected source."
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
