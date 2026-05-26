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
PreToolUse hook for Edit/Write — thin dispatcher.

Per-phase allow/block for file targets lives in phase_machine.check_edit.
This hook handles two concerns check_edit can't express as a pure function:

  1. Files outside the active task dir are rejected. The agent's job is to
     optimise the kernel inside <task_dir>; touching the source workspace/
     files, repo configs, or any other path outside the task is out of
     scope and was previously a silent footgun (e.g., the agent "fixing"
     a workspace/<op>_ref.py the user shared with git / CI).
  2. EDIT-phase dirty-tree gate — needs live git state, not just phase.
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring,wrong-import-position
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.utils import (read_hook_input, block_decision, block_with_guidance,
                         extract_target_path)
from phase_machine import (
    read_phase, get_task_dir, touch_heartbeat,
    edit_marker_path, check_edit, EDIT, DIAGNOSE,
    diagnose_state, pending_settle_path,
)


def _rel_to_task(file_path: str, task_dir: str):
    """Return task-relative forward-slash path, or None if outside task_dir.

    Uses os.path.commonpath to test containment instead of `startswith`,
    which would false-match siblings whose name is a prefix of the active
    task's name (e.g. file in `ar_tasks/sinkhorn` would be misidentified
    as inside the active task `ar_tasks/sinkhorn_v2`). With unique
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

    Note on the message wording: "uncommitted changes from previous round"
    used to be the only diagnosis here, but the same gate also fires when
    something off-flow edited an editable file. Keep the message neutral
    so the LLM doesn't latch onto "previous round" as the only possible
    cause.
    """
    try:
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=task_dir, capture_output=True, text=True, timeout=5, check=False,
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
                cwd=repo_root, capture_output=True, text=True, timeout=5, check=False,
            )
        except Exception:
            continue
        if diff.stdout.strip():
            block_decision(
                f"[AR] Uncommitted change in {ef!r} on entry to EDIT phase. "
                "Likely an unfinalized previous round, but could also be "
                "a seed commit that didn't land or an off-flow edit. "
                "Run pipeline.py to settle the current diff into a round "
                "before editing more: "
                f"python .autoresearch/scripts/engine/pipeline.py \"{task_dir}\""
            )

    # Start-of-round marker so re-editing the same file doesn't re-fire this gate
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("1")


_WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") not in _WRITE_TOOLS:
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    file_path = extract_target_path(hook_input)
    if not file_path:
        sys.exit(0)

    rel = _rel_to_task(file_path, task_dir)
    if rel is None:
        # Out-of-scope edit. Block instead of allowing — agent should only
        # touch files inside the active task_dir. Most common offender:
        # editing the source workspace/<op>_ref.py the user passed via
        # --ref. scaffold has already copied it into task_dir, so any
        # legitimate "fix the ref" decision belongs to a fresh /autoresearch
        # invocation by the user, not a side-effect of the current task.
        block_with_guidance(
            task_dir,
            f"Edit target {file_path!r} is outside the active task "
            f"directory ({task_dir}). The agent's scope is the task_dir "
            "only — source files in workspace/, repo configs, hooks, "
            "and anything else outside are off-limits. If you need to "
            "change a source --ref or --kernel file, exit the task and "
            "re-run /autoresearch with the corrected source."
        )

    from task_config import load_task_config
    config = load_task_config(task_dir)
    editable_files = list(config.editable_files) if config else []

    phase = read_phase(task_dir)
    # plan_items.xml in DIAGNOSE is gated on the three-state action; in
    # EDIT it's gated on pending-settle recovery. Compute both so
    # check_edit can apply the rules.
    diag_action = (diagnose_state(task_dir).action
                   if phase == DIAGNOSE else None)
    pending = (phase == EDIT
               and os.path.isfile(pending_settle_path(task_dir)))
    ok, reason = check_edit(phase, rel, editable_files,
                            diagnose_action=diag_action,
                            pending_settle=pending)
    if not ok:
        block_with_guidance(task_dir, reason)

    # Phase says OK. For EDIT writes to editable files, also check the git
    # state gate (dirty tree on entry to a new round).
    if phase == EDIT and rel in set(editable_files):
        _edit_phase_git_gate(task_dir, editable_files)

    sys.exit(0)


if __name__ == "__main__":
    main()
