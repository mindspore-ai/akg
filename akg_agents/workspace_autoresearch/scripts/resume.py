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

"""Resume an existing autoresearch task.

Usage:
    python scripts/resume.py [task_dir] [--force]

If task_dir is omitted, auto-detects the most recently active task
(prefers the current session's task via the session index).

Opens a Task with role="agent" — heal + consistency check + claim
ownership happen in one place. Refusal to claim a fresh-but-foreign
task is the TaskOwnershipError path; --force takes over via Task's
force flag. The journal handles the partial-baseline crash window
(SEED row landed, state didn't commit progress_initialized=True)
transparently: replay_intent inside open_task rebuilds state before
this script reads anything.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_machine import (
    plan_path, edit_marker_path, has_pending_items,
    find_active_task_dir,
)
from task_handle import (
    open_task, Role,
    TaskOwnershipError, TaskConsistencyError, TaskNotInitialized,
)


def _validate_resumable(t) -> tuple[bool, str]:
    """Post-open validation. open_task already did heal + consistency +
    claim; here we only check resume-specific shape (progress
    initialised, plan.md still valid if non-empty)."""
    try:
        progress = t.progress
    except TaskNotInitialized:
        # Resumability is keyed on PHASE, not progress presence. A task
        # parked at BASELINE with no committed baseline is the legitimate
        # "baseline pending" state (the gate refused to commit because no
        # valid ref baseline) — resume re-runs baseline.py after the env/
        # ref/worker is fixed. Any other phase with no progress is a task
        # that was never initialised.
        from phase_machine import read_phase, BASELINE
        if read_phase(t.task_dir) == BASELINE:
            return True, ""
        return False, ("Baseline never committed and phase is not "
                       "BASELINE — task was never initialised. Run "
                       "/autoresearch without --resume to start fresh.")
    required_fields = {"task", "eval_rounds", "max_rounds"}
    missing = required_fields - set(progress.keys())
    if missing:
        return False, f"state.json progress fields missing: {missing}"
    # plan.md present + has pending items → must be structurally valid.
    # A fully-consumed plan (0 pending) is legal (compute_resume_phase
    # routes it to REPLAN). validate_plan rejects 0-pending plans for
    # lack of an ACTIVE item, so only validate when items exist.
    if os.path.exists(plan_path(t.task_dir)) and has_pending_items(t.task_dir):
        from phase_machine import validate_plan
        ok, err = validate_plan(t.task_dir)
        if not ok:
            return False, f"plan.md invalid: {err}"
    return True, ""


def main():
    args = sys.argv[1:]
    force = "--force" in args
    args = [a for a in args if a != "--force"]
    task_dir = args[0] if args else None

    if task_dir:
        task_dir = os.path.abspath(task_dir)
    else:
        # find_active_task_dir prefers current_session_task_dir
        # (index-first), so external --output-dir tasks are visible.
        task_dir = find_active_task_dir() or ""
        if not task_dir:
            print("[resume] ERROR: No existing task found in ar_tasks/",
                  file=sys.stderr)
            sys.exit(1)

    if not os.path.isdir(task_dir):
        print(f"[resume] ERROR: Not a directory: {task_dir}",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(os.path.join(task_dir, "task.yaml")):
        print(f"[resume] ERROR: Missing task.yaml in {task_dir}",
              file=sys.stderr)
        sys.exit(1)

    try:
        with open_task(task_dir, role=Role.AGENT, force=force) as t:
            ok, err = _validate_resumable(t)
            if not ok:
                print(f"[resume] ERROR: Cannot resume {task_dir}",
                      file=sys.stderr)
                print(f"[resume] {err}", file=sys.stderr)
                sys.exit(1)

            # Clean stale edit marker (git clean means marker is stale).
            marker = edit_marker_path(task_dir)
            if os.path.exists(marker):
                from utils.git_utils import is_working_tree_clean
                if is_working_tree_clean(task_dir):
                    try:
                        os.remove(marker)
                        print("[resume] Cleaned stale edit marker.",
                              file=sys.stderr)
                    except OSError:
                        pass

            # Status line from the summary facade. Bundled view so
            # adding/renaming fields touches one place.
            summary = t.summary or {}
            print(f"[resume] Task: {summary.get('task')}")
            print(f"[resume] Round: {summary.get('eval_rounds')}/"
                  f"{summary.get('max_rounds')}")
            print(f"[resume] Best: {summary.get('best_metric')} | "
                  f"Baseline: {summary.get('baseline_metric')}")
            print(f"[resume] Phase: {summary.get('phase')}")
            # Print task_dir on last line (for easy parsing by callers)
            print(task_dir)
    except TaskConsistencyError as e:
        print(f"[resume] ERROR: state inconsistent for {task_dir}",
              file=sys.stderr)
        print(f"[resume] {e}", file=sys.stderr)
        sys.exit(1)
    except TaskOwnershipError as e:
        print(f"[resume] ERROR: {e}", file=sys.stderr)
        print(f"[resume] Another Claude Code session may be running it.",
              file=sys.stderr)
        print(f"[resume] If you're sure no other session is running, "
              f"add --force:", file=sys.stderr)
        print(f"[resume]   /autoresearch --resume --force",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
