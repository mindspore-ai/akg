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

"""PostToolUse hook for Task — DIAGNOSE artifact validator.

Runs after every Task call. Only acts when phase==DIAGNOSE; otherwise no-op.

Behavior (driven by `state.action` from `diagnose_state`):
  - DIAGNOSE_READY (artifact valid) → reset diagnose_attempts; emit
    success status; main agent then runs create_plan.py.
  - else → increment diagnose_attempts (per plan_version). The new value
    determines next state:
      * still NEED_DIAGNOSIS → emit retry context (re-issue Task).
      * crossed cap → emit manual-planning context. From here on,
        hooks/guard_task blocks further Task calls and hooks/guard_bash's
        DIAGNOSE gate (which checks `state.action == DIAGNOSE_NEED_DIAGNOSIS`)
        no longer blocks create_plan.py, so the main agent can write
        plan_items.xml itself and proceed.

The DIAGNOSE phase always exits via create_plan.py advancing phase to EDIT
— manual-planning is a fallback path to that same exit, not a termination.
"""

# pylint: disable=missing-function-docstring,wrong-import-position
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.utils import read_hook_input, emit_status
from phase_machine import (
    DIAGNOSE, DIAGNOSE_ATTEMPTS_CAP, diagnose_artifact_path,
    diagnose_marker, diagnose_state, get_task_dir, read_phase,
    update_progress,
    DIAGNOSE_READY,
)


def _emit_retry_context(task_dir: str, plan_version: int, reason: str,
                        attempts: int) -> None:
    """Tell the main agent what went wrong and to retry Task — never to
    backstop the diagnose work itself. We surface attempts/cap so the model
    knows it has finite tries."""
    artifact = diagnose_artifact_path(task_dir, plan_version)
    marker = diagnose_marker(plan_version)
    msg = (
        f"[AR Phase: DIAGNOSE retry {attempts}/{DIAGNOSE_ATTEMPTS_CAP}] "
        f"Subagent did not produce a valid artifact: {reason}\n"
        f"\n"
        "Required action: re-issue Task with subagent_type='ar-diagnosis'. "
        "In your prompt, restate that the subagent's FINAL action must be "
        f"a Write call to:\n"
        f"  {artifact}\n"
        "and that the file body must contain headings 'Root cause', "
        "'Fix directions', 'What to avoid', and end with this exact marker "
        f"line (plan-version-specific, do not paraphrase):\n"
        f"  {marker}\n"
        f"\n"
        "Do NOT call create_plan.py, do NOT Edit kernel.py, do NOT Stop. "
        "Only Task is legal in DIAGNOSE until the artifact validates."
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": msg,
        }
    }))


def _emit_manual_planning_context(task_dir: str, plan_version: int,
                                  reason: str) -> None:
    """At cap, tell the agent to switch to manual planning.

    Subagent route is exhausted; DIAGNOSE still needs a new plan. The
    artifact gate is dropped (state.action == DIAGNOSE_MANUAL_FALLBACK
    means hooks/guard_bash no longer blocks create_plan.py and
    hooks/guard_task blocks further Task calls). Agent proceeds via the
    normal PLAN/REPLAN flow: write plan_items.xml from history.jsonl +
    plan.md, then run create_plan.py.
    """
    msg = (
        "[AR Phase: DIAGNOSE — manual planning fallback] Subagent failed "
        f"{DIAGNOSE_ATTEMPTS_CAP}x for plan_v={plan_version} "
        f"(last reason: {reason}). Further Task calls are blocked. Build "
        "plan_items.xml from history.jsonl + plan.md (same as PLAN/REPLAN "
        "flow), then run create_plan.py. Artifact gate is relaxed."
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": msg,
        }
    }))


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Task":
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    if read_phase(task_dir) != DIAGNOSE:
        sys.exit(0)

    state = diagnose_state(task_dir)
    pv = state.plan_version

    if state.action == DIAGNOSE_READY:
        # Reset attempts so a future DIAGNOSE round (different
        # plan_version) starts fresh.
        update_progress(task_dir, diagnose_attempts=0,
                        diagnose_attempts_for_version=pv)
        emit_status(
            f"[AR] DIAGNOSE artifact validated for plan_version={pv}. "
            f"Proceed to create_plan.py with diagnose_v{pv}.md as input."
        )
        sys.exit(0)

    # Failure path: persist incremented counter; branch on whether the
    # new attempt count crossed the cap. (Earlier code also computed a
    # `next_action` enum here, but it was just a rename of the same
    # comparison — drop the indirection.)
    new_attempts = state.attempts + 1
    update_progress(task_dir, diagnose_attempts=new_attempts,
                    diagnose_attempts_for_version=pv,
                    last_diagnose_failure_reason=state.artifact_reason)
    if new_attempts >= DIAGNOSE_ATTEMPTS_CAP:
        emit_status(
            f"[AR] DIAGNOSE subagent exhausted {DIAGNOSE_ATTEMPTS_CAP} "
            f"attempts for plan_version={pv}; switching to manual "
            f"planning fallback. Last reason: {state.artifact_reason}"
        )
        _emit_manual_planning_context(task_dir, pv, state.artifact_reason)
    else:
        _emit_retry_context(task_dir, pv, state.artifact_reason, new_attempts)
    sys.exit(0)


if __name__ == "__main__":
    main()
