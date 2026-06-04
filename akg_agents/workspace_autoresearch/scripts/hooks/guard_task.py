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

"""PreToolUse hook for Task — DIAGNOSE-phase gate.

Enforces the DIAGNOSE invariant from `CLAUDE.md`:
  In phase DIAGNOSE, the only legal Task call has
  subagent_type='ar-diagnosis'.

Other phases: Task is left alone (host doesn't gate it). Wrong subagent_type
in DIAGNOSE → block with a clear retry reason.

This hook does NOT do artifact validation — that's PostToolUse's job
(hooks/post_task.py). PreToolUse's only role is "is this Task call legal at
all in the current phase".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.utils import read_hook_input, block_decision
from phase_machine import (
    DIAGNOSE, DIAGNOSE_ATTEMPTS_CAP, get_task_dir, read_phase,
    touch_heartbeat, diagnose_state,
    DIAGNOSE_READY, DIAGNOSE_MANUAL_FALLBACK,
)


_REQUIRED_SUBAGENT = "ar-diagnosis"


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Task":
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        # No active autoresearch task → don't gate. The user may be using
        # Task for unrelated workflows.
        sys.exit(0)
    touch_heartbeat(task_dir)

    if read_phase(task_dir) != DIAGNOSE:
        # Outside DIAGNOSE we don't restrict Task. Other phases use Bash /
        # Edit gates instead.
        sys.exit(0)

    # Hard cap: once subagent attempts hit the limit on this plan_version,
    # the manual-planning fallback is in effect (see hooks/post_task). Block
    # further Task calls so the agent doesn't burn context retrying a path
    # that has empirically failed; redirect them to write plan_items.xml
    # directly and run create_plan.py.
    state = diagnose_state(task_dir)
    if state.action == DIAGNOSE_READY:
        block_decision(
            f"[AR] DIAGNOSE artifact already validated for plan_version="
            f"{state.plan_version}. Do not re-issue Task; write "
            f".ar_state/plan_items.xml from the diagnosis and run "
            f"create_plan.py."
        )

    if state.action == DIAGNOSE_MANUAL_FALLBACK:
        block_decision(
            f"[AR] DIAGNOSE subagent already failed "
            f"{DIAGNOSE_ATTEMPTS_CAP} times for plan_version="
            f"{state.plan_version}. Switch to manual planning: read "
            f".ar_state/history.jsonl + plan.md, Write <items>...</items> "
            f"to .ar_state/plan_items.xml, then run create_plan.py. The "
            f"artifact gate on create_plan.py is relaxed in this state."
        )

    tool_input = hook_input.get("tool_input", {}) or {}
    subagent_type = tool_input.get("subagent_type")
    if subagent_type != _REQUIRED_SUBAGENT:
        block_decision(
            f"[AR] DIAGNOSE phase requires Task with "
            f"subagent_type='{_REQUIRED_SUBAGENT}', not "
            f"{subagent_type!r}. Re-issue Task with subagent_type set "
            f"exactly to '{_REQUIRED_SUBAGENT}'. The subagent definition "
            f"is at .claude/agents/{_REQUIRED_SUBAGENT}.md and is the "
            f"only diagnostician the host accepts here."
        )

    # subagent_type is correct → let it run; PostToolUse will validate the
    # artifact afterward.
    sys.exit(0)


if __name__ == "__main__":
    main()
