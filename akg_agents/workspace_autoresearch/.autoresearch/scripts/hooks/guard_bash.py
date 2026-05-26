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
PreToolUse hook for Bash — thin dispatcher.

Per-phase allow/block logic lives in phase_machine.check_bash. This hook
only adds the concerns that need state outside check_bash's pure-function
contract:
  1. Script-name sanity (blessed names / hallucinated-name suggestions),
     run BEFORE check_bash so the agent sees "Unknown script 'eval.py'"
     instead of the generic canonical-form rejection.
  2. DIAGNOSE artifact gate — create_plan.py is blocked until the
     subagent's diagnose_v<N>.md validates (or the attempts cap relaxes
     the gate).
  3. EDIT-recovery gate — create_plan.py is allowed in EDIT iff
     `.pending_settle.json` exists, as the recovery path out of a
     settle.py deadlock.
  4. Turning check_bash's (False, reason) into the `{decision: block}`
     wire format Claude Code expects.
"""

# pylint: disable=missing-function-docstring,wrong-import-position
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.utils import read_hook_input, block_decision, block_with_guidance
from phase_machine import (
    DIAGNOSE, DIAGNOSE_ATTEMPTS_CAP, EDIT, REPLAN, read_phase,
    get_task_dir, touch_heartbeat, check_bash, parse_script_names,
    diagnose_state, parse_invoked_ar_script, pending_settle_path,
    is_single_foreground_ar_invocation,
    DIAGNOSE_NEED_DIAGNOSIS,
)
from utils.settings import hallucinated_scripts

# Real CLI scripts under .autoresearch/scripts/. Anything not listed
# here (and not in _LIBRARY_NOT_CLI / hallucinated_scripts aliases) is
# treated as an unknown script and rejected with a sorted list of
# valid names.
_BLESSED_SCRIPTS = {
    "quick_check.py", "eval_wrapper.py",
    "scaffold.py", "baseline.py", "dashboard.py",
    "create_plan.py", "settle.py", "pipeline.py", "resume.py",
    "parse_args.py",
}

# Library modules under .autoresearch/scripts/ that are not CLI-invocable.
# Each entry maps the module name to a pointed message so the LLM gets a
# specific nudge instead of the generic "unknown script" rejection.
_LIBRARY_NOT_CLI = {
    "phase_machine.py": (
        "phase_machine.py is a library used by hooks, not a CLI. "
        "Guidance ([AR Phase: ...]) is auto-emitted on stderr after every "
        "legal Bash/Edit. If you haven't seen a fresh guidance message, "
        "wait for the next hook output — do not try to fetch it manually."
    ),
    "task_config.py": "task_config.py is a library, not a CLI.",
    "settings.py": "settings.py is a library, not a CLI.",
    "hw_detect.py": "hw_detect.py is a library, not a CLI.",
    "utils.py": "hooks/utils.py is a library, not a CLI.",
    "failure_extractor.py": "failure_extractor.py is a library, not a CLI.",
    "code_checker.py": (
        "code_checker.py is a library used by quick_check.py, not a CLI. "
        "It runs automatically as part of the pipeline."
    ),
}

# Alias → real script mapping lives in .autoresearch/config.yaml under
# `hallucinated_scripts`; loaded lazily so the config can be hot-edited.


def _script_name_check(command: str):
    """Flag unknown / hallucinated .autoresearch/scripts/*.py names before
    they reach the phase rule — gives a clearer message than 'not allowed'.

    Under the canonical-form policy, parse_script_names returns at most
    one entry (chains are rejected by check_bash before this could
    matter). Non-canonical commands return [] and fall through to
    check_bash's canonical-form rejection."""
    aliases = hallucinated_scripts()
    for script_path, script_name in parse_script_names(command):
        if script_name in aliases:
            real = aliases[script_name]
            block_decision(f"[AR] '{script_name}' does not exist. "
                   f"Use: python .autoresearch/scripts/{real}")
        if script_name in _LIBRARY_NOT_CLI:
            block_decision(f"[AR] {_LIBRARY_NOT_CLI[script_name]}")
        if ".autoresearch/scripts/" in script_path and script_name not in _BLESSED_SCRIPTS:
            block_decision(f"[AR] Unknown script '{script_name}'. "
                   f"Valid scripts: {sorted(_BLESSED_SCRIPTS)}")


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Bash":
        sys.exit(0)

    task_dir = get_task_dir()
    if not task_dir:
        sys.exit(0)
    touch_heartbeat(task_dir)

    command = (hook_input.get("tool_input") or {}).get("command", "")
    _script_name_check(command)

    phase = read_phase(task_dir)
    invoked = parse_invoked_ar_script(command)

    # DIAGNOSE-specific Bash gate: create_plan.py must come AFTER the
    # subagent artifact validates — UNLESS the subagent attempts cap has
    # been reached, in which case the manual-planning fallback applies and
    # the artifact gate is dropped.
    if phase == DIAGNOSE and invoked == "create_plan.py":
        state = diagnose_state(task_dir)
        if state.action == DIAGNOSE_NEED_DIAGNOSIS:
            block_decision(
                "[AR] create_plan.py blocked in DIAGNOSE: artifact "
                f"check failed ({state.artifact_reason}). Issue Task "
                "with subagent_type='ar-diagnosis' first; only after "
                "the artifact validates may you run create_plan.py. "
                f"(Subagent attempts so far: {state.attempts}/"
                f"{DIAGNOSE_ATTEMPTS_CAP}; at the cap the gate is "
                "relaxed and you may write the plan directly.)"
            )

    # EDIT-phase recovery gate: when settle.py keeps failing on a malformed
    # plan.md, the agent has no legal action under normal EDIT rules
    # (kernel.py edits don't help; create_plan.py isn't in EDIT's
    # allowlist). If `.pending_settle.json` exists, allow create_plan.py
    # as a recovery path; hooks/post_bash clears the sidecar on successful
    # create_plan validation.
    #
    # is_single_foreground_ar_invocation reuses the canonical-form regex
    # from phase_policy, so "single foreground call" stays defined in one
    # place; FD redirects (`2>&1`, `&> log`) are part of the canonical
    # grammar and pass naturally. The follow-up check_bash(REPLAN) is
    # defense-in-depth for the global subprocess-script bans.
    if phase == EDIT and invoked == "create_plan.py" \
            and os.path.exists(pending_settle_path(task_dir)):
        ok, reason = is_single_foreground_ar_invocation(
            command, script="create_plan.py")
        if not ok:
            block_decision(
                "[AR] Recovery path requires a single foreground "
                "create_plan.py invocation while .pending_settle.json "
                f"exists: {reason}. Re-issue without chaining; FD "
                "redirects (`2>&1`, `> log.txt`) are fine."
            )
        ok, reason = check_bash(REPLAN, command)
        if not ok:
            block_with_guidance(task_dir, reason)
        sys.exit(0)

    ok, reason = check_bash(phase, command)
    if not ok:
        block_with_guidance(task_dir, reason)
    sys.exit(0)


if __name__ == "__main__":
    main()
