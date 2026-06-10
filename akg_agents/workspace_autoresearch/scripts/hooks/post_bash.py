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
PostToolUse hook for Bash — phase auto-advancement after user-issued commands.

The only commands that advance phase from this hook are those Claude runs
directly via the Bash tool:
  - `export AR_TASK_DIR=...`  → activate task, compute starting phase
                                (fresh task always lands at BASELINE; both
                                reference.py and kernel.py are guaranteed
                                present because /autoresearch requires both)
  - `baseline.py`             → PLAN on success or on kernel_fail;
                                infra_fail (no valid ref) parks at
                                BASELINE with no committed progress
  - `pipeline.py`             → whatever phase pipeline.py itself wrote
  - `create_plan.py`          → EDIT on plan validation pass
                                (called from PLAN / DIAGNOSE / REPLAN)

The inner pipeline steps (quick_check subprocess + in-process run_eval +
in-process record_round + settle subprocess) run beneath pipeline.py and
never re-enter this hook, so they don't need their own phase constants
or branches here.
"""
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hooks.utils import read_hook_input, emit_status, emit_todowrite_context
from workflow import PhaseController
from phase_machine import (
    read_phase, get_guidance, compute_resume_phase,
    set_task_dir, get_active_item, touch_heartbeat,
    load_progress, update_progress,
    parse_invoked_ar_script,
    history_path, plan_path, edit_marker_path, state_record_path,
    BASELINE, PLAN, EDIT, DIAGNOSE, REPLAN,
)


def _activation_target(command: str) -> str | None:
    r"""Extract the path from `export AR_TASK_DIR=<path>`. Uses shlex
    so quoted values with spaces (`AR_TASK_DIR="/path with space"`)
    survive — the earlier `[^"\';\s&]+` regex truncated at the first
    space."""
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


def _task_dir_from_command(command: str) -> str | None:
    """Find a Bash token that names an existing task directory (has
    ``.ar_state/state.json``). The Bash command tells us which task it
    operates on; this is a hard fact, independent of session ownership."""
    try:
        tokens = shlex.split(command, posix=True, comments=False)
    except ValueError:
        return None
    for tok in tokens:
        if tok and os.path.isfile(os.path.join(tok, ".ar_state", "state.json")):
            return os.path.abspath(tok)
    return None


# Script-invocation parsing lives in phase_machine.parse_invoked_ar_script,
# a thin view over `classify(command)` — returns the AR script basename
# only when the classifier sees a canonical AR shape (and None otherwise,
# including for non-canonical AR-mentions which PreToolUse already
# rejected). Under that contract the basename returned here is
# unambiguous, and shapes like `python --version ...X.py` or
# `python -c ... .../X.py` no longer falsely advance phase.


def _clean_stale_edit_marker(task_dir: str):
    """Remove .edit_started if git is clean (nothing to resume)."""
    marker = edit_marker_path(task_dir)
    if not os.path.exists(marker):
        return
    from utils.git_utils import is_working_tree_clean
    if is_working_tree_clean(task_dir):
        try:
            os.remove(marker)
            emit_status("[AR] Cleaned stale edit marker (git is clean).")
        except OSError:
            pass


def _handle_activation(new_task_dir: str):
    new_task_dir = os.path.abspath(new_task_dir)
    if not os.path.isdir(new_task_dir):
        emit_status(f"[AR] ERROR: task_dir not found: {new_task_dir}")
        return

    # state.json existence pre-check: it's the fresh-vs-resume
    # discriminator below. open_task heals + checks; we need to know
    # the pre-replay shape to decide which activation event to fire
    # (on_activation_ready for fresh, on_activation_resume for resume).
    has_state = os.path.exists(state_record_path(new_task_dir))

    from task_handle import (
        open_task as _open_task, Role as _Role,
        TaskOwnershipError as _Ownership,
        TaskConsistencyError as _Consistency,
    )
    try:
        with _open_task(new_task_dir, role=_Role.AGENT) as t:
            _clean_stale_edit_marker(new_task_dir)
            if not has_state:
                # Fresh-scaffolded task. Initial phase transition →
                # BASELINE.
                t.advance_on_activation_fresh()
                emit_status(f"[AR] Fresh start. Phase -> BASELINE. "
                            f"{get_guidance(new_task_dir)}")
                return

            phase = t.phase
            # Stale-planning recovery: phase says PLAN or REPLAN but
            # plan.md + progress show a validated plan with an active
            # item — state left by a create_plan.py that finished
            # disk writes but crashed before PostToolUse advanced to
            # EDIT. Without recovery, the agent re-runs create_plan,
            # bumps to vN+1, and loses the pending items of vN.
            #
            # DIAGNOSE deliberately NOT in this list: it has its own
            # gate (diagnose_state.action requires the subagent's
            # diagnose_v<N>.md artifact). compute_resume_phase
            # doesn't model that gate — leave DIAGNOSE to the normal
            # PostToolUse(Task) flow.
            if phase in (PLAN, REPLAN):
                recomputed = t.advance_on_activation_resume()
                if recomputed != phase:
                    emit_status(
                        f"[AR] Phase file was {phase} but plan.md + "
                        f"state show round-ready — advancing to "
                        f"{recomputed} (create_plan.py likely "
                        f"crashed before PostToolUse could advance).")
                    phase = recomputed
            emit_status(f"[AR] Resuming. Phase: {phase}.")
            _print_resume_context(new_task_dir)
            emit_status(get_guidance(new_task_dir))
    except _Consistency as e:
        # Replay couldn't heal — off-flow corruption. Surface the
        # recovery message verbatim.
        emit_status(f"[AR] {e}")
    except _Ownership as e:
        # Refused to claim: another live session owns the task.
        emit_status(
            f"[AR] ERROR: refused to activate {new_task_dir} — {e} "
            f"Stop the other session before retrying."
        )


def _fresh_start(task_dir: str):
    """Pick initial phase for a fresh task. With /autoresearch requiring
    both --ref and --kernel, scaffold has already gated on reference
    runnability and written the user's seed kernel; the next legal step
    is always BASELINE. baseline.py exercises the kernel; on failure the
    hook routes to PLAN so the agent rewrites via plan->edit."""
    PhaseController(task_dir).on_activation_ready()
    emit_status(f"[AR] Fresh start. Phase -> BASELINE. {get_guidance(task_dir)}")


def _baseline_message(outcome, new_phase, progress, guidance):
    # infra_fail never reaches this function: the ref-baseline gate
    # refuses to commit progress in that case, so the caller hits the
    # "no progress" branch and emits the baseline-pending message
    # instead. Only OK / KERNEL_FAIL outcomes are reachable here.
    if outcome != "ok":
        reason = ("seed kernel produced no timing"
                  if progress.seed_metric is None
                  else "seed kernel failed correctness / profile")
        return (f"[AR] Baseline failed: {reason}. Phase -> PLAN. Plan a "
                f"kernel fix/rewrite via the standard plan->edit loop. "
                f"{guidance}")
    return f"[AR] Baseline complete. Phase -> PLAN. {guidance}"


def _reset_failures_for_diagnose(task_dir: str, phase: str):
    """Zero consecutive_failures only on DIAGNOSE replan validation
    (PLAN/REPLAN keep the streak — failures led to the replan)."""
    if phase == DIAGNOSE:
        update_progress(task_dir, consecutive_failures=0)


def main():
    hook_input = read_hook_input()
    if hook_input.get("tool_name", "") != "Bash":
        sys.exit(0)

    command = hook_input.get("tool_input", {}).get("command", "")
    stdout = str(hook_input.get("tool_output", ""))

    # --- Activation (export AR_TASK_DIR=...) ---
    # Activation arrives as its own Bash call under the canonical-form
    # gate (any chain is rejected at PreToolUse), so we can return as
    # soon as `_handle_activation` has set up the task pointer + emitted
    # guidance — there is no AR-script invocation in the same command
    # to dispatch on.
    target = _activation_target(command)
    if target:
        _handle_activation(target)
        sys.exit(0)

    # Path-driven routing: the Bash command tells us which task it
    # operated on. Session ownership is only used as bookkeeping below
    # (set_task_dir adopts the task if the previous owner went silent;
    # rule 5 refuses cleanly when a live owner exists).
    task_dir = _task_dir_from_command(command)
    if not task_dir:
        sys.exit(0)
    set_task_dir(task_dir, force=False)
    touch_heartbeat(task_dir)

    phase = read_phase(task_dir)
    invoked = parse_invoked_ar_script(command)

    if invoked == "baseline.py" and phase == BASELINE:
        progress = load_progress(task_dir)
        if not progress:
            # No committed progress at BASELINE = ref-baseline gate
            # refused to commit (no valid PyTorch reference). The
            # specific cause (env/ref/worker) isn't recorded; stop_save
            # surfaces the recovery path on Stop.
            emit_status("[AR] Baseline pending: no valid PyTorch reference "
                        "(env/ref/worker). Fix the cause and re-run "
                        "baseline.py.")
        else:
            # baseline.py / workflow.run_baseline_init already advanced the
            # phase via PhaseController.on_baseline_settled before exiting.
            # Read it back here — do NOT re-run the transition.
            new_phase = read_phase(task_dir)
            outcome = getattr(progress, "baseline_outcome", None)
            emit_status(_baseline_message(outcome, new_phase, progress,
                                          get_guidance(task_dir)))

    elif invoked == "pipeline.py":
        # pipeline.py updates state.json phase itself; just project state + notify.
        new_phase = read_phase(task_dir)
        emit_status(f"[AR] Pipeline complete. Phase -> {new_phase}. {get_guidance(task_dir)}")
        emit_todowrite_context(task_dir, f"[AR] Round settled. Phase -> {new_phase}.")

    elif invoked == "create_plan.py" and phase in (PLAN, DIAGNOSE, REPLAN, EDIT):
        _handle_post_create_plan(task_dir, phase)

    sys.exit(0)


def _handle_post_create_plan(task_dir: str, phase: str) -> None:
    """create_plan completion handler. Extracted into its own function
    so the "nothing to do" early-out paths can use `return` instead of
    sys.exit(0) — sys.exit raises SystemExit which Task.__exit__ counts
    as a failure and would release the claim, leaving subsequent
    hooks with no active task.

    PLAN/DIAGNOSE/REPLAN: normal plan-creation flow.
    EDIT: only legal as a recovery path when settle kept failing on a
    malformed plan.md (gated in hooks/guard_bash by state.pending_
    settle being non-null). The new plan retires the broken
    plan_version, so the orphan kd_json is no longer actionable;
    clear it as part of the post-create_plan transition.
    """
    from phase_machine import validate_plan
    from task_handle import (
        open_task as _open_task, Role as _Role,
        TaskConsistencyError as _Consistency,
        TaskOwnershipError as _Ownership,
    )
    try:
        with _open_task(task_dir, role=_Role.AGENT) as t:
            pending = t.pending_settle
            if phase == EDIT and not pending:
                # Defense-in-depth: guard_bash should have blocked
                # this, but if it slipped through, refuse cleanly
                # (no claim release).
                emit_status("[AR] create_plan.py in EDIT phase "
                            "requires a pending settle recovery "
                            "state; nothing to do.")
                return
            ok, err = validate_plan(task_dir)
            if not ok:
                emit_status(f"[AR] Plan not valid yet: {err}")
                return
            _reset_failures_for_diagnose(task_dir, phase)
            t.advance_on_plan_validated()
            if phase == EDIT:
                # Recovery completed: discard the orphan kd_json.
                t.clear_pending_settle()
                emit_status(f"[AR] Pending settle abandoned; new "
                            f"plan installed. Phase -> EDIT. "
                            f"{get_guidance(task_dir)}")
            else:
                emit_status(f"[AR] Plan validated. Phase -> EDIT. "
                            f"{get_guidance(task_dir)}")
            emit_todowrite_context(
                task_dir, "[AR] Plan validated. Phase -> EDIT.")
    except _Consistency as e:
        emit_status(
            f"[AR] state.json and plan.md/history.jsonl are out "
            f"of sync: {e}. Re-run create_plan.py with the same "
            f"XML — replay_intent at the next open_task will "
            f"reconcile.")
    except _Ownership as e:
        emit_status(f"[AR] post_bash: cannot advance — {e}")


def _print_resume_context(task_dir: str):
    progress = load_progress(task_dir)
    if not progress:
        return
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", "?")
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    failures = progress.get("consecutive_failures", 0)
    plan_ver = progress.get("plan_version", 0)

    emit_status(
        f"[AR] Resume context: Round {rounds}/{max_rounds} | "
        f"Best: {best} | Baseline: {baseline} | "
        f"Failures: {failures} | Plan v{plan_ver}"
    )

    hpath = history_path(task_dir)
    if os.path.exists(hpath):
        with open(hpath, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        if lines:
            emit_status(f"[AR] Last {min(3, len(lines))} rounds:")
            for rec in lines[-3:]:
                rnd = rec.get("round")
                rnd = "?" if rnd is None else str(rnd)
                dec = rec.get("decision", "?")
                desc = rec.get("description", "")[:40]
                emit_status(f"[AR]   R{rnd}: {dec} — {desc}")

    if os.path.exists(plan_path(task_dir)):
        active = get_active_item(task_dir)
        if active:
            emit_status(f"[AR] Active item: {active['id']}: {active['description'][:50]}")
        emit_status("[AR] Read .ar_state/plan.md and .ar_state/history.jsonl for full context.")


if __name__ == "__main__":
    main()
