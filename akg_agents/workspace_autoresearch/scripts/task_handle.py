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

"""Task handle — single entry point for "enter a task and do something with it."

Task is the only legal way for script-level entries (engine/baseline.py,
engine/pipeline.py, engine/create_plan.py, resume.py,
hooks/post_bash._handle_activation, batch/run.py) to manipulate the
state-store. Going through it ensures that heal / consistency-check /
ownership-claim / progress-load / phase-read / history+state write /
intent clear all fire in the correct order, and that schema additions
propagate to every entry without per-script patches.

  with open_task(task_dir, role="agent") as t:
      # __enter__: replay_intent → consistency check → claim ownership
      #             per role; raise on failure.
      if not t.progress_initialized:
          rc = t.record_baseline(eval_data)
      else:
          t.record_round(eval_data, description=desc, plan_item=pid)
          t.settle_round()

Roles
=====

  - ``agent``       — caller IS the Claude session driving the task.
                      enter: heal + check + claim ownership (refuse
                      if owned by a different live session, unless
                      force=True). engine/baseline, engine/pipeline,
                      engine/create_plan, resume, post_bash activation.
  - ``supervisor``  — caller orchestrates a separate Claude session
                      (batch/run.py). enter: heal + check, NO claim.
                      Used to read post-run state without taking
                      ownership.
  - ``reader``      — read-only consumers (dashboard, monitor).
                      enter: NO heal, NO claim. Pure observation.

Why context manager
===================
__enter__ enforces order (heal → check → claim) that callers
otherwise composed by hand. __exit__ on exception leaves intent.json
in place so the next caller's replay can finish what we started — the
journal is the recovery mechanism, not the with-block. On a clean
exit, individual transaction methods have already cleared their own
intent files.

What stays free-functional
==========================
Hook paths are read-only gates/reporters over committed state. Activation is
the sole hook path that opens a Task because it claims ownership and selects
the executable resume phase. Baseline, plan, and round transitions are
committed by their engine transactions before post-tool reporting runs.

Caller pattern: do NOT sys.exit inside the with-block
=====================================================
__exit__ releases ownership on exception. ``sys.exit(rc)`` raises
``SystemExit``, which is an exception — so a script that uses
``sys.exit(rc)`` inside the with-block to express NORMAL completion
(rc=0 success, rc=4 INFRA_FAIL, etc.) would unclaim the task. The
next post_bash hook would then call ``get_task_dir()`` and find
nothing, dropping every subsequent phase guidance / todo / status
emission.

The contract for entry scripts:

  def main():
      rc = 1  # fallback for caught open_task exceptions
      try:
          with open_task(td, role=AGENT) as t:
              rc = _do_work(t)   # _do_work RETURNS rc, never sys.exit
      except (TaskConsistencyError, TaskOwnershipError) as e:
          ...
      sys.exit(rc)  # AFTER the with-block — claim already kept

  def _do_work(t) -> int:
      # Genuine failures that should release the claim: RAISE.
      # Normal completion (any rc): RETURN.
      if must_stop:
          raise TaskCorrupted(...)   # __exit__ releases claim
      return 4                       # __exit__ keeps claim

This split is the load-bearing rule: ``return rc`` means "I'm done,
keep my ownership"; ``raise`` means "I failed, drop my ownership."
resume.py is the one entry that intentionally uses ``sys.exit(1)``
inside the with-block — that's the validation-failure path where
ownership release IS the right outcome.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from enum import Enum
from typing import Any, Iterator, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from phase_machine import (  # noqa: E402
    BASELINE, DIAGNOSE, EDIT, FINISH, INIT, PLAN, REPLAN,
    # Reads
    load_state, load_progress, read_phase, task_summary,
    task_owner_info,
    get_active_item, diagnose_state,
    # Writes / lifecycle
    save_state, state_transaction,
    set_task_dir, clear_active_task, touch_heartbeat,
    # Journal
    write_intent, clear_intent,
    # Consistency
    require_state_consistency, format_state_inconsistency,
    # Paths
    edit_marker_path, state_record_path,
)
from utils.git_utils import auto_rollback  # noqa: E402
from task_config import load_task_config  # noqa: E402
from workflow.baseline import (  # noqa: E402
    precheck_baseline, BaselinePrecheckOutcome,
    run_baseline_init,
)
from workflow.round import record_round as _record_round  # noqa: E402
from workflow.transition import phase_after_round, phase_on_resume  # noqa: E402
from workflow.planning import PlanStore  # noqa: E402


# ---------------------------------------------------------------------------
# Typed errors — callers catch on exit-code-mapping or message routing
# ---------------------------------------------------------------------------

class TaskError(Exception):
    """Base for all Task-lifecycle errors."""


class TaskOwnershipError(TaskError):
    """Refused to claim because another live session owns the task."""


class TaskConsistencyError(TaskError):
    """Post-replay state still inconsistent (off-flow corruption)."""


class TaskNotInitialized(TaskError):
    """Caller asked for `progress` on a task whose baseline hasn't
    committed. The `progress_initialized` discriminator gates this."""


class TaskCorrupted(TaskError):
    """A transaction precondition failed in a way the journal can't
    auto-heal — typically an orphan body artifact with no intent."""


class TaskPhaseError(TaskError):
    """An event was invoked from a phase that cannot own it."""


# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------

class Role(str, Enum):
    AGENT      = "agent"       # heals + checks + claims
    SUPERVISOR = "supervisor"  # heals + checks, no claim
    READER     = "reader"      # no heal, no check, no claim


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class Task:
    """Lifecycle wrapper around state.json + bodies + journal.

    NOT thread-safe. NOT re-entrant — one __enter__ per instance.
    Construct via ``open_task(task_dir, role=...)`` (a thin alias);
    direct instantiation works too.
    """

    def __init__(self, task_dir: str, *, role: Role | str = Role.AGENT,
                 force: bool = False):
        self.task_dir = os.path.abspath(task_dir)
        self.role = Role(role) if isinstance(role, str) else role
        self.force = force
        self._entered = False
        self._claimed = False

    # ---- Lifecycle ------------------------------------------------------

    def __enter__(self) -> "Task":
        if self._entered:
            raise RuntimeError("Task already entered (use a new instance)")
        if not os.path.isdir(self.task_dir):
            raise TaskError(f"Not a directory: {self.task_dir}")

        if self.role == Role.READER:
            # Pure observation — no heal, no claim.
            self._entered = True
            return self

        # 1. Heal + check. Folded in require_state_consistency since
        # commit 327e89b: replay_intent runs first (auto_replay=True
        # default), then the cross-file consistency check. Any
        # inconsistency that survives is genuinely off-flow.
        report = require_state_consistency(self.task_dir,
                                           on_inconsistent="report")
        if not report["consistent"]:
            raise TaskConsistencyError(format_state_inconsistency(report))

        # 2. Claim ownership (agent only) + bump heartbeat. Supervisor
        # doesn't claim and doesn't bump — it observes a task that
        # another Claude session is (or was) driving, and bumping
        # heartbeat from a supervisor would make a dead task look
        # alive to is_task_active for the next window.
        if self.role == Role.AGENT:
            if not set_task_dir(self.task_dir, force=self.force):
                owner = task_owner_info(self.task_dir) or {}
                raise TaskOwnershipError(
                    f"refused to claim {self.task_dir} — owned by "
                    f"session_id={owner.get('session_id') or '<none>'} "
                    f"(heartbeat fresh). Stop the other session or "
                    f"re-open with force=True to take over.")
            self._claimed = True
            touch_heartbeat(self.task_dir)
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Journal and ownership are ORTHOGONAL concerns; __exit__
        # handles them independently.
        #
        # Journal (intent.json): NEVER touched here. Body bytes +
        # intent.json are the recovery mechanism — the next caller's
        # replay_intent at open_task heals whatever we left. Individual
        # transaction methods (record_baseline, record_round,
        # settle_round, commit_plan) clear their own intent on success;
        # the leftover-on-exception state is the journal doing its job.
        #
        # Ownership (state.owner): released on exception, kept on
        # clean exit. Earlier the contract was "never release on exit"
        # — but that was the ownership and journal concerns tangled
        # into one rule. The journal doesn't need ownership to replay
        # (it's file-based), and a Task body that raised is by
        # definition unable to follow through; freeing the claim lets
        # the next caller (manual retry, --resume from another
        # session, supervisor re-run) take over without --force. Clean
        # exit keeps the claim so subsequent in-session subprocesses
        # see "owner = us" via get_task_dir.
        if exc_type is not None and self._claimed:
            try:
                clear_active_task(expected_task_dir=self.task_dir)
            except Exception:
                # Release is best-effort. If state.json is unreadable
                # at this point the next caller will see a stale
                # owner + stale heartbeat (which goes cold quickly);
                # not worth crashing on top of an existing exception.
                pass
            self._claimed = False
        return False  # don't suppress

    # ---- Read properties / accessors ------------------------------------

    @property
    def state(self) -> dict:
        """Raw state.json dict (empty when missing). Internal helper for
        the other accessors; external callers should prefer the typed
        properties below."""
        return load_state(self.task_dir) or {}

    @property
    def phase(self) -> str:
        """Current phase (one of ALL_PHASES). INIT when state is
        missing or unrecognised."""
        return read_phase(self.task_dir)

    @property
    def progress_initialized(self) -> bool:
        """True once baseline has committed. False during the window
        between scaffold and the first baseline.py — accessing
        .progress in that window raises TaskNotInitialized."""
        return bool(self.state.get("progress_initialized"))

    @property
    def progress(self):
        """Progress dataclass. Raises TaskNotInitialized when baseline
        hasn't committed yet — the discriminator is progress_initialized
        on state.json, set by run_baseline_init's final save_state."""
        p = load_progress(self.task_dir)
        if p is None:
            raise TaskNotInitialized(
                f"baseline has not committed for {self.task_dir}; "
                f"progress fields are not yet meaningful. Run "
                f"baseline.py first (or check .progress_initialized "
                f"before reading .progress).")
        return p

    @property
    def summary(self) -> Optional[dict]:
        """Bundled view (phase / owner / progress / consistency).
        None when no state.json exists."""
        return task_summary(self.task_dir)

    @property
    def owner(self) -> Optional[dict]:
        return self.state.get("owner")

    @property
    def pending_settle(self) -> Optional[dict]:
        return self.state.get("pending_settle")

    @property
    def plan_version(self) -> int:
        return int(self.state.get("plan_version") or 0)

    @property
    def config(self):
        """task.yaml loaded into TaskConfig, or None when missing."""
        return load_task_config(self.task_dir)

    def active_plan_item(self) -> Optional[dict]:
        return get_active_item(self.task_dir)

    def diagnose_state(self):
        """DiagnoseState NamedTuple (action / attempts / artifact_reason).
        Valid in any phase but only meaningful in DIAGNOSE."""
        return diagnose_state(self.task_dir)

    # ---- Transactional methods -----------------------------------------
    # All journal+body+commit sequences live here. Callers never
    # touch write_intent / append_history / save_state directly.

    def baseline_preflight(self):
        """Read-only classification of the baseline transaction's
        pre-run state. Returns a BaselinePrecheck (outcome + detail).
        Pure read after replay — call this BEFORE expensive work
        (run_eval) to decide whether to skip, fail, or proceed.

        The classify/act split exists because run_eval is expensive
        (device time, worker round-trip). preflight is read-only, so
        calling it before run_eval lets callers decide cheaply whether
        to skip, fail, or proceed without paying the eval cost.
        """
        self._require_entered()
        precheck = precheck_baseline(self.task_dir)
        # A crash may have committed the baseline event (including phase=PLAN)
        # before the CLI returned.  Permit that one idempotent retry shape;
        # every event that could still evaluate must start in BASELINE.
        if not (precheck.outcome == BaselinePrecheckOutcome.ALREADY_DONE
                and self.phase == PLAN):
            self.require_phase(BASELINE, action="baseline")
        return precheck

    def record_baseline(self, eval_data: dict) -> int:
        """Commit the baseline transaction (journal → SEED row →
        state → clear journal → phase transition). Returns the
        engine-baseline exit code (0 / 4 per _EXIT_FOR).

        Refuses non-PROCEED preflight outcomes (ALREADY_DONE /
        ORPHAN_SEED / MISSING_SEED) — the caller MUST inspect
        baseline_preflight() first and dispatch on the outcome. This
        defensive gate keeps record_baseline as a pure commit: it
        doesn't decide whether to act, it just acts.

        Refuses empty / structurally-incomplete eval_data — earlier
        engine/baseline.py would pass `{}` as a fast-path signal,
        which silently routed through reduce_baseline_init and wrote
        null metrics into state. The contract here is: eval_data MUST
        be the dict shape utils.eval_summary.eval_result_to_dict
        produces (outcome + correctness + metrics keys at minimum).
        """
        self._require_entered()
        self.require_phase(BASELINE, action="record baseline")
        # Defensive: refuse misuse. ALREADY_DONE etc. are caller
        # responsibility — see baseline_preflight().
        pre = precheck_baseline(self.task_dir)
        if pre.outcome != BaselinePrecheckOutcome.PROCEED:
            raise TaskCorrupted(
                f"record_baseline called with preflight outcome "
                f"{pre.outcome.value!r}: {pre.detail} Caller must "
                f"branch on baseline_preflight() and only call "
                f"record_baseline on PROCEED.")
        # Defensive: real eval_data required. The minimal shape is
        # what eval_result_to_dict produces; absence of `outcome` is
        # the cheap discriminator for "empty / placeholder dict".
        if not isinstance(eval_data, dict) or "outcome" not in eval_data:
            raise TaskCorrupted(
                f"record_baseline requires a real eval_data dict "
                f"(outcome / correctness / metrics). Got: "
                f"{type(eval_data).__name__} with keys "
                f"{sorted(eval_data) if isinstance(eval_data, dict) else '<n/a>'}.")
        # run_baseline_init owns journal + history + state + phase +
        # outcome→rc mapping.
        return run_baseline_init(self.task_dir, eval_data)

    def record_round(self, eval_data: dict, *,
                     description: str = "optimization round",
                     plan_item: Optional[str] = None) -> dict:
        """Round transaction (journal → history → state with
        pending_settle → clear journal). Returns kd_json which the
        caller hands to settle_round."""
        self._require_entered()
        self.require_phase(EDIT, action="record round")
        if self.pending_settle is not None:
            raise TaskPhaseError(
                "cannot record a new round while pending_settle must be replayed")
        return _record_round(self.task_dir, eval_data,
                             description=description, plan_item=plan_item)

    def settle_round(self) -> dict:
        with state_transaction(self.task_dir):
            return self._settle_round()

    def _settle_round(self) -> dict:
        """Settle plan.md, phase, and pending_settle as one event."""
        self.require_phase(EDIT, action="settle round")
        pending = self.pending_settle
        if pending is None:
            raise TaskPhaseError("cannot settle without pending_settle")
        try:
            settle_json = PlanStore(self.task_dir).settle_result(pending)
            next_phase = phase_after_round(self.task_dir)
        except RuntimeError as exc:
            raise TaskCorrupted(f"settle failed: {exc}") from exc

        # plan.md is the body.  Commit the target phase and sentinel clear
        # together: before this save replay remains legal; after it there is
        # no in-flight settlement left for a hook or CLI to repair.
        state = load_state(self.task_dir)
        if state is None:
            raise TaskCorrupted("state.json disappeared during settlement")
        state["phase"] = next_phase
        state["pending_settle"] = None
        try:
            save_state(self.task_dir, state)
        except OSError as exc:
            raise TaskCorrupted(
                f"state commit failed after plan settlement: {exc}") from exc

        marker = edit_marker_path(self.task_dir)
        if os.path.exists(marker):
            os.remove(marker)

        # FINISH report generation — was inlined in pipeline._post_settle.
        finish = (next_phase == FINISH)
        if finish:
            try:
                from report import write_report
                rp = write_report(self.task_dir)
                if rp:
                    print(f"[task] FINISH report: "
                          f"{os.path.relpath(rp, self.task_dir)}")
            except Exception as e:
                print(f"[task] report generation failed: {e}")

        return {
            "settled_item": settle_json.get("settled_item"),
            "decision":     pending.get("decision"),
            "next_phase":   next_phase,
            "finish":       finish,
        }

    def commit_plan(self, items: list) -> dict:
        with state_transaction(self.task_dir):
            return self._commit_plan(items)

    def _commit_plan(self, items: list) -> dict:
        """Plan transaction (journal → plan.md → state with new
        plan_version + next_pid → clear journal).

        `items` is the validated list from create_plan._parse_items_xml
        + _validate_items + _check_diversity — Task does NOT re-parse
        XML. create_plan.py is the only XML parser; Task is the only
        plan committer.

        Returns: {version, item_ids, active, dropped, path}.
        """
        self._require_entered()
        prog = load_progress(self.task_dir)
        if prog is None or not os.path.exists(state_record_path(self.task_dir)):
            raise TaskCorrupted("cannot commit a plan before baseline progress")
        version = prog.plan_version + 1
        ps = PlanStore(self.task_dir)
        next_pid = ps.compute_next_pid(prog.next_pid)
        settled_rows = ps.parse_settled_history()
        dropped_pids = [it["id"] for it in ps.parse_pending()]
        item_ids, new_next_pid = PlanStore.allocate_ids(len(items), next_pid)

        progress_fields: dict[str, Any] = {
            "plan_version": version,
            "next_pid":     new_next_pid,
        }

        state_before = load_state(self.task_dir) or {}
        source_phase = state_before.get("phase", INIT)
        pending_recovery = source_phase == EDIT and bool(
            state_before.get("pending_settle"))
        if source_phase not in (PLAN, REPLAN, DIAGNOSE) \
                and not pending_recovery:
            raise TaskPhaseError(
                "commit plan requires PLAN/REPLAN/DIAGNOSE or "
                f"EDIT+pending_settle; current phase={source_phase}")
        if source_phase == DIAGNOSE:
            progress_fields["consecutive_failures"] = 0

        state_patch: dict[str, Any] = {
            **progress_fields,
            "phase": EDIT,
            "pending_settle": None,
            "expected_plan_version": version,
        }

        # ---- Journal ----
        # Plan intent payload mirrors round/baseline: enough to
        # reconstruct state.json from the body on crash recovery.
        write_intent(self.task_dir, {
            "kind":            "plan",
            "version":         version,
            "state_patch":     state_patch,
        })

        # ---- Body: plan.md ----
        ps.write(version, item_ids, items, settled_rows)

        # ---- Commit: state.json ----
        new_prog = prog.apply(**progress_fields)
        state = load_state(self.task_dir)
        if state is None:
            raise TaskCorrupted("state.json disappeared during plan commit")
        state.update(new_prog.to_dict())
        state.update(state_patch)
        try:
            save_state(self.task_dir, state)
        except OSError as exc:
            raise TaskCorrupted(
                f"state commit failed after writing plan.md: {exc}") from exc

        # ---- Journal clear ----
        clear_intent(self.task_dir)

        return {
            "version":    version,
            "item_ids":   item_ids,
            "active":     item_ids[0],
            "dropped":    dropped_pids,
            "path":       ps.path,
        }

    # ---- Activation -----------------------------------------------------

    def activate(self, *, fresh: bool) -> str:
        with state_transaction(self.task_dir):
            return self._activate(fresh=fresh)

    def _activate(self, *, fresh: bool) -> str:
        """Commit the executable phase for a fresh or resumed task.

        ``pending_settle`` takes precedence to heal tasks interrupted by
        older versions between their phase write and sentinel clear.
        """
        self._require_entered()
        current = self.phase
        if fresh:
            target = BASELINE
        elif self.pending_settle is not None:
            target = EDIT
        elif current in (INIT, PLAN, REPLAN) \
                or (current == BASELINE and self.progress_initialized):
            target = phase_on_resume(self.task_dir)
        else:
            target = current
        if target != current:
            state = load_state(self.task_dir) or {}
            state["phase"] = target
            save_state(self.task_dir, state)
        return target

    # ---- Rollback / sentinel maintenance --------------------------------

    def rollback_edit(self) -> None:
        """auto_rollback the editable files and drop the .edit_started
        marker. Used by pipeline.py on quick_check / infra_fail
        outcomes."""
        self._require_entered()
        auto_rollback(self.task_dir)
        marker = edit_marker_path(self.task_dir)
        if os.path.exists(marker):
            try:
                os.remove(marker)
            except OSError:
                pass

    # ---- Explicit ownership ops (rare; prefer role-based __enter__) -----

    def release(self, *, force: bool = False) -> bool:
        """Explicit ownership release. Idempotent. Returns True on
        success (or already-released), False when refused."""
        ok = clear_active_task(expected_task_dir=self.task_dir, force=force)
        if ok:
            self._claimed = False
        return ok

    # ---- Internals ------------------------------------------------------

    def _require_entered(self) -> None:
        if not self._entered:
            raise RuntimeError("Task method called before __enter__")

    def require_phase(self, *allowed: str, action: str) -> str:
        """Fail closed when a domain event is invoked from the wrong phase."""
        self._require_entered()
        phase = self.phase
        if phase not in allowed:
            expected = "/".join(allowed)
            raise TaskPhaseError(
                f"{action} requires phase {expected}; current phase={phase}")
        return phase


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

@contextmanager
def open_task(task_dir: str, *, role: Role | str = Role.AGENT,
              force: bool = False) -> Iterator[Task]:
    """Open a Task and yield it. The context manager guarantees
    __enter__ ran (heal + check + claim per role) before the body
    executes, and that __exit__ leaves intent in place on exception
    so the next caller can replay it.

    Example:
        with open_task(td, role="agent") as t:
            if not t.progress_initialized:
                t.record_baseline(eval_data)
    """
    t = Task(task_dir, role=role, force=force)
    with t:
        yield t


__all__ = [
    "Task", "Role", "open_task",
    "TaskError", "TaskOwnershipError", "TaskConsistencyError",
    "TaskNotInitialized", "TaskCorrupted",
]
