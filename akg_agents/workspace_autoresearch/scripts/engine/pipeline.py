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

"""Post-edit pipeline — runs ALL mechanical steps after Claude Code edits code.

Claude Code does the LLM work (plan, edit, diagnose). Then calls this:
    python scripts/engine/pipeline.py <task_dir>

Steps inside `with open_task(td, role="agent")`:
    1. quick_check → fail? rollback, report
    2. eval → get metrics
    3. t.record_round → KEEP/DISCARD/FAIL (journals + history + state)
    4. t.settle_round → plan.md + atomic phase/pending_settle commit
    5. print status + next guidance

Output: human-readable status to stdout. Claude Code sees it and acts accordingly.

Recovery: open_task at entry calls replay_intent + consistency check.
Any in-flight round transaction (intent.json + body landed before
state save) is reconstructed before this script proceeds. If state
.pending_settle is non-null after replay, this script runs the
replay-only settle branch (skips quick_check/eval/record_round).
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPTS_ROOT)
sys.path.insert(0, SCRIPT_DIR)

# Logging is owned by akg_agents/__init__.py (root -> stdout) so the eval
# INFO chain stays chronological and the AR Phase guidance lands last; the
# imports below trigger it. No basicConfig here — single owner.

from quick_check import (
    check_editable_files, effective_edit_issue,
    _run_smoke_test as _run_smoke,
)
from task_config import run_eval
from utils.eval_summary import (
    eval_result_to_dict, print_eval_metrics, print_failure_signals,
)
from utils.settings import recorded_speedup
from phase_machine import EDIT, get_guidance
from task_handle import (
    open_task, Role, TaskCorrupted, TaskConsistencyError,
    TaskOwnershipError, TaskPhaseError,
)


def _emit_settle_failure(task_dir: str, error_tail: str) -> None:
    print(f"[PIPELINE] SETTLE INCOMPLETE. plan.md may already contain "
          f"the idempotent settlement, but state.pending_settle is still "
          f"authoritative; re-running this script will RETRY SETTLE ONLY "
          f"(kd_json was persisted to state.pending_settle) — it "
          f"will NOT re-run quick_check/eval/record_round.\n"
          f"\n"
          f"Recovery options (do NOT hand-edit plan.md):\n"
          f"  1. Fix the underlying cause from the error tail below, "
          f"then re-run pipeline.py — the replay-only path will "
          f"retry settle on the same kd_json.\n"
          f"  2. If the failure is structural (plan.md malformed, "
          f"no (ACTIVE) item, etc.) and settle cannot recover, run "
          f"create_plan.py to write a fresh plan.md. While "
          f"state.pending_settle is non-null, hooks/guard_bash "
          f"allows create_plan.py in EDIT phase as a recovery path; "
          f"on successful create_plan validation hooks/post_bash "
          f"clears state.pending_settle. The orphan history.jsonl "
          f"row stays (audit trail).\n"
          f"\n"
          f"error: {error_tail}", file=sys.stderr)


def _print_round_summary(t, decision: str, settled_id: str,
                         next_phase: str) -> None:
    """Status line + guidance after a settled round.

    Commit hash is included on KEEP rounds — that's the only decision
    where best_commit changed this round, so reporting it here surfaces
    "what kernel just got committed" without a duplicate stderr print
    from record_round.
    """
    # Progress is guaranteed initialised here (we're in EDIT, which
    # implies baseline committed). Read via Task's typed accessor.
    progress = t.progress
    rounds = progress.eval_rounds
    max_rounds = progress.max_rounds
    best = progress.best_metric
    failures = progress.consecutive_failures

    # Stored geomean speedup (best_speedup); pct derived from it so both numbers
    # are tied. Empty when unset — never re-derive from baseline/best latencies.
    speedup = recorded_speedup(progress)
    improv = ""
    if speedup is not None:
        pct = (1.0 - 1.0 / speedup) * 100
        improv = f" ({speedup:.2f}x vs ref, {pct:+.1f}%)"

    best_str = f"{best:.2f}" if isinstance(best, (int, float)) else str(best)
    commit_str = ""
    if decision == "KEEP" and progress.best_commit:
        commit_str = f" | commit: {progress.best_commit}"

    print(f"\n{'=' * 50}")
    print(f"[{decision}] {settled_id} | Round {rounds}/{max_rounds} | "
          f"Best: {best_str}{improv} | Failures: {failures}{commit_str}")
    print(f"Phase -> {next_phase}")
    print(f"{'=' * 50}")
    # Drain any pending stderr (e.g. tracebacks from settle path) so
    # the AR Phase guidance lands as the very last line in the capture.
    sys.stderr.flush()
    sys.stdout.flush()
    print(get_guidance(t.task_dir))


def _run_with_task(t) -> int:
    """Body of pipeline.py inside the Task context manager. Returns rc;
    does NOT sys.exit. Caller invokes sys.exit(rc) after the with-block
    so __exit__'s release-on-exception path only fires for genuine
    failures (raised exceptions), not for normal completion with rc != 0.

    All pipeline-level failures here (quick_check fail, eval crash,
    settle failure, etc.) return rc instead of raising so the session
    keeps its ownership claim — the agent will re-edit / retry and the
    next post_bash hook needs to find the active task."""
    t.require_phase(EDIT, action="pipeline")

    # === Replay-only settle ===
    # open_task already ran replay_intent; if state.pending_settle is
    # non-null after that, a prior round committed history but the
    # settle didn't finish. Re-run settle only — do NOT re-eval.
    kd_json = t.pending_settle
    if kd_json:
        print("[PIPELINE] Retrying settle from state.pending_settle "
              "(skipping quick_check/eval/record_round).", flush=True)
        try:
            result = t.settle_round()
        except TaskCorrupted as exc:
            _emit_settle_failure(t.task_dir, str(exc))
            return 1
        _print_round_summary(t, kd_json.get("decision", "?"),
                             result["settled_item"] or "?",
                             result["next_phase"])
        return 0

    # Normal pipeline: quick_check → eval → record → settle.
    config = t.config
    if config is None:
        print("[PIPELINE] ERROR: task.yaml not found")
        return 1

    active = t.active_plan_item()
    desc = active["description"] if active else "optimization round"
    plan_item = active["id"] if active else None

    # === Step 1: Quick check ===
    print("[PIPELINE] Running quick_check...", flush=True)
    try:
        edit_issue = effective_edit_issue(t.task_dir, config)
        file_issues = [edit_issue] if edit_issue else []
        if not file_issues:
            file_issues = check_editable_files(t.task_dir, config)
        smoke_errors = _run_smoke(t.task_dir, config)
    except Exception as exc:
        file_issues = [{"file": "(internal)",
                        "report": f"quick_check crashed: "
                                  f"{type(exc).__name__}: {exc}",
                        "errors": []}]
        smoke_errors = []

    if file_issues or smoke_errors:
        t.rollback_edit()
        blob: dict = {"ok": False}
        if file_issues:
            blob["file_issues"] = file_issues
        if smoke_errors:
            blob["smoke_errors"] = smoke_errors
        print(f"[PIPELINE] QUICK CHECK FAIL: "
              f"{json.dumps(blob, ensure_ascii=False)[:200]}")
        print("[PIPELINE] Auto-rolled back. Fix and re-edit.")
        print(get_guidance(t.task_dir))
        return 0  # rollback is normal pipeline flow — agent will re-edit

    print("[PIPELINE] Quick check PASS", flush=True)

    # === Step 2: Eval ===
    print("[PIPELINE] Running eval...", flush=True)
    try:
        # Verify-dir step = the round about to be recorded. Same SSOT rule
        # (progress.next_round) record_round derives from, so the StepNN
        # artifacts and the history row agree — no number passed across.
        result = run_eval(t.task_dir, config,
                          current_step=t.progress.next_round)
    except Exception as e:
        t.rollback_edit()
        print(f"[PIPELINE] EVAL ERROR: run_eval raised "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    eval_json = eval_result_to_dict(result)

    # Compact summary + per-shape table replaces the old dict dump
    # (per_shape_descs alone was ~4KB of unstructured noise). Full
    # per-shape arrays still live in .ar_state/akg_verify/<task>/...
    # for offline inspection.
    print_eval_metrics(eval_json, "PIPELINE")

    # infra_fail: eval pipeline broke before kernel was meaningfully
    # exercised. Roll back and skip the round — recording a FAIL here
    # would mislead later DIAGNOSE / KEEP / DISCARD.
    if eval_json.get("outcome") == "infra_fail":
        t.rollback_edit()
        print(f"[PIPELINE] INFRA_FAIL: "
              f"{eval_json.get('error', 'no data')}. "
              f"Rolled back, not recording round.", flush=True)
        return 0

    print_failure_signals(eval_json, "PIPELINE")

    # === Step 3: Record round ===
    kd_json = t.record_round(eval_json, description=desc,
                             plan_item=plan_item)
    if kd_json.get("decision") == "ERROR":
        print(f"[PIPELINE] KEEP/DISCARD ERROR: {kd_json.get('error')}")
        return 1
    decision = kd_json.get("decision", "FAIL")

    # === Step 4 + 5: Settle (advances phase + clears pending) ===
    try:
        result = t.settle_round()
    except TaskCorrupted as exc:
        _emit_settle_failure(t.task_dir, str(exc))
        return 1
    _print_round_summary(t, decision, result["settled_item"] or "?",
                         result["next_phase"])
    return 0


def main():
    argv = sys.argv[1:]
    # --trace: keep the msprof trace dirs (timeline + CSVs) for analysis.
    if "--trace" in argv:
        os.environ["AKG_PROF_KEEP_RES"] = "1"
        argv = [a for a in argv if a != "--trace"]
    if not argv:
        print("Usage: python pipeline.py <task_dir> [--trace]")
        sys.exit(1)

    task_dir = os.path.abspath(argv[0])

    # The body returns rc; sys.exit happens AFTER the with-block so
    # SystemExit from a non-zero normal completion doesn't trip
    # __exit__'s release-on-exception path (which would unclaim the
    # task and break the next post_bash hook's get_task_dir()).
    rc = 1
    try:
        with open_task(task_dir, role=Role.AGENT) as t:
            rc = _run_with_task(t)
    except TaskConsistencyError as e:
        print(f"[PIPELINE] REFUSING TO RUN — {e}", file=sys.stderr)
    except TaskOwnershipError as e:
        print(f"[PIPELINE] cannot run: {e}", file=sys.stderr)
        rc = 2
    except TaskPhaseError as e:
        print(f"[PIPELINE] refused: {e}", file=sys.stderr)
        rc = 2

    sys.exit(rc)


if __name__ == "__main__":
    main()
