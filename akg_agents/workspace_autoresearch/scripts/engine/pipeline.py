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
    4. t.settle_round → update plan.md, advance phase, clear pending_settle
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
from quick_check import check_editable_files, _run_smoke_test as _run_smoke
from task_config import run_eval
from utils.failure_extractor import extract_failure_signals, format_for_stdout
from phase_machine import get_guidance, FINISH
from task_handle import (
    open_task, Role, TaskCorrupted, TaskConsistencyError,
    TaskOwnershipError,
)


# pipeline._run_settle is still defined here (not on Task) because
# Task.settle_round imports it via a local-scope `from engine.pipeline
# import _run_settle` to avoid the package import cycle (workflow ←
# task_handle ← engine.pipeline ← workflow). The function stays
# self-contained: it's the only piece of "what does settle MEAN"
# logic that has to know about plan.md row encoding + idempotency.
def _run_settle(task_dir: str, kd_json: dict) -> tuple:
    """Settle the active plan item in-process. Returns
    ``(ok: bool, error_tail: str, settle_json: dict | None)``.

    Idempotent w.r.t. plan_item: when kd_json's `plan_item` already
    appears in plan.md's Settled History table, settle is considered
    already-done (replay-safe). Otherwise it actually runs.
    """
    from workflow import PlanStore
    from phase_machine import get_active_item
    try:
        decision = kd_json.get("decision", "FAIL")
        # THIS round's measured metric, regardless of decision. None
        # only when eval never produced a number (kernel crash before
        # any case ran) — plan.md renders that as "N/A". KEEP /
        # DISCARD / correctness-FAIL all surface the real latency.
        metric_val = kd_json.get("round_metric")

        store = PlanStore(task_dir)
        if not store.exists():
            return False, "plan.md not found", None

        expected_item = kd_json.get("plan_item")
        if expected_item:
            current_active = get_active_item(task_dir)
            current_id = (current_active or {}).get("id")
            if current_id != expected_item:
                settled_rows = store.parse_settled_history() or ""
                if f"| {expected_item} |" in settled_rows:
                    return True, "", {
                        "settled_item": expected_item,
                        "decision": decision,
                        "metric": metric_val,
                        "already_settled": True,
                    }
                return False, (
                    f"plan.md ACTIVE is {current_id!r}, kd_json "
                    f"expected {expected_item!r}, and {expected_item} "
                    f"does NOT appear in Settled History. Plan is "
                    f"either malformed or was rewritten by an "
                    f"unrelated create_plan — pretending settle "
                    f"succeeded would lose this round's kd_json. "
                    f"Refusing; state.pending_settle retained for "
                    f"manual inspection."
                ), None

        settled_id, _ = store.settle_active(decision, metric_val)
        return True, "", {
            "settled_item": settled_id,
            "decision": decision,
            "metric": metric_val,
        }
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def _emit_settle_failure(task_dir: str, error_tail: str) -> None:
    print(f"[PIPELINE] SETTLE FAILED. plan.md was NOT updated. "
          f"history.jsonl + state.json already moved during this "
          f"round; re-running this script will RETRY SETTLE ONLY "
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
    """Status line + guidance after a settled round."""
    # Progress is guaranteed initialised here (we're in EDIT, which
    # implies baseline committed). Read via Task's typed accessor.
    progress = t.progress
    rounds = progress.eval_rounds
    max_rounds = progress.max_rounds
    best = progress.best_metric
    baseline = progress.baseline_metric
    failures = progress.consecutive_failures

    improv = ""
    if (best is not None and baseline is not None
            and isinstance(best, (int, float))
            and isinstance(baseline, (int, float))
            and baseline != 0 and best != 0):
        pct = (baseline - best) / abs(baseline) * 100
        speedup = baseline / best
        improv = f" ({speedup:.2f}x vs ref, {pct:+.1f}%)"

    print(f"\n{'=' * 50}")
    print(f"[{decision}] {settled_id} | Round {rounds}/{max_rounds} | "
          f"Best: {best}{improv} | Failures: {failures}")
    print(f"Phase -> {next_phase}")
    print(f"{'=' * 50}")
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
    # === Replay-only settle ===
    # open_task already ran replay_intent; if state.pending_settle is
    # non-null after that, a prior round committed history but the
    # settle didn't finish. Re-run settle only — do NOT re-eval.
    kd_json = t.pending_settle
    if kd_json:
        print(f"[PIPELINE] Retrying settle from state.pending_settle "
              f"(skipping quick_check/eval/record_round).", flush=True)
        try:
            result = t.settle_round(kd_json)
        except TaskCorrupted:
            # _emit_settle_failure already wrote the stderr banner
            # inside Task.settle_round. The agent can re-run
            # pipeline.py to re-enter this branch idempotently —
            # keep claim.
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
        print(f"[PIPELINE] Auto-rolled back. Fix and re-edit.")
        print(get_guidance(t.task_dir))
        return 0  # rollback is normal pipeline flow — agent will re-edit

    print("[PIPELINE] Quick check PASS", flush=True)

    # === Step 2: Eval ===
    print("[PIPELINE] Running eval...", flush=True)
    try:
        result = run_eval(t.task_dir, config)
    except Exception as e:
        t.rollback_edit()
        print(f"[PIPELINE] EVAL ERROR: run_eval raised "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1

    eval_json = {
        "outcome": result.outcome.value,
        "correctness": result.correctness,
        "metrics": result.metrics or {},
        "error": result.error,
        "error_source": result.error_source,
    }
    if not result.correctness or result.error:
        eval_json["failure_signals"] = extract_failure_signals(
            result.raw_output).to_dict()
        eval_json["raw_output_tail"] = (result.raw_output or "")[-4000:]

    correctness = eval_json.get("correctness", False)
    metrics = eval_json.get("metrics", {})
    print(f"[PIPELINE] Eval: correctness={correctness}, "
          f"metrics={metrics}", flush=True)

    # infra_fail: eval pipeline broke before kernel was meaningfully
    # exercised. Roll back and skip the round — recording a FAIL here
    # would mislead later DIAGNOSE / KEEP / DISCARD.
    if eval_json.get("outcome") == "infra_fail":
        t.rollback_edit()
        print(f"[PIPELINE] INFRA_FAIL: "
              f"{eval_json.get('error', 'no data')}. "
              f"Rolled back, not recording round.", flush=True)
        return 0

    if not correctness or eval_json.get("error"):
        if eval_json.get("error"):
            print(f"[PIPELINE] Error: {eval_json['error']}", flush=True)
        pretty = format_for_stdout(eval_json.get("failure_signals") or {})
        if pretty:
            print(pretty, flush=True)
        elif eval_json.get("raw_output_tail"):
            print("[PIPELINE] Eval log tail (no structured signals "
                  "matched):", flush=True)
            print(eval_json["raw_output_tail"], flush=True)

    # === Step 3: Record round ===
    kd_json = t.record_round(eval_json, description=desc,
                             plan_item=plan_item)
    if kd_json.get("decision") == "ERROR":
        print(f"[PIPELINE] KEEP/DISCARD ERROR: {kd_json.get('error')}")
        return 1
    decision = kd_json.get("decision", "FAIL")

    # === Step 4 + 5: Settle (advances phase + clears pending) ===
    try:
        result = t.settle_round(kd_json)
    except TaskCorrupted:
        # banner already printed by Task.settle_round; keep claim so
        # the agent's next pipeline.py invocation enters the replay
        # branch.
        return 1
    _print_round_summary(t, decision, result["settled_item"] or "?",
                         result["next_phase"])
    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <task_dir>")
        sys.exit(1)

    task_dir = os.path.abspath(sys.argv[1])

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

    sys.exit(rc)


if __name__ == "__main__":
    main()
