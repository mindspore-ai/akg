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

"""Round-N (post-EDIT) eval recorder. `record_round` is called in-process
by engine/pipeline.py and returns {decision, best_metric, eval_rounds,
max_rounds, consecutive_failures}."""
from __future__ import annotations

import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    Progress, append_history, load_progress,
    load_state, save_state,
    write_intent, clear_intent, state_transaction,
)
# save_progress not imported — record_round writes progress fields
# directly into state.json as part of one atomic save_state, so it
# can also bundle in pending_settle + expected_history_round.
from task_config import (  # noqa: E402
    EvalOutcome, EvalResult, check_constraints, is_improvement,
    load_task_config,
)
from utils.git_utils import (  # noqa: E402
    auto_rollback, commit_in_task, current_head_short,
)

from .progress_reducer import eval_result_from_data, reduce_round_progress


def record_round(task_dir: str, eval_data: dict,
                 description: str = "optimization round",
                 plan_item: Optional[str] = None) -> dict:
    with state_transaction(task_dir):
        return _record_round(
            task_dir, eval_data, description=description, plan_item=plan_item)


def _record_round(task_dir: str, eval_data: dict,
                  description: str = "optimization round",
                  plan_item: Optional[str] = None) -> dict:
    """Single library entry point for one round of EDIT settlement.

    Atomically commits progress fields + pending_settle + the
    expected_history_round marker in one save_state. Decision flow:
    correctness → constraints → primary-metric presence → improvement."""
    config = load_task_config(task_dir)
    if config is None:
        return {"decision": "ERROR", "error": "task.yaml not found"}

    progress = load_progress(task_dir) or Progress()
    eval_result = eval_result_from_data(eval_data)

    # SSOT round number — derived here, not accepted from the caller (eval
    # reads the same progress.next_round for its verify-dir step). This row's
    # artifacts live at akg_verify/<op>/Iteration<op>_Step<round>_verify.
    round_num = progress.next_round
    decision = "DISCARD"
    commit_hash: Optional[str] = None
    new_failures = progress.consecutive_failures
    new_best_metric = progress.best_metric
    new_best_commit = progress.best_commit
    new_best_speedup = progress.best_speedup

    if not eval_result.correctness:
        decision = "FAIL"
        new_failures = progress.consecutive_failures + 1
        print("[record_round] FAIL: correctness check failed")
    else:
        violations = (check_constraints(eval_result, config.constraints)
                      if config.constraints else [])
        if violations:
            decision = "FAIL"
            new_failures = progress.consecutive_failures + 1
            print(f"[record_round] FAIL: constraint violations: "
                  f"{violations}")
        else:
            cur = eval_result.metrics.get(config.primary_metric)
            best = progress.best_metric
            if (not isinstance(cur, (int, float))
                    or cur != cur):  # NaN guard
                decision = "FAIL"
                new_failures = progress.consecutive_failures + 1
                print(f"[record_round] FAIL: correctness=PASS but primary "
                      f"metric '{config.primary_metric}' missing from "
                      f"{sorted(eval_result.metrics)}")
            elif best is None:
                decision = "KEEP"
            else:
                best_er = EvalResult(outcome=EvalOutcome.OK,
                                     metrics={config.primary_metric: best})
                if is_improvement(
                    eval_result, best_er,
                    metric=config.primary_metric,
                    lower_is_better=config.lower_is_better,
                    threshold=config.improvement_threshold,
                ):
                    decision = "KEEP"
                else:
                    decision = "DISCARD"

    if decision == "KEEP":
        metric_val = eval_result.metrics.get(config.primary_metric)
        metric_str = f"{config.primary_metric}={metric_val}"
        ok, info = commit_in_task(
            task_dir, config.editable_files,
            f"autoresearch: {description} | {metric_str}",
        )
        if not ok:
            # Couldn't preserve kernel state. Earlier we still wrote
            # best_metric=<this round's value> and best_commit=None,
            # which left state.json pointing at a kernel that no
            # commit captured - rollback / resume / report all became
            # unreliable. Demote to FAIL: roll the working tree back,
            # bump consecutive_failures, leave best_* untouched.
            print(f"[record_round] git commit failed: {info}; demoting "
                  f"KEEP -> FAIL (kernel state not preserved)")
            decision = "FAIL"
            new_failures = progress.consecutive_failures + 1
            auto_rollback(task_dir)
        else:
            # "noop" means the edit produced no git-visible diff (e.g.
            # whitespace-only change, or a roll-back to an existing
            # commit's bytes). The kernel we just evaluated IS what HEAD
            # points at, so resolve commit_hash to HEAD instead of None
            # — otherwise a noisier rerun of an existing best would
            # advance best_metric while nulling best_commit, leaving
            # dashboard / report unable to retrieve the winning kernel.
            if info == "noop":
                commit_hash = current_head_short(task_dir) or progress.best_commit
            else:
                commit_hash = info
            new_best_metric = metric_val
            new_best_commit = commit_hash
            # Capture this round's geomean speedup_vs_ref alongside the new
            # best — display reads it back instead of recomputing a
            # mean-ratio from baseline_metric / best_metric.
            _round_speedup = eval_result.metrics.get("speedup_vs_ref")
            new_best_speedup = (float(_round_speedup)
                                if isinstance(_round_speedup, (int, float))
                                and _round_speedup > 0 else new_best_speedup)
            new_failures = 0
            # KEEP outcome is reported by pipeline._print_round_summary
            # (with Round n/m + improvement % + commit hash). Suppress
            # the duplicate stderr print here — record_round still emits
            # for FAIL / DISCARD / git-commit-failed / anchor-message
            # paths where the summary doesn't carry the extra detail.
    else:
        auto_rollback(task_dir)
        print(f"[record_round] {decision}: rolled back editable files")

    # Keep baseline ownership centralized with baseline_init. This covers
    # missing anchors and fingerprint re-anchors.
    reduction = reduce_round_progress(
        progress, eval_result, round_num,
        consecutive_failures=new_failures,
        best_metric=new_best_metric,
        best_commit=new_best_commit,
        best_speedup=new_best_speedup,
    )
    if reduction.anchor.changed and reduction.anchor.message:
        print(f"[record_round] {reduction.anchor.message} from R{round_num}")

    progress = reduction.progress

    kd_json = {
        "decision": decision,
        "best_metric": progress.best_metric,
        # This round's actually-measured primary metric, regardless of
        # whether it became the new best. Used by pipeline.py (settle)
        # to record the real number in plan.md's Settled History for
        # DISCARD and correctness-FAIL rows too (KEEP already had it
        # via best_metric). None for crash-FAIL where eval never
        # produced a number — plan.md renders that as "N/A".
        "round_metric": eval_result.metrics.get(config.primary_metric),
        "eval_rounds": round_num,
        "max_rounds": progress.max_rounds or config.max_rounds,
        "consecutive_failures": progress.consecutive_failures,
        # Identity fields — pipeline.py's replay verifies it's settling
        # the same plan item record_round saw, not the next ACTIVE one
        # after a partial settle.
        "plan_item": plan_item,
        "plan_version": progress.plan_version,
        "round": round_num,
    }

    # Build the history row up-front so the journaled intent and the
    # body write share the same dict — a crash between the two can't
    # leave the journal claiming a row the body never wrote.
    hist: dict[str, Any] = {
        "round": round_num,
        "plan_item": plan_item,
        "description": description,
        "decision": decision,
        "metrics": eval_result.metrics,
        "correctness": eval_result.correctness,
        "error": eval_result.error,
        "commit": commit_hash,
    }
    if decision == "FAIL":
        sig = eval_data.get("failure_signals")
        if isinstance(sig, dict) and (sig.get("primary")
                                      or sig.get("python_error")
                                      or sig.get("signals")):
            hist["failure_signals"] = sig
        tail = (eval_data.get("raw_output_tail") or "").strip()
        if tail:
            hist["raw_output_tail"] = tail[-1500:]

    # ---- Journal/WAL ----
    # Write intent FIRST. Carries enough to reconstruct the
    # post-action state.json on crash recovery: progress fields,
    # kd_json (becomes pending_settle), and round number (the
    # discriminator the replay path keys off).
    state_patch = {
        **progress.to_dict(),
        "pending_settle": kd_json,
        "expected_history_round": round_num,
    }
    write_intent(task_dir, {
        "kind": "round",
        "round": round_num,
        "kd_json": kd_json,
        "state_patch": state_patch,
    })

    # ---- Body: history.jsonl ----
    append_history(task_dir, hist)

    # ---- Commit: state.json ----
    # Single atomic merge of new progress + pending_settle + expected_
    # history_round. A SIGKILL between append_history and this save
    # leaves history.jsonl ahead of state — pipeline.py's
    # replay_intent() at entry reconstructs pending_settle from the
    # journal so the existing replay branch can finish.
    state = load_state(task_dir) or {}
    state.update(state_patch)
    save_state(task_dir, state)

    # ---- Journal clear ----
    # Last; a crash here just leaves a leftover intent.json which
    # replay_intent will recognise (state already caught up) and
    # drop with "cleared" status.
    clear_intent(task_dir)

    return kd_json
