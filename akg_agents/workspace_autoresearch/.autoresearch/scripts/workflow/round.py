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

# pylint: disable=wrong-import-position
from __future__ import annotations

import math
import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    Progress, append_history, auto_rollback, load_progress, save_progress,
)
from task_config import (  # noqa: E402
    EvalOutcome, EvalResult, check_constraints, is_improvement,
    load_task_config,
)
from utils.git_utils import commit_in_task, current_head_short  # noqa: E402

from .progress_reducer import eval_result_from_data, reduce_round_progress


def _judge_pre_commit(eval_result, config) -> tuple:
    """Run the correctness / constraint / metric gates. Returns
    (decision, fail_log) where decision is "KEEP_CANDIDATE" or "FAIL";
    fail_log is the stderr line to emit (empty for non-FAIL)."""
    if not eval_result.correctness:
        return "FAIL", "[keep_or_discard] FAIL: correctness check failed"
    violations = (check_constraints(eval_result, config.constraints)
                  if config.constraints else [])
    if violations:
        return "FAIL", ("[keep_or_discard] FAIL: constraint violations: "
                        f"{violations}")
    cur = eval_result.metrics.get(config.primary_metric)
    if not isinstance(cur, (int, float)) or math.isnan(float(cur)):
        return "FAIL", ("[keep_or_discard] FAIL: correctness=PASS but "
                        f"primary metric '{config.primary_metric}' missing "
                        f"from {sorted(eval_result.metrics)}")
    return "KEEP_CANDIDATE", ""


def _decide_keep_vs_discard(eval_result, progress, config) -> str:
    """KEEP_CANDIDATE -> KEEP or DISCARD via the improvement threshold."""
    best = progress.best_metric
    if best is None:
        return "KEEP"
    best_er = EvalResult(outcome=EvalOutcome.OK,
                         metrics={config.primary_metric: best})
    if is_improvement(
        eval_result, best_er,
        metric=config.primary_metric,
        lower_is_better=config.lower_is_better,
        threshold=config.improvement_threshold,
    ):
        return "KEEP"
    return "DISCARD"


def _apply_keep(task_dir: str, eval_result, progress, config,
                description: str) -> tuple:
    """Commit the kernel for KEEP. Returns
    (final_decision, commit_hash, new_best_metric, new_best_commit,
    new_failures). Demotes to FAIL on commit_in_task failure — without a
    commit, progress.json would point at a kernel no commit captured."""
    metric_val = eval_result.metrics.get(config.primary_metric)
    metric_str = f"{config.primary_metric}={metric_val}"
    ok, info = commit_in_task(
        task_dir, config.editable_files,
        f"autoresearch: {description} | {metric_str}",
    )
    if not ok:
        print(f"[keep_or_discard] git commit failed: {info}; demoting "
              "KEEP -> FAIL (kernel state not preserved)",
              file=sys.stderr)
        auto_rollback(task_dir)
        return ("FAIL", None, progress.best_metric, progress.best_commit,
                progress.consecutive_failures + 1)
    if info == "noop":
        commit_hash = current_head_short(task_dir) or progress.best_commit
    else:
        commit_hash = info
    print(f"[keep_or_discard] KEEP: {metric_str} (commit: {commit_hash})",
          file=sys.stderr)
    return "KEEP", commit_hash, metric_val, commit_hash, 0


def _compose_history(round_num: int, plan_item: Optional[str],
                     description: str, decision: str, eval_result,
                     eval_data: dict, commit_hash: Optional[str]) -> dict:
    """Assemble the history.jsonl row. FAIL rows carry structured
    failure_signals + raw_output_tail (truncated 1500) so DIAGNOSE has
    something concrete to reason about."""
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
    if decision != "FAIL":
        return hist
    sig = eval_data.get("failure_signals")
    if isinstance(sig, dict) and (sig.get("primary")
                                  or sig.get("python_error")
                                  or sig.get("signals")):
        hist["failure_signals"] = sig
    tail = (eval_data.get("raw_output_tail") or "").strip()
    if tail:
        hist["raw_output_tail"] = tail[-1500:]
    return hist


def record_round(task_dir: str, eval_data: dict,
                 description: str = "optimization round",
                 plan_item: Optional[str] = None) -> dict:
    """Single library entry point for one round of EDIT settlement.

    Decision flow: correctness gate -> constraint gate -> primary-metric
    presence -> improvement check. KEEP attempts a git commit; commit
    failure demotes to FAIL. `consecutive_failures` counts only real
    failures (FAIL = kernel broken); DISCARD is a REPLAN signal, not a
    failure."""
    config = load_task_config(task_dir)
    if config is None:
        return {"decision": "ERROR", "error": "task.yaml not found"}

    progress = load_progress(task_dir) or Progress()
    eval_result = eval_result_from_data(eval_data)
    round_num = progress.eval_rounds + 1

    pre_decision, fail_log = _judge_pre_commit(eval_result, config)
    if pre_decision == "FAIL":
        decision = "FAIL"
        commit_hash = None
        new_failures = progress.consecutive_failures + 1
        new_best_metric = progress.best_metric
        new_best_commit = progress.best_commit
        print(fail_log, file=sys.stderr)
        auto_rollback(task_dir)
        print(f"[keep_or_discard] {decision}: rolled back editable files",
              file=sys.stderr)
    else:
        decision = _decide_keep_vs_discard(eval_result, progress, config)
        if decision == "KEEP":
            (decision, commit_hash, new_best_metric, new_best_commit,
             new_failures) = _apply_keep(task_dir, eval_result, progress,
                                         config, description)
        else:
            commit_hash = None
            new_failures = progress.consecutive_failures
            new_best_metric = progress.best_metric
            new_best_commit = progress.best_commit
            auto_rollback(task_dir)
            print(f"[keep_or_discard] {decision}: rolled back editable files",
                  file=sys.stderr)

    reduction = reduce_round_progress(
        progress, config, eval_result, round_num,
        consecutive_failures=new_failures,
        best_metric=new_best_metric,
        best_commit=new_best_commit,
    )
    if reduction.anchor.changed and reduction.anchor.message:
        print(f"[keep_or_discard] {reduction.anchor.message} from "
              f"R{round_num}", file=sys.stderr)

    progress = reduction.progress
    save_progress(task_dir, progress)

    hist = _compose_history(round_num, plan_item, description, decision,
                            eval_result, eval_data, commit_hash)
    append_history(task_dir, hist)

    return {
        "decision": decision,
        "best_metric": progress.best_metric,
        "eval_rounds": round_num,
        "max_rounds": progress.max_rounds or config.max_rounds,
        "consecutive_failures": progress.consecutive_failures,
    }
