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

"""Run baseline eval and initialize .ar_state.

Calls `task_config.run_eval` in-process — same Python tree as
pipeline.py and the rest of the engine, so no subprocess / JSON-tail
parse between this script and the eval pipeline. The eval pipeline
itself still spawns `eval_kernel.py` (via utils.eval_runner.local_eval);
that's where the crash isolation lives.

Usage:
    python scripts/engine/baseline.py <task_dir>
        [--device-id N]
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPTS_ROOT)
sys.path.insert(0, SCRIPT_DIR)
from task_config import load_task_config, run_eval
from task_config.metric_policy import EvalOutcome, EvalResult
from utils.failure_extractor import extract_failure_signals, format_for_stdout
from task_handle import (
    open_task, Role,
    TaskCorrupted, TaskConsistencyError, TaskOwnershipError,
)
from workflow.baseline import BaselinePrecheckOutcome, baseline_exit_code
from workflow.transition import PhaseController


def _eval_result_to_dict(result) -> dict:
    """Match the dict shape `Task.record_baseline` consumes."""
    eval_data = {
        "outcome": result.outcome.value,
        "correctness": result.correctness,
        "metrics": result.metrics or {},
        "error": result.error,
        "error_source": result.error_source,
    }
    if not result.correctness or result.error:
        eval_data["failure_signals"] = extract_failure_signals(
            result.raw_output).to_dict()
        eval_data["raw_output_tail"] = (result.raw_output or "")[-4000:]
    return eval_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir")
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--worker-url", default=None,
                        help="Worker URL(s), comma-separated. Overrides "
                             "task.yaml worker.urls.")
    args = parser.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)

    # Validate task.yaml is loadable up front. open_task's heal+check
    # gate doesn't read task.yaml, so we surface a missing config here
    # before claiming ownership.
    if load_task_config(task_dir) is None:
        print("[baseline] ERROR: task.yaml not found in task_dir",
              file=sys.stderr)
        sys.exit(1)

    worker_urls = None
    if args.worker_url:
        worker_urls = [u.strip() for u in args.worker_url.split(",") if u.strip()]

    # open_task heals (replay_intent) + consistency check + claims
    # ownership. On exception inside the with-block, Task.__exit__
    # releases the claim — failed runs can't leak ownership. Normal
    # completion (any rc) must NOT raise inside the with — sys.exit
    # raises SystemExit which __exit__ counts as a failure and would
    # release the claim, breaking the next post_bash hook. The body
    # is extracted into _run_baseline returning rc; main does
    # sys.exit(rc) AFTER the with-block ends cleanly.
    rc = 4  # fallback for caught exceptions
    try:
        with open_task(task_dir, role=Role.AGENT) as t:
            rc = _run_baseline(task_dir, t, args, worker_urls)
    except TaskConsistencyError as e:
        print(f"[baseline] state inconsistent: {e}", file=sys.stderr)
    except TaskOwnershipError as e:
        print(f"[baseline] cannot run: {e}", file=sys.stderr)
        rc = 2
    except TaskCorrupted as e:
        # ORPHAN_SEED / MISSING_SEED raise here (caller-flagged
        # corruption); __exit__ released claim. Defensive: also
        # catches a misused record_baseline call (non-PROCEED outcome
        # or empty eval_data), which would be a bug in the dispatch.
        print(f"[baseline] FATAL: {e}", file=sys.stderr)

    sys.exit(rc)


def _run_baseline(task_dir: str, t, args, worker_urls) -> int:
    """Body of baseline.py main. Returns rc; does NOT sys.exit. Caller
    invokes sys.exit(rc) after the with-block ends, so __exit__'s
    release-on-exception path only fires for genuine failures (raises),
    not for normal completion with rc != 0.

    Classify pre-run state BEFORE run_eval so non-PROCEED outcomes
    don't burn device time. Four outcomes:
      PROCEED      → normal first run; run eval + record.
      ALREADY_DONE → state + body agree; advance phase, return
                     committed outcome's rc.
      ORPHAN_SEED  → body ahead of state; raise TaskCorrupted (eval
                     would silently overwrite the state metric while
                     keeping the orphan row's metric in history).
      MISSING_SEED → state ahead of body; raise TaskCorrupted (eval
                     would write a fresh row whose metrics don't
                     match the committed state).
    """
    pre = t.baseline_preflight()

    if pre.outcome == BaselinePrecheckOutcome.ALREADY_DONE:
        # Idempotent: ensure phase advanced past BASELINE, return rc
        # mapped from the committed outcome.
        PhaseController(task_dir).on_baseline_settled()
        rc = baseline_exit_code(task_dir)
        print(f"[baseline] {pre.detail} (committed outcome → "
              f"exit={rc})", file=sys.stderr)
        return rc

    if pre.outcome in (BaselinePrecheckOutcome.ORPHAN_SEED,
                       BaselinePrecheckOutcome.MISSING_SEED):
        # Raise so __exit__ releases the claim — genuine corruption
        # the operator must inspect, no further work in this session.
        raise TaskCorrupted(f"({pre.outcome.value}): {pre.detail}")

    # PROCEED: run eval + commit.
    print("[baseline] Running baseline eval...", flush=True)
    try:
        result = run_eval(task_dir, t.config,
                          device_id=args.device_id,
                          worker_urls=worker_urls)
    except Exception as e:
        # run_eval normally converts internal failures to
        # EvalResult(INFRA_FAIL, ...); reaching here means it raised.
        # Funnel the exception through the same path as a normal
        # INFRA_FAIL so the ref-baseline gate inside run_baseline_init
        # owns parking the task at BASELINE with no committed progress.
        # Single owner of the baseline-pending invariant — no duplicate
        # write_phase / clear_intent here.
        print(f"[baseline] run_eval raised "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        result = EvalResult(
            outcome=EvalOutcome.INFRA_FAIL,
            error=f"run_eval raised {type(e).__name__}: {e}",
            error_source="infra",
        )

    eval_data = _eval_result_to_dict(result)

    # Pretty-print structured failure signals (UB overflow, aivec
    # trap, OOM, correctness mismatch, ...) — mirrors pipeline.py so
    # the seed-failure → PLAN flow surfaces the same actionable
    # summary the EDIT loop does.
    if not eval_data.get("correctness", False) or eval_data.get("error"):
        if eval_data.get("error"):
            print(f"[baseline] Error: {eval_data['error']}",
                  flush=True)
        pretty = format_for_stdout(
            eval_data.get("failure_signals") or {})
        if pretty:
            print(pretty, flush=True)
        elif eval_data.get("raw_output_tail"):
            print("[baseline] Eval log tail (no structured "
                  "signals matched):", flush=True)
            print(eval_data["raw_output_tail"], flush=True)

    return t.record_baseline(eval_data)


if __name__ == "__main__":
    main()
