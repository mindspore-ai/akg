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

"""Round-0 SEED eval recorder. `run_baseline_init(task_dir, eval_json)`
is called in-process by engine/baseline.py and returns that script's
exit code (see `_EXIT_FOR`). Owns the post-baseline phase transition
via PhaseController.on_baseline_settled — the post-Bash hook only
emits guidance off the phase already on disk."""

# pylint: disable=wrong-import-position
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    Progress, append_history, load_progress, save_progress,
)
from task_config import EvalOutcome, load_task_config  # noqa: E402
from utils.git_utils import current_head_short  # noqa: E402

from .progress_reducer import reduce_baseline_init
from .transition import PhaseController


# Outcome → exit code. Binary: 0 = task is activatable (kernel may need
# rewrite via PLAN, but state machine handles that), non-zero = task is
# NOT activatable. scaffold.py's "rc != 0 → surface error" stays accurate;
# the slash command's "non-zero exit → stop and report" gates only on
# INFRA_FAIL now. The full 3-way outcome lives in progress.baseline_outcome
# for downstream readers (post_bash, dashboard, stop_save).
_EXIT_FOR = {
    EvalOutcome.OK: 0,
    EvalOutcome.KERNEL_FAIL: 0,
    EvalOutcome.INFRA_FAIL: 4,
}


def run_baseline_init(task_dir: str, eval_json: str) -> int:
    """Library entry point. engine/baseline.py calls this after
    eval_wrapper finishes; the return value becomes that script's exit
    code. Side effects (progress, history, phase) are durable on disk
    before this returns."""
    config = load_task_config(task_dir)
    if config is None:
        print("[baseline] ERROR: task.yaml not found", file=sys.stderr)
        return 1

    eval_data = json.loads(eval_json)
    existing = load_progress(task_dir) or Progress()
    head_commit = current_head_short(task_dir) or "unknown"
    reduction = reduce_baseline_init(
        existing, config, eval_data, best_commit=head_commit)

    if reduction.dropped_seed_metric is not None:
        print("[baseline] dropping wrong-output seed timing "
              f"(latency_us={reduction.dropped_seed_metric:.1f}) — "
              "kernel failed correctness "
              "so its measurement cannot anchor best_metric.",
              file=sys.stderr)
    if reduction.anchor.message:
        print(f"[baseline] {reduction.anchor.message}", file=sys.stderr)

    save_progress(task_dir, reduction.progress, stamp=True)

    # Round 0 logs the SEED kernel's initial eval. `metrics.latency_us` is the
    # seed's timing; `metrics.ref_latency_us` (if present) is the PyTorch
    # baseline used as the speedup anchor.
    append_history(task_dir, {
        "round": 0,
        "description": "seed kernel initial eval",
        "decision": "SEED",
        "metrics": reduction.metrics,
        "outcome": reduction.outcome.value,
        "correctness": reduction.correctness,
        "commit": head_commit,
    })

    if reduction.outcome != EvalOutcome.OK:
        # Phase transition (PLAN for kernel_fail, untouched for infra_fail)
        # is owned by on_baseline_settled, which dispatches on the
        # `baseline_outcome` we just persisted. Calling it here keeps
        # non-hook callers (notebook re-runs, tests) on the same code
        # path as the Bash-hook flow.
        PhaseController(task_dir).on_baseline_settled()
        print(f"[baseline] {reduction.outcome.value}: "
              f"{eval_data.get('error') or '(no detail)'}",
              file=sys.stderr)
        return _EXIT_FOR[reduction.outcome]

    if reduction.seed_metric is None:
        # Degenerate: outcome=OK but no primary metric (rare). Treat as
        # kernel-no-timing — leave phase at BASELINE so the agent retries.
        print("[baseline] ERROR: outcome=OK but no valid "
              f"{config.primary_metric}; treating as kernel-no-timing.",
              file=sys.stderr)
        return 2

    PhaseController(task_dir).on_baseline_settled()
    print(f"[baseline] Initialized: task={config.name}, "
          f"seed_{config.primary_metric}={reduction.seed_metric}, "
          f"baseline({reduction.anchor.source})={reduction.anchor.metric}, "
          f"commit={head_commit}", file=sys.stderr)
    return 0
