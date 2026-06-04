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

"""Round-0 SEED eval recorder. `run_baseline_init(task_dir, eval_data)`
is called in-process by engine/baseline.py and returns that script's
exit code (see `_EXIT_FOR`). Owns the post-baseline phase transition
via PhaseController.on_baseline_settled — the post-Bash hook only
emits guidance off the phase already on disk."""
from __future__ import annotations

import os
import sys
from enum import Enum
from typing import NamedTuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase_machine import (  # noqa: E402
    BASELINE, Progress, append_history, load_progress, load_state,
    save_state, history_path, write_intent, clear_intent, replay_intent,
    write_phase,
)
from task_config import EvalOutcome, load_task_config  # noqa: E402
from utils.baseline_anchor import valid_metric  # noqa: E402
from utils.git_utils import current_head_short  # noqa: E402

from .progress_reducer import reduce_baseline_init
from .transition import PhaseController


# ---------------------------------------------------------------------------
# Baseline retry precheck
# ---------------------------------------------------------------------------
# The baseline transaction owns the SEED history.jsonl row + the
# progress fields in state.json. A crash between the SEED append and
# the state save leaves an orphan SEED row that a retried baseline
# would happily overwrite the state metrics for — without touching the
# row itself, since _existing_seed_row() de-duplicates the append.
# Result: SEED row carries the OLD eval's metrics, state carries the
# NEW eval's metrics, and consistency check can't see the divergence
# because expected_history_round=0 matches the orphan row.
#
# precheck_baseline runs at engine/baseline.py entry, BEFORE
# run_eval, and decides whether to skip eval entirely (already
# committed, or rebuilt from journal), refuse (orphan SEED with no
# journal), or proceed. Skipping is what closes the
# silently-divergent-metrics window — the second eval never runs
# when the first one's row is already on disk.

class BaselinePrecheckOutcome(Enum):
    PROCEED      = "proceed"        # no body, no state — run eval fresh
    ALREADY_DONE = "already_done"   # state + body agree on a prior commit
    ORPHAN_SEED  = "orphan_seed"    # body ahead of state, no journal
    MISSING_SEED = "missing_seed"   # state ahead of body, no journal


class BaselinePrecheck(NamedTuple):
    outcome: BaselinePrecheckOutcome
    detail:  str


def precheck_baseline(task_dir: str) -> BaselinePrecheck:
    """Classify the baseline transaction's pre-run state. Pure read
    after replay — the caller decides whether to act on the outcome.
    Does NOT mutate state beyond the replay heal.

    Always runs replay_intent first so any in-flight baseline
    transaction is healed before we read state. After replay, four
    disk shapes are possible (matrix of progress_initialized × SEED
    row presence):

      progress_initialized | SEED row | outcome
      ---------------------|----------|--------
      True                 | True     | ALREADY_DONE
      True                 | False    | MISSING_SEED  (state ahead of body)
      False                | True     | ORPHAN_SEED   (body ahead of state)
      False                | False    | PROCEED       (normal first run)

    ALREADY_DONE: caller advances the phase (idempotent
    on_baseline_settled) and exits with baseline_exit_code, no eval.

    ORPHAN_SEED / MISSING_SEED: state and body have diverged in an
    off-flow way the journal cannot reconstruct. Caller fails loud
    instead of running eval — re-evaluating would either overwrite
    the surviving artifact's metric (ORPHAN_SEED → state) or fabricate
    a body that doesn't match state (MISSING_SEED).

    PROCEED: caller runs run_eval then commits via record_baseline.
    """
    # Heal first. replay rebuilds state from journal when the SEED
    # row landed but state didn't; discards the journal when the
    # SEED row never landed; clears it when state already caught up.
    replay = replay_intent(task_dir)
    if replay is not None:
        print(f"[baseline] replay_intent {replay['action']}: "
              f"{replay['detail']}", file=sys.stderr)

    state = load_state(task_dir) or {}
    seed_on_disk = _existing_seed_row(task_dir)
    progress_done = bool(state.get("progress_initialized"))

    if progress_done and seed_on_disk:
        return BaselinePrecheck(
            BaselinePrecheckOutcome.ALREADY_DONE,
            "baseline already committed (state.progress_initialized "
            "and SEED row on disk); skipping re-eval.")

    if progress_done and not seed_on_disk:
        return BaselinePrecheck(
            BaselinePrecheckOutcome.MISSING_SEED,
            "state.progress_initialized is True but history.jsonl has "
            "no SEED row. State and body have diverged in an off-flow "
            "way the journal cannot heal (intent.json absent). "
            "Running eval here would write a fresh SEED row whose "
            "metrics don't match the committed state.baseline_metric. "
            "Recover by either (a) restoring the SEED row from a "
            "backup of history.jsonl, or (b) clearing the offending "
            "state fields (progress_initialized + baseline_*) and "
            "re-running baseline.py from scratch.")

    if seed_on_disk and not progress_done:
        return BaselinePrecheck(
            BaselinePrecheckOutcome.ORPHAN_SEED,
            "history.jsonl carries a SEED row but state.progress_"
            "initialized is False and no intent.json journal is "
            "present. A retry that runs run_eval here would write the "
            "new measurement into state.baseline_metric while the "
            "orphan SEED row keeps the prior measurement, silently "
            "divorcing dashboards/report from sticky baseline. "
            "Recover by either (a) hand-setting state.baseline_metric "
            "/ state.seed_metric / state.progress_initialized to "
            "match the SEED row, or (b) deleting the SEED row from "
            "history.jsonl to start fresh.")

    return BaselinePrecheck(BaselinePrecheckOutcome.PROCEED, "")


def _existing_seed_row(task_dir: str) -> bool:
    """True iff history.jsonl already has a round=0 SEED row.
    Idempotency hook for the baseline transaction: a SIGKILL between
    append_history and save_state can leave an orphan SEED row. With
    this gate + the journal at the call site, a retry either: (a) sees
    no SEED row and starts fresh, or (b) sees the orphan SEED row and
    skips the append while still rewriting state from the journal."""
    import os as _os, json as _json
    path = history_path(task_dir)
    if not _os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
    except OSError:
        return False
    if not first:
        return False
    try:
        row = _json.loads(first)
    except _json.JSONDecodeError:
        return False
    return row.get("round") == 0 and row.get("decision") == "SEED"


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


def baseline_exit_code(task_dir: str) -> int:
    """Map the COMMITTED baseline_outcome (in state.json's progress
    bundle) back to engine/baseline.py's exit-code convention.

    Used by the ALREADY_DONE retry path: precheck_baseline skipped
    run_eval, so the exit code can't come from this round's outcome.
    It has to come from the prior committed outcome — otherwise a
    retry after a crashed INFRA_FAIL baseline would silently return
    0, and scaffold.py / batch / supervisors that key off rc==4 to
    refuse activation would treat the task as healthy.

    Returns 0 when progress isn't readable or outcome is missing
    (defensive — ALREADY_DONE shouldn't ever reach here without
    progress_initialized, but the consequence of a None is "treat as
    activatable" which matches the silent default that existed
    before this helper)."""
    progress = load_progress(task_dir)
    if progress is None:
        return 0
    outcome_str = progress.get("baseline_outcome")
    if not outcome_str:
        return 0
    try:
        outcome = EvalOutcome(outcome_str)
    except ValueError:
        return 0
    return _EXIT_FOR.get(outcome, 0)


def run_baseline_init(task_dir: str, eval_data: dict) -> int:
    """Library entry point. engine/baseline.py calls this after
    run_eval finishes; the return value becomes that script's exit
    code. Side effects (progress, history, phase) are durable on disk
    before this returns."""
    config = load_task_config(task_dir)
    if config is None:
        print("[baseline] ERROR: task.yaml not found", file=sys.stderr)
        return 1

    # Single hard gate: a baseline EXISTS iff the PyTorch reference was
    # measured. Without a valid ref latency there is nothing to optimise
    # against (speedup / KEEP-DISCARD are meaningless), so commit nothing
    # — no Progress, no SEED row, progress_initialized stays False — and
    # park the task at BASELINE so a retry after the env/ref/worker is
    # fixed re-evaluates cleanly. Seed timing is NOT a substitute.
    metrics = eval_data.get("metrics") or {}
    if not valid_metric(metrics.get("ref_latency_us")):
        write_phase(task_dir, BASELINE)
        clear_intent(task_dir)
        print(f"[baseline] no valid ref baseline "
              f"({eval_data.get('error') or 'ref_latency_us missing'}) — "
              f"baseline pending; fix env/ref/worker and re-run.",
              file=sys.stderr)
        return _EXIT_FOR[EvalOutcome.INFRA_FAIL]

    existing = load_progress(task_dir) or Progress()
    head_commit = current_head_short(task_dir) or "unknown"
    reduction = reduce_baseline_init(
        existing, config, eval_data, best_commit=head_commit)

    if reduction.dropped_seed_metric is not None:
        print(f"[baseline] dropping wrong-output seed timing "
              f"(latency_us={reduction.dropped_seed_metric:.1f}) — "
              f"kernel failed correctness "
              f"so its measurement cannot anchor best_metric.",
              file=sys.stderr)
    if reduction.anchor.message:
        print(f"[baseline] {reduction.anchor.message}", file=sys.stderr)

    progress_fields = reduction.progress.to_dict()

    # ---- Journal ----
    # Write intent FIRST so a crash between body append and state
    # commit is recoverable: pipeline.py / future baseline invocations
    # call replay_intent which rebuilds state.json from this payload
    # when it sees an orphan SEED row.
    write_intent(task_dir, {
        "kind": "baseline",
        "round": 0,
        "progress_fields": progress_fields,
    })

    # ---- Body: history.jsonl ----
    # Idempotent: a previous crash may have left an orphan SEED row.
    # Skip the append in that case — replay_intent / this save_state
    # below will reconcile state.json against the existing row.
    if not _existing_seed_row(task_dir):
        append_history(task_dir, {
            "round": 0,
            "description": "seed kernel initial eval",
            "decision": "SEED",
            "metrics": reduction.metrics,
            "outcome": reduction.outcome.value,
            "correctness": reduction.correctness,
            "commit": head_commit,
        })

    # ---- Commit: state.json ----
    # Single atomic commit: merge progress fields + bump
    # expected_history_round so the consistency check matches the row
    # above. progress_initialized flips on here — this is the
    # discriminator load_progress() uses to distinguish "claimed by a
    # session but never measured" from "has baseline data". Resume /
    # dashboard rely on it to avoid offering a Round 0/0 view on a task
    # that hasn't run baseline yet.
    state = load_state(task_dir) or {}
    for k, v in progress_fields.items():
        state[k] = v
    state["expected_history_round"] = 0
    state["progress_initialized"] = True
    save_state(task_dir, state)

    # ---- Journal clear ----
    clear_intent(task_dir)

    # Phase transition is owned by on_baseline_settled. Only KERNEL_FAIL
    # reaches here now (OK falls through below; INFRA_FAIL was gated out
    # above before any commit): kernel_fail commits the ref baseline and
    # routes to PLAN so the main loop rewrites the broken seed.
    if reduction.outcome != EvalOutcome.OK:
        PhaseController(task_dir).on_baseline_settled()
        print(f"[baseline] {reduction.outcome.value}: "
              f"{eval_data.get('error') or '(no detail)'}",
              file=sys.stderr)
        return _EXIT_FOR[reduction.outcome]

    if reduction.seed_metric is None:
        # Degenerate: outcome=OK but no primary metric. Leave phase at
        # BASELINE so the agent retries.
        print(f"[baseline] ERROR: outcome=OK but no valid "
              f"{config.primary_metric}; treating as kernel-no-timing.",
              file=sys.stderr)
        return 2

    PhaseController(task_dir).on_baseline_settled()
    print(f"[baseline] Initialized: task={config.name}, "
          f"seed_{config.primary_metric}={reduction.seed_metric}, "
          f"baseline({reduction.anchor.source})={reduction.anchor.metric}, "
          f"commit={head_commit}", file=sys.stderr)
    return 0
