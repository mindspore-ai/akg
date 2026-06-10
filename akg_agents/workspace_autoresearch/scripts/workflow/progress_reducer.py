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

"""Pure-ish reducers for workflow progress updates.

workflow.baseline and workflow.round still own durable I/O (save_progress
/ append_history) and side effects (git commit / rollback / phase
transition). This module owns the Progress field transitions so baseline
and round cannot quietly drift on baseline anchor, seed metric, and
round counter rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from phase_machine import Progress  # type: ignore
from task_config import EvalOutcome, EvalResult  # type: ignore
from utils.baseline_anchor import (  # type: ignore
    AnchorDecision,
    refresh_round_anchor,
    resolve_baseline_init_anchor,
    valid_metric,
)


def read_outcome(eval_data: dict) -> EvalOutcome:
    """Resolve outcome from eval_data, falling back to the `correctness`
    bool when `outcome` is absent."""
    s = eval_data.get("outcome")
    if s is None:
        s = (EvalOutcome.OK.value if eval_data.get("correctness")
             else EvalOutcome.KERNEL_FAIL.value)
    try:
        return EvalOutcome(s)
    except ValueError:
        return EvalOutcome.INFRA_FAIL


def eval_result_from_data(eval_data: dict) -> EvalResult:
    return EvalResult(
        outcome=read_outcome(eval_data),
        metrics=eval_data.get("metrics", {}),
        error=eval_data.get("error"),
        error_source=eval_data.get("error_source"),
    )


def _shape_progress_fields(metrics: dict) -> dict[str, Any]:
    """Progress fields derived from the current eval's shape metadata."""
    fields: dict[str, Any] = {}
    n_cases = metrics.get("num_cases")
    if isinstance(n_cases, int) and n_cases >= 1:
        fields["num_cases"] = int(n_cases)
    descs = metrics.get("per_shape_descs")
    if isinstance(descs, list) and descs:
        fields["per_shape_descs"] = [str(d) for d in descs if d]
    return fields


@dataclass(frozen=True)
class BaselineReduction:
    progress: Progress
    outcome: EvalOutcome
    correctness: bool
    metrics: dict
    seed_metric: Optional[float]
    dropped_seed_metric: Optional[float]
    anchor: AnchorDecision


def reduce_baseline_init(existing: Progress, config: Any, eval_data: dict,
                         best_commit: str) -> BaselineReduction:
    """`best_commit` pins Progress.best_commit when the SEED kernel
    produced a valid timing."""
    outcome = read_outcome(eval_data)
    correctness = outcome == EvalOutcome.OK
    metrics = eval_data.get("metrics", {})
    error_source = eval_data.get("error_source")

    raw_seed = metrics.get(config.primary_metric)
    seed_metric = float(raw_seed) if valid_metric(raw_seed) else None
    dropped_seed_metric = None
    if not correctness and seed_metric is not None:
        dropped_seed_metric = seed_metric
        seed_metric = None

    anchor = resolve_baseline_init_anchor(existing, metrics)

    progress = Progress(
        task=config.name,
        eval_rounds=0,
        max_rounds=config.max_rounds,
        best_metric=seed_metric,
        best_commit=(best_commit if seed_metric is not None
                     else "seed_profile_failed"),
        baseline_metric=anchor.metric,
        baseline_source=anchor.source,
        baseline_outcome=outcome.value,
        baseline_error_source=error_source,
        baseline_per_shape_us=anchor.per_shape_us,
        baseline_fingerprint=anchor.fingerprint,
        seed_metric=seed_metric,
        consecutive_failures=0,
        plan_version=0,
        **_shape_progress_fields(metrics),
    )
    return BaselineReduction(
        progress=progress,
        outcome=outcome,
        correctness=correctness,
        metrics=metrics,
        seed_metric=seed_metric,
        dropped_seed_metric=dropped_seed_metric,
        anchor=anchor,
    )


@dataclass(frozen=True)
class RoundReduction:
    progress: Progress
    anchor: AnchorDecision


def reduce_round_progress(progress: Progress, eval_result: EvalResult,
                          round_num: int,
                          consecutive_failures: int,
                          best_metric: Optional[float],
                          best_commit: Optional[str]) -> RoundReduction:
    anchor = refresh_round_anchor(progress, eval_result.metrics)
    new_progress = progress.apply(
        eval_rounds=round_num,
        consecutive_failures=consecutive_failures,
        best_metric=best_metric,
        best_commit=best_commit,
        baseline_metric=anchor.metric,
        baseline_source=anchor.source,
        baseline_per_shape_us=anchor.per_shape_us,
        baseline_fingerprint=anchor.fingerprint,
        **_shape_progress_fields(eval_result.metrics),
    )
    return RoundReduction(progress=new_progress, anchor=anchor)
