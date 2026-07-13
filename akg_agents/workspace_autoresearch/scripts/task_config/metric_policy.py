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

"""Metric comparison and constraint checking.

Pure data-shape and arithmetic logic — no I/O, no subprocess, no YAML.
The `EvalResult` dataclass is the contract eval_runner writes into;
downstream consumers (keep_or_discard, baseline_init, dashboard) read
from it.

What lives here:
  - `EvalOutcome`          — classification enum, single source of truth for
                             what happened (OK / kernel fail / infra fail).
  - `EvalResult`           — the result dataclass.
  - `is_improvement`       — current-vs-best comparison with relative-%
                             threshold and direction (`lower_is_better`).
  - `check_constraints`    — hard-constraint check
                             ({metric: (op_str, threshold)} →
                              list of violation strings).

Why a separate module: the comparison logic is the only piece of
task_config that has zero external dependencies and zero side effects;
splitting it out lets every other module that needs only EvalResult
import from here without dragging in YAML / urllib / tarfile.
"""
import operator as _op
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EvalOutcome(str, Enum):
    """What happened in eval. KERNEL_FAIL = agent can fix via PLAN→EDIT;
    INFRA_FAIL = operator only (broken --ref, missing env, transport down)."""
    OK = "ok"
    KERNEL_FAIL = "kernel_fail"
    INFRA_FAIL = "infra_fail"


@dataclass
class EvalResult:
    outcome: EvalOutcome = EvalOutcome.INFRA_FAIL
    metrics: dict = field(default_factory=dict)
    error: Optional[str] = None
    raw_output: str = ""
    # "ref" → broken --ref file (the only sub-flavor of INFRA_FAIL the
    # downstream messages distinguish). None on success or other failures.
    error_source: Optional[str] = None
    # Path to the on-disk FAIL report (full per-case + complete log) the agent
    # opens with its file reader, instead of a truncated stdout dump.
    fail_report: Optional[str] = None
    # failure_extractor signals, already parsed by akg_eval from the FULL log;
    # forwarded so pipeline doesn't re-parse a truncated tail.
    failure_signals: dict = field(default_factory=dict)

    @property
    def correctness(self) -> bool:
        return self.outcome == EvalOutcome.OK


# ---------------------------------------------------------------------------
# Constraint check
# ---------------------------------------------------------------------------

_CONSTRAINT_OPS = {"<=": _op.le, ">=": _op.ge, "<": _op.lt, ">": _op.gt, "==": _op.eq}


def check_constraints(result: EvalResult, constraints: dict) -> list:
    """Check hard constraints. Returns list of violation strings (empty = ok)."""
    violations = []
    for metric_name, (op_str, threshold) in constraints.items():
        func = _CONSTRAINT_OPS.get(op_str)
        if func is None:
            violations.append(f"{metric_name}: unknown operator '{op_str}'")
            continue
        value = result.metrics.get(metric_name)
        if value is None:
            violations.append(f"{metric_name}: metric missing (required {op_str} {threshold})")
            continue
        if not isinstance(value, (int, float)):
            violations.append(f"{metric_name}: non-numeric value {value!r}")
            continue
        if not func(value, threshold):
            violations.append(f"{metric_name}: {value} violates {op_str} {threshold}")
    return violations


# ---------------------------------------------------------------------------
# Improvement comparison
# ---------------------------------------------------------------------------

def is_improvement(
    current: EvalResult,
    best: EvalResult,
    metric: str = "latency_ms",
    lower_is_better: bool = True,
    threshold: float = 0.0,
) -> bool:
    """Check if current result improves on best.

    threshold is a relative percentage (e.g. 2.0 = needs >2% improvement).
    """
    if not current.correctness:
        return False
    cur_val = current.metrics.get(metric)
    best_val = best.metrics.get(metric)
    if cur_val is None:
        return False
    if best_val is None:
        return True
    if best_val == 0:
        return cur_val < 0 if lower_is_better else cur_val > 0
    if lower_is_better:
        relative_pct = (best_val - cur_val) / abs(best_val) * 100
    else:
        relative_pct = (cur_val - best_val) / abs(best_val) * 100
    return relative_pct > threshold
