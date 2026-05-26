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

"""Pure helpers for sticky baseline anchor ownership.

The eval pipeline has three callers that care about the PyTorch/ref
anchor:
  - eval_client decides whether sticky baseline can skip ref profiling.
  - workflow.baseline records the round-0 anchor.
  - workflow.round refreshes the anchor when a later round finally has a
    comparable fresh ref measurement.

Keeping the rules here prevents those paths from drifting.
"""

# pylint: disable=missing-class-docstring,missing-function-docstring
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def valid_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0 < value < float("inf")


def valid_per_shape(values: Any) -> Optional[list[float]]:
    if (isinstance(values, list) and values
            and all(valid_metric(v) for v in values)):
        return [float(v) for v in values]
    return None


def current_fingerprint(config: Any, num_cases: Any) -> dict[str, int]:
    """Fingerprint the eval settings that make ref timings comparable."""
    return {
        "warmup_times": int(getattr(config, "warmup_times", 10)),
        "run_times": int(getattr(config, "run_times", 100)),
        "num_cases": int(num_cases or 1),
    }


def fingerprint_mismatch(stored: Any,
                         current: dict[str, int]) -> Optional[dict[str, Any]]:
    """Return changed keys, or None when sticky reuse is allowed.

    Missing/empty fingerprints are tolerated for sticky override so legacy
    progress.json files keep working. Callers that have a fresh ref can
    still re-anchor and write a modern fingerprint.
    """
    if not isinstance(stored, dict) or not stored:
        return None
    mismatch = {
        k: (stored.get(k), current[k])
        for k in current
        if stored.get(k) != current[k]
    }
    return mismatch or None


def exact_fingerprint_match(stored: Any,
                            current: dict[str, int]) -> bool:
    return isinstance(stored, dict) and stored == current


@dataclass(frozen=True)
class StickyOverride:
    metric: float
    per_shape_us: Optional[list[float]]


@dataclass(frozen=True)
class StickyDecision:
    override: Optional[StickyOverride]
    mismatch: Optional[dict[str, Any]] = None


def sticky_override_from_progress(progress: Any,
                                  fingerprint: dict[str, int]
                                  ) -> StickyDecision:
    """Return the sticky baseline override when the stored anchor is valid."""
    if progress is None or progress.get("baseline_source") != "ref":
        return StickyDecision(None)
    metric = progress.get("baseline_metric")
    if not valid_metric(metric):
        return StickyDecision(None)

    mismatch = fingerprint_mismatch(
        progress.get("baseline_fingerprint"), fingerprint)
    if mismatch:
        return StickyDecision(None, mismatch=mismatch)

    return StickyDecision(StickyOverride(
        metric=float(metric),
        per_shape_us=valid_per_shape(progress.get("baseline_per_shape_us")),
    ))


@dataclass(frozen=True)
class AnchorDecision:
    metric: Optional[float]
    source: Optional[str]
    per_shape_us: Optional[list[float]]
    fingerprint: Optional[dict[str, int]]
    reused_existing: bool
    changed: bool
    message: Optional[str] = None


def _anchor_tuple(progress: Any) -> tuple[Any, Any, Any, Any]:
    return (
        progress.get("baseline_metric"),
        progress.get("baseline_source"),
        progress.get("baseline_per_shape_us"),
        progress.get("baseline_fingerprint"),
    )


def _changed(progress: Any, decision: AnchorDecision) -> bool:
    return _anchor_tuple(progress) != (
        decision.metric,
        decision.source,
        decision.per_shape_us,
        decision.fingerprint,
    )


def _ref_from_metrics(metrics: dict[str, Any]) -> Optional[float]:
    value = metrics.get("ref_latency_us")
    return float(value) if valid_metric(value) else None


def _per_shape_from_metrics(metrics: dict[str, Any]) -> Optional[list[float]]:
    return valid_per_shape(metrics.get("per_shape_base_us"))


def resolve_baseline_init_anchor(progress: Any, metrics: dict[str, Any],
                                 config: Any,
                                 seed_metric: Optional[float],
                                 ) -> AnchorDecision:
    """Choose the anchor written by round-0 baseline initialization."""
    ref_metric = _ref_from_metrics(metrics)
    fp = current_fingerprint(config, metrics.get("num_cases") or 1)

    existing_metric = progress.get("baseline_metric")
    existing_source = progress.get("baseline_source")
    existing_fp = progress.get("baseline_fingerprint")

    if valid_metric(existing_metric) and existing_source == "ref":
        if exact_fingerprint_match(existing_fp, fp):
            decision = AnchorDecision(
                metric=float(existing_metric),
                source="ref",
                per_shape_us=valid_per_shape(
                    progress.get("baseline_per_shape_us")),
                fingerprint=existing_fp,
                reused_existing=True,
                changed=False,
                message=(f"sticky baseline = {existing_metric} "
                         f"(fingerprint match; this round's ref="
                         f"{ref_metric} ignored)"),
            )
            return decision
        if ref_metric is not None:
            decision = AnchorDecision(
                metric=ref_metric,
                source="ref",
                per_shape_us=_per_shape_from_metrics(metrics),
                fingerprint=fp,
                reused_existing=False,
                changed=True,
                message=(f"fingerprint changed ({existing_fp} -> {fp}); "
                         f"re-anchoring to fresh ref={ref_metric}"),
            )
            return decision
        decision = AnchorDecision(
            metric=float(existing_metric),
            source="ref",
            per_shape_us=valid_per_shape(
                progress.get("baseline_per_shape_us")),
            fingerprint=existing_fp,
            reused_existing=True,
            changed=False,
            message=(f"WARN: fingerprint changed ({existing_fp} -> {fp}) "
                     f"but no fresh ref this round; keeping stale "
                     f"baseline={existing_metric} with stale fingerprint "
                     f"to avoid misrepresenting the anchor"),
        )
        return decision

    if ref_metric is not None:
        decision = AnchorDecision(
            metric=ref_metric,
            source="ref",
            per_shape_us=_per_shape_from_metrics(metrics),
            fingerprint=fp,
            reused_existing=False,
            changed=True,
            message=f"baseline = ref_latency_us = {ref_metric} (PyTorch reference)",
        )
        return decision

    fallback = float(seed_metric) if valid_metric(seed_metric) else None
    decision = AnchorDecision(
        metric=fallback,
        source="seed_fallback",
        per_shape_us=None,
        fingerprint=None,
        reused_existing=False,
        changed=True,
        message="WARNING: ref_latency_us missing - baseline falls back to seed metric",
    )
    return decision


def refresh_round_anchor(progress: Any, metrics: dict[str, Any],
                         config: Any) -> AnchorDecision:
    """Refresh Progress.baseline_* after a normal optimization round.

    This is deliberately conservative: a ref anchor remains sticky while
    its fingerprint matches. When a mismatch forced eval_client to re-run
    ref profiling and the round returns a fresh ref_latency_us, we
    re-anchor here so future rounds become sticky again.
    """
    ref_metric = _ref_from_metrics(metrics)
    fp = current_fingerprint(config, metrics.get("num_cases") or 1)

    existing_metric = progress.get("baseline_metric")
    existing_source = progress.get("baseline_source")
    existing_per_shape = valid_per_shape(
        progress.get("baseline_per_shape_us"))
    existing_fp = progress.get("baseline_fingerprint")

    if valid_metric(existing_metric) and existing_source == "ref":
        mismatch = fingerprint_mismatch(existing_fp, fp)
        if mismatch and ref_metric is not None:
            decision = AnchorDecision(
                metric=ref_metric,
                source="ref",
                per_shape_us=_per_shape_from_metrics(metrics),
                fingerprint=fp,
                reused_existing=False,
                changed=True,
                message=(f"fingerprint mismatch {mismatch}; refreshed "
                         f"baseline_metric={ref_metric:.2f}us from fresh ref"),
            )
            return decision
        decision = AnchorDecision(
            metric=float(existing_metric),
            source="ref",
            per_shape_us=existing_per_shape,
            fingerprint=existing_fp,
            reused_existing=True,
            changed=False,
            message=None,
        )
        return decision

    if ref_metric is not None:
        decision = AnchorDecision(
            metric=ref_metric,
            source="ref",
            per_shape_us=_per_shape_from_metrics(metrics),
            fingerprint=fp,
            reused_existing=False,
            changed=True,
            message=(f"captured baseline_metric={ref_metric:.2f}us "
                     f"(source=ref)"),
        )
        decision = AnchorDecision(
            metric=decision.metric,
            source=decision.source,
            per_shape_us=decision.per_shape_us,
            fingerprint=decision.fingerprint,
            reused_existing=decision.reused_existing,
            changed=_changed(progress, decision),
            message=decision.message,
        )
        return decision

    decision = AnchorDecision(
        metric=(float(existing_metric) if valid_metric(existing_metric)
                else None),
        source=existing_source,
        per_shape_us=existing_per_shape,
        fingerprint=existing_fp,
        reused_existing=True,
        changed=False,
        message=None,
    )
    return decision
