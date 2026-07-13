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

"""Pure policy: how to size / skip one eval's profile run.

``akg_eval`` (the bridge) measures the free per-shape walls during verify and
reads the committed baseline anchor; this module turns those facts into the
``profile_settings`` ``KernelVerifier`` honours (``run_times`` /
``skip_base_profile`` / ``override_base_section``) plus a ``too_slow`` verdict.
No I/O — every input is a plain value, so the precedence rules are unit-testable
in isolation (mirrors :mod:`utils.baseline_anchor`).

Base-handling precedence (highest first):
  1. sticky committed baseline    -> reuse it (override + skip), run_times-free
  2. reference too slow for budget -> skip base (no baseline this round)
  3. otherwise                     -> profile the reference normally
``run_times`` is then sized to the slowest side *actually profiled*, so a
skipped slow reference never throttles the kernel's own measurement.
The kernel itself being too slow -> ``too_slow`` (caller -> kernel_fail,
deliberately never ``inf``, which ``_make_ok_payload`` collapses to infra_fail).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ProfilePlan:
    """Profile sizing decision. ``settings`` extends run_profile's
    ``profile_settings``; ``too_slow`` non-empty -> caller returns
    ``kernel_fail``."""
    settings: Dict[str, Any]
    too_slow: Optional[str] = None


def plan_profile(
    *,
    ref_walls: List[float],
    impl_walls: List[float],
    eval_timeout: float,
    warmup: int,
    repeats: int,
    sticky_section: Optional[Dict[str, Any]] = None,
    base_only: bool = False,
) -> ProfilePlan:
    """Decide profile sizing + base handling from per-shape walls (us).

    ``sticky_section`` is the committed baseline as a Section dict
    (``{avg_us, per_case_us, method}``) when a fingerprint-matched anchor
    exists, else ``None``. ``base_only`` (ref profiling after a failed
    verify) never reuses/skips the base — that run exists to measure it.
    Empty plan when no walls present (pre-change sidecar / sync-less
    backend) — behaviour unchanged.
    """
    budget_s = float(eval_timeout or 0)
    if budget_s <= 0 or not (ref_walls or impl_walls):
        return ProfilePlan({})

    def _fit(est_us: float) -> int:
        # Largest run_times with (1 + warmup + run_times) * est <= budget.
        return math.floor(budget_s / (est_us / 1e6)) - (1 + warmup)

    max_impl = max(impl_walls) if (impl_walls and not base_only) else 0.0
    if max_impl and _fit(max_impl) < 1:
        return ProfilePlan({}, too_slow=(
            f"kernel too slow to profile within the per-shape budget "
            f"({max_impl / 1e6:.1f}s/call > eval_timeout {budget_s:.0f}s); "
            f"optimise the kernel"))

    settings: Dict[str, Any] = {}
    max_ref = max(ref_walls) if ref_walls else 0.0
    base_profiled = True  # base_only profiles the ref; only skip flips this off

    if not base_only:
        if sticky_section is not None:
            # (1) reuse the committed baseline; ref not measured this round.
            settings["override_base_section"] = sticky_section
            settings["skip_base_profile"] = True
            base_profiled = False
        elif max_ref and _fit(max_ref) < 1:
            # (2) ref alone overflows the budget -> no baseline this round.
            settings["skip_base_profile"] = True
            base_profiled = False

    # (3) size run_times to the slowest side actually profiled, so a skipped
    # slow ref never throttles the kernel's own measurement.
    binding = max(max_impl, max_ref if base_profiled else 0.0)
    if binding:
        run_eff = max(1, min(repeats, _fit(binding)))
        if run_eff < repeats:
            settings["run_times"] = run_eff
    return ProfilePlan(settings)
