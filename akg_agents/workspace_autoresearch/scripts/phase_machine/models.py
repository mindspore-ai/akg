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

"""Authoritative schema for the Progress view of state.json.

Single dataclass owns the field set. Every writer constructs a complete
`Progress` (or applies field deltas via `.apply(**fields)`) so that:
  - Adding a field requires editing one place.
  - Forgetting to set a field in a writer no longer drops it from disk.
  - Readers can stay on `progress.get("X", default)` or move to attribute
    access; both work (`Progress.get` mirrors `dict.get`).

Co-located with state_store.py because they're inseparable: state_store
is the only entry point that turns these objects into JSON on disk.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields, replace
from typing import Any, Optional


@dataclass
class Progress:
    # Identity
    task: str = ""

    # Round counters
    eval_rounds: int = 0
    # max_rounds defaults HIGH (not 0) so a state.json missing
    # this field doesn't trip `eval_rounds >= max_rounds` -> FINISH on the
    # first lookup. Real writers (_baseline_init via workflow.baseline)
    # always set the actual config value; the default only fires for
    # incomplete files. compute_next_phase and compute_resume_phase still
    # call `progress.get("max_rounds", 999)` for dict-era compatibility,
    # which now matches the dataclass field default and is consistent.
    max_rounds: int = 999
    consecutive_failures: int = 0

    # Best kernel measured so far
    best_metric: Optional[float] = None
    best_commit: Optional[str] = None

    # Sticky pytorch baseline (anchors speedup display; pinned by the first
    # baseline_init that captured ref_latency_us, see workflow/seed.py).
    baseline_metric: Optional[float] = None
    baseline_source: Optional[str] = None      # "ref" (only committed value)
    baseline_outcome: Optional[str] = None     # task_config.EvalOutcome value
    # error_source: "ref" | "kernel" | None. Set by run_verify's tagged
    # try/excepts. "ref" => scaffold rejects + user must fix --ref source.
    # "kernel" => normal seed-fail recovery via PLAN. None on success.
    baseline_error_source: Optional[str] = None
    # Per-shape ref timings (us) from the SEED round, sticky alongside
    # baseline_metric. Without this, sticky-baseline rounds (which skip
    # profile_base) had no per-shape ref to compute speedup_vs_ref as a
    # geomean of per-shape ratios — speedup_vs_ref silently flipped from
    # geomean (round 0) to scalar (round 1+).
    baseline_per_shape_us: Optional[list] = None
    # Fingerprint of the config used when baseline_metric was last
    # measured. eval_client invalidates the sticky baseline when this
    # doesn't match the current config — protects against users changing
    # eval.warmup_times / eval.run_times or the case-count for the same
    # task, where the old ref anchor would no longer be comparable.
    # Shape: {"warmup_times": int, "run_times": int, "num_cases": int}.
    baseline_fingerprint: Optional[dict] = None
    seed_metric: Optional[float] = None

    # Plan
    plan_version: int = 0
    next_pid: int = 0

    # Multi-shape detail (single-shape ops keep these absent)
    num_cases: Optional[int] = None
    per_shape_descs: Optional[list] = None

    # Diagnose subagent state
    diagnose_attempts: int = 0
    diagnose_attempts_for_version: Optional[int] = None
    last_diagnose_failure_reason: Optional[str] = None

    # Stop-hook trace
    last_stop_reason: Optional[str] = None
    last_stop_time: Optional[str] = None

    # Auto-stamped by state_store.save_progress when stamp=True
    last_updated: Optional[str] = None

    # ---- dict-compat read API --------------------------------------------
    # Readers do `progress.get("X", default)` everywhere; supplying this
    # method keeps them working. `keys()` / `__iter__` / `__getitem__`
    # cover the rest of the dict surface (`set(progress.keys())`,
    # `for k in progress`, `progress["X"]`).
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:  # `"X" in progress`
        return key in {f.name for f in fields(self)}

    def keys(self):
        return [f.name for f in fields(self)]

    def __iter__(self):
        return iter(self.keys())

    def __getitem__(self, key: str) -> Any:
        if key in {f.name for f in fields(self)}:
            return getattr(self, key)
        raise KeyError(key)

    # ---- mutation -------------------------------------------------------
    def apply(self, **changes: Any) -> "Progress":
        """Return a new Progress with `changes` overlaid. Validates field
        names so a typo becomes TypeError instead of being silently dropped."""
        return replace(self, **changes)

    # ---- (de)serialisation ---------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Progress":
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        kept = {k: v for k, v in data.items() if k in known}
        # Underscore-prefixed keys are sidecar metadata written by the
        # storage layer (e.g. `_txn_id` from the .ar_state transaction
        # marker) — Progress doesn't expose them as typed fields, but
        # they're not "unknown" in the warn-worthy sense; suppress.
        unknown = sorted(
            k for k in set(data) - known - _LEGACY_DROPPED
            if not k.startswith("_"))
        if unknown:
            import sys
            print(f"[Progress.from_dict] dropping unknown fields: {unknown}",
                  file=sys.stderr)
        return cls(**kept)


# State.json keys not represented on the dataclass — suppress the
# unknown-field warning for them on load.
_LEGACY_DROPPED = frozenset({
    "baseline_commit",
    "baseline_correctness",
    "status",
})
