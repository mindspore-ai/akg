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

"""
RunCounters — Single source of truth for all per-run counters that drive
AgentLoop's gating decisions.

Before P3 the counters were scattered across loop.py and turn.py:
``_consecutive_failures`` was incremented in turn.py's settle path but
reset in loop.py's diagnose path; ``_consecutive_no_improvement`` had
its semantic rule (FAIL doesn't count) hidden in a one-line comment;
the per-item retry dict was a bare attribute with no encapsulation.
To trace the lifetime of any counter you had to hop between files.

This module collects them all into one dataclass. Mutations go through
``record_*`` / ``reset_after_*`` methods so the semantic rules are
encoded in code (not in comments) and unit-testable in isolation:

- ``record_eval('eval_keep')``    — KEEP resets failures + no_improvement
- ``record_eval('eval_discard')`` — DISCARD resets failures, bumps no_improvement
- ``record_eval('eval_fail')``    — FAIL bumps failures, leaves no_improvement alone
- ``record_edit_attempt(False)``  — non-eval edit failure (rollback / quick_check)
- ``reset_after_diagnose()``      — diagnose path zeroes failures

Persistence: ``to_dict()`` / ``from_dict()``. ``from_dict`` accepts both
the new top-level ``counters`` schema AND the legacy session.json format
where each counter was a top-level field, so resuming an old run still
works. Legacy ``hint_retries`` field is one-shot migrated to
``supervisor_retries`` on load.
"""

from dataclasses import dataclass, field, fields, asdict


@dataclass
class RunCounters:
    """All per-run counters that gate AgentLoop's decisions.

    Counters are mutated exclusively via ``record_*`` / ``reset_*``
    methods so the semantic rules live in code, not comments.
    """

    # Eval / failure / improvement gating
    consecutive_failures: int = 0
    consecutive_no_edit_turns: int = 0
    consecutive_no_improvement: int = 0
    consecutive_no_tool_turns: int = 0

    # Round budgets
    eval_calls_made: int = 0
    total_api_calls: int = 0
    total_keeps: int = 0

    # Compaction failure escalation (PTL recovery)
    compact_failures: int = 0

    # ===== Eval recording (3-way: KEEP / DISCARD / FAIL) =====================

    def record_eval(self, outcome: str) -> None:
        """Record one eval result. Outcome is one of:
            ``eval_keep``    — accepted (correct + improved + constraints met)
            ``eval_discard`` — correct but no improvement
            ``eval_fail``    — broken (correctness/constraint/infrastructure)

        Each branch encodes one of the three semantic rules:
          - KEEP zeroes both failures and no_improvement (clean slate)
          - DISCARD zeroes failures but bumps no_improvement (we tried
            something correct but unhelpful — DON'T count it as failure,
            DO count it against the stagnation budget)
          - FAIL bumps failures but leaves no_improvement alone (a
            broken edit should not push us toward "stuck" hint replan
            prematurely — that's what diagnose is for)
        """
        self.eval_calls_made += 1
        if outcome == "eval_keep":
            self.total_keeps += 1
            self.consecutive_failures = 0
            self.consecutive_no_improvement = 0
        elif outcome == "eval_discard":
            self.consecutive_failures = 0
            self.consecutive_no_improvement += 1
        elif outcome == "eval_fail":
            self.consecutive_failures += 1
            # FAIL is intentionally neutral on no_improvement.

    def record_edit_attempt(self, ok: bool) -> None:
        """Record a non-eval edit failure (atomic rollback / quick_check).

        These never reach ``record_eval``, so failures from edit
        validation or quick_check are tracked here. ``ok=True`` is a
        no-op — successful edits are tracked via ``record_eval`` once
        they actually run.
        """
        if not ok:
            self.consecutive_failures += 1

    # ===== Edit / tool turn tracking =========================================

    def record_turn_with_no_edits(self) -> None:
        self.consecutive_no_edit_turns += 1

    def record_turn_with_edits(self) -> None:
        self.consecutive_no_edit_turns = 0

    def record_no_tool_use(self) -> None:
        self.consecutive_no_tool_turns += 1

    def record_tool_use(self) -> None:
        self.consecutive_no_tool_turns = 0

    def record_api_call(self) -> None:
        self.total_api_calls += 1

    # ===== Reset after coarser actions =======================================

    def reset_after_diagnose(self) -> None:
        """Diagnose forced a hard direction change — zero failures so the
        agent can attempt the new plan with a clean slate."""
        self.consecutive_failures = 0

    # ===== Compaction failure tracking =======================================

    def record_compact_success(self) -> None:
        self.compact_failures = 0

    def record_compact_failure(self) -> None:
        self.compact_failures += 1

    # ===== Persistence =======================================================

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for session.json)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RunCounters":
        """Deserialize from a dict.

        Unknown keys (including legacy supervisor_retries / hint_retries)
        are silently dropped — the supervisor mechanism has been removed.
        """
        if not d:
            return cls()
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in valid}
        return cls(**kwargs)
