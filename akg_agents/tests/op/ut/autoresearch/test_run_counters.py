# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
Tests for RunCounters — single source of truth for AgentLoop's per-run
counter state (P3 refactor).

These tests pin down the semantic rules that used to live as one-line
comments in loop.py / turn.py:

- KEEP zeroes BOTH failures and no_improvement
- DISCARD zeroes failures and bumps no_improvement
- FAIL bumps failures but is intentionally NEUTRAL on no_improvement
- record_edit_attempt(False) bumps failures (non-eval failure path)
- reset_after_diagnose() / reset_after_hint_replan() are scoped resets
- advance_after_failed_hint() bumps no_improvement past the trigger
- legacy session.json format (top-level fields) still loads via from_dict
"""

from akg_agents.op.autoresearch.agent.counters import RunCounters


# ---------------------------------------------------------------------------
# record_eval — the three-way semantic rules
# ---------------------------------------------------------------------------


class TestRecordEval:
    def test_eval_keep_zeroes_failures_and_no_improvement(self):
        c = RunCounters(consecutive_failures=4, consecutive_no_improvement=7)
        c.record_eval("eval_keep")
        assert c.consecutive_failures == 0
        assert c.consecutive_no_improvement == 0
        assert c.eval_calls_made == 1

    def test_eval_discard_zeroes_failures_bumps_no_improvement(self):
        c = RunCounters(consecutive_failures=2, consecutive_no_improvement=3)
        c.record_eval("eval_discard")
        assert c.consecutive_failures == 0
        assert c.consecutive_no_improvement == 4
        assert c.eval_calls_made == 1

    def test_eval_fail_bumps_failures_neutral_on_no_improvement(self):
        """The critical 'FAIL doesn't count as no_improvement' rule.

        Before P3 this was a one-line comment in loop.py — easy to break
        accidentally. Now it's encoded in record_eval and pinned here.
        """
        c = RunCounters(consecutive_failures=1, consecutive_no_improvement=4)
        c.record_eval("eval_fail")
        assert c.consecutive_failures == 2
        assert c.consecutive_no_improvement == 4  # unchanged
        assert c.eval_calls_made == 1

    def test_unknown_outcome_only_bumps_eval_count(self):
        """Defensive: unknown status string just bumps eval_calls_made."""
        c = RunCounters(consecutive_failures=2, consecutive_no_improvement=3)
        c.record_eval("eval_weird")
        assert c.consecutive_failures == 2
        assert c.consecutive_no_improvement == 3
        assert c.eval_calls_made == 1

    def test_discard_then_fail_keeps_no_improvement_climbing(self):
        """Documents the 'no_improvement is cumulative across FAILs' rule:
        DISCARD -> FAIL -> DISCARD should yield no_improvement=2, not
        reset by the intermediate FAIL."""
        c = RunCounters()
        c.record_eval("eval_discard")
        assert c.consecutive_no_improvement == 1
        c.record_eval("eval_fail")
        assert c.consecutive_no_improvement == 1  # FAIL didn't reset
        c.record_eval("eval_discard")
        assert c.consecutive_no_improvement == 2

    def test_keep_after_failure_streak_clears_both(self):
        c = RunCounters()
        for _ in range(3):
            c.record_eval("eval_fail")
        assert c.consecutive_failures == 3
        c.record_eval("eval_keep")
        assert c.consecutive_failures == 0
        assert c.consecutive_no_improvement == 0


# ---------------------------------------------------------------------------
# Non-eval edit failure path (atomic rollback / quick_check)
# ---------------------------------------------------------------------------


class TestRecordEditAttempt:
    def test_edit_attempt_failure_bumps_consecutive_failures(self):
        c = RunCounters(consecutive_failures=2)
        c.record_edit_attempt(ok=False)
        assert c.consecutive_failures == 3
        # eval_calls_made not touched — this is the non-eval failure path
        assert c.eval_calls_made == 0

    def test_edit_attempt_success_is_noop(self):
        c = RunCounters(consecutive_failures=2)
        c.record_edit_attempt(ok=True)
        assert c.consecutive_failures == 2


# ---------------------------------------------------------------------------
# Turn / tool / API counters
# ---------------------------------------------------------------------------


class TestTurnAndToolCounters:
    def test_record_turn_with_no_edits_bumps(self):
        c = RunCounters()
        c.record_turn_with_no_edits()
        c.record_turn_with_no_edits()
        assert c.consecutive_no_edit_turns == 2

    def test_record_turn_with_edits_resets(self):
        c = RunCounters(consecutive_no_edit_turns=3)
        c.record_turn_with_edits()
        assert c.consecutive_no_edit_turns == 0

    def test_no_tool_use_then_tool_use(self):
        c = RunCounters()
        c.record_no_tool_use()
        c.record_no_tool_use()
        assert c.consecutive_no_tool_turns == 2
        c.record_tool_use()
        assert c.consecutive_no_tool_turns == 0

    def test_record_api_call_increments(self):
        c = RunCounters()
        c.record_api_call()
        c.record_api_call()
        assert c.total_api_calls == 2


# ---------------------------------------------------------------------------
# Reset methods (used by diagnose / hint_replan paths)
# ---------------------------------------------------------------------------


class TestResetMethods:
    def test_reset_after_diagnose_zeroes_failures_only(self):
        c = RunCounters(consecutive_failures=5, consecutive_no_improvement=3)
        c.reset_after_diagnose()
        assert c.consecutive_failures == 0
        assert c.consecutive_no_improvement == 3  # untouched


# ---------------------------------------------------------------------------
# Compaction failure tracking
# ---------------------------------------------------------------------------


class TestCompactCounters:
    def test_record_compact_failure_increments(self):
        c = RunCounters()
        c.record_compact_failure()
        c.record_compact_failure()
        assert c.compact_failures == 2

    def test_record_compact_success_resets(self):
        c = RunCounters(compact_failures=2)
        c.record_compact_success()
        assert c.compact_failures == 0


# ---------------------------------------------------------------------------
# Supervisor retry tracking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Persistence — to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_to_dict_round_trip(self):
        c = RunCounters(
            consecutive_failures=2,
            consecutive_no_edit_turns=1,
            consecutive_no_improvement=4,
            eval_calls_made=10,
            total_api_calls=15,
        )
        d = c.to_dict()
        c2 = RunCounters.from_dict(d)
        assert c2 == c

    def test_from_dict_drops_unknown_keys(self):
        """Legacy supervisor_retries / hint_retries silently ignored —
        the supervisor mechanism has been removed."""
        d = {
            "consecutive_failures": 3,
            "supervisor_retries": {"p1": 5},
            "hint_retries": {"p1": 999},
        }
        c = RunCounters.from_dict(d)
        assert c.consecutive_failures == 3
        assert not hasattr(c, "supervisor_retries")

    def test_from_dict_empty(self):
        c = RunCounters.from_dict({})
        assert c == RunCounters()

    def test_from_dict_none(self):
        c = RunCounters.from_dict(None)
        assert c == RunCounters()

    def test_from_dict_legacy_top_level_format(self):
        """Pre-P3 sessions had counters as top-level session.json keys
        (alongside baseline_commit, plan_state, etc). RunCounters.from_dict
        must accept that layout for backward compatibility."""
        legacy_session = {
            "consecutive_failures": 3,
            "consecutive_no_improvement": 7,
            "eval_calls_made": 12,
            "total_api_calls": 20,
            # Legacy session also had unrelated top-level fields:
            "baseline_commit": "abc123",
            "plan_state": {"items": []},
            "saved_at": "2026-01-01",
        }
        c = RunCounters.from_dict(legacy_session)
        assert c.consecutive_failures == 3
        assert c.consecutive_no_improvement == 7
        assert c.eval_calls_made == 12
        assert c.total_api_calls == 20
        # Unrelated fields are silently ignored — no AttributeError
        assert not hasattr(c, "baseline_commit")
