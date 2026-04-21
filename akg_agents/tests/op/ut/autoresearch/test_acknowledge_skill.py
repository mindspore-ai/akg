# Copyright 2025-2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the acknowledge_skill tool + gate + feedback wiring.

Covers:
  - TurnExecutor._handle_acknowledge_skill schema validation
    (valuable_aspects + kernel_application + applicability=apply|unbind)
  - FeedbackBuilder.record_skill_acknowledgement side effects
    (ack stored / skill unbound on applicability='unbind' — note:
    unbind is NOT terminal, skill stays bindable for future items)
  - ConversationBuffer.inject_backing_skill synthetic tool_result +
    unload_item_reads eviction
"""

from types import SimpleNamespace

import pytest

from akg_agents.op.autoresearch.agent.conversation import ConversationBuffer
from akg_agents.op.autoresearch.agent.skill_builder import SkillBuilder
from akg_agents.op.autoresearch.agent.turn import TurnExecutor


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_VALID_VALUABLE = (
    "This skill teaches a deterministic grid configuration using fixed "
    "core count with an interleaved 1D loop over program_id, keeping "
    "the pattern stable across input shapes and avoiding 2D group "
    "scheduling overhead."
)

_VALID_APPLICATION = (
    "For this item I will restructure the grid from 2D to 1D + inner "
    "loop over (pid, total, CORE_NUM) and rewrite the index arithmetic "
    "to match — a structural change to scheduling, not a parameter "
    "sweep."
)


def _make_executor(active_item: dict | None,
                   skill_builder: SkillBuilder | None = None) -> TurnExecutor:
    """Bare TurnExecutor with only the fields the handler touches."""
    te = TurnExecutor.__new__(TurnExecutor)
    te.config = SimpleNamespace(name="op_test", agent=SimpleNamespace())
    sb = skill_builder if skill_builder is not None else SkillBuilder()
    te.feedback = SimpleNamespace(
        get_active_item=lambda: active_item,
        record_skill_acknowledgement=lambda *a, **kw: False,
        _plan_items=[active_item] if active_item else [],
        skill_builder=sb,
        _plan_version=1,
    )
    return te


class TestAcknowledgeSkillSchema:
    def test_rejects_when_no_active_item(self):
        te = _make_executor(None)
        reply = te._handle_acknowledge_skill({
            "plan_item_id": "p1",
            "valuable_aspects": _VALID_VALUABLE,
            "kernel_application": _VALID_APPLICATION,
            "applicability": "apply",
        })
        assert "rejected" in reply and "no active" in reply

    def test_rejects_mismatched_item_id(self):
        te = _make_executor({"id": "p1", "backing_skill": "matmul"})
        reply = te._handle_acknowledge_skill({
            "plan_item_id": "p99",
            "valuable_aspects": _VALID_VALUABLE,
            "kernel_application": _VALID_APPLICATION,
            "applicability": "apply",
        })
        assert "not the active item" in reply

    def test_rejects_unbound_item(self):
        te = _make_executor({"id": "p1", "backing_skill": None})
        reply = te._handle_acknowledge_skill({
            "plan_item_id": "p1",
            "valuable_aspects": _VALID_VALUABLE,
            "kernel_application": _VALID_APPLICATION,
            "applicability": "apply",
        })
        assert "no backing_skill" in reply

    def test_rejects_bad_applicability(self):
        te = _make_executor({"id": "p1", "backing_skill": "matmul"})
        reply = te._handle_acknowledge_skill({
            "plan_item_id": "p1",
            "valuable_aspects": _VALID_VALUABLE,
            "kernel_application": _VALID_APPLICATION,
            "applicability": "maybe",
        })
        assert "applicability must be" in reply

    def test_rejects_legacy_applicability_values(self):
        """The old enum { apply, partial, not_applicable } is gone.
        'partial' and 'not_applicable' must now be rejected — they
        are silent data shape traps if the agent still emits them."""
        te = _make_executor({"id": "p1", "backing_skill": "matmul"})
        for stale in ("partial", "not_applicable"):
            reply = te._handle_acknowledge_skill({
                "plan_item_id": "p1",
                "valuable_aspects": _VALID_VALUABLE,
                "kernel_application": _VALID_APPLICATION,
                "applicability": stale,
            })
            assert "applicability must be" in reply, (
                f"legacy {stale!r} should be rejected"
            )

    def test_rejects_missing_valuable_aspects(self):
        te = _make_executor({"id": "p1", "backing_skill": "matmul"})
        reply = te._handle_acknowledge_skill({
            "plan_item_id": "p1",
            "kernel_application": _VALID_APPLICATION,
            "applicability": "apply",
        })
        assert "valuable_aspects is required" in reply

    def test_rejects_missing_kernel_application(self):
        te = _make_executor({"id": "p1", "backing_skill": "matmul"})
        reply = te._handle_acknowledge_skill({
            "plan_item_id": "p1",
            "valuable_aspects": _VALID_VALUABLE,
            "applicability": "apply",
        })
        assert "kernel_application is required" in reply

    def test_rejects_too_short_valuable_aspects(self):
        te = _make_executor({"id": "p1", "backing_skill": "matmul"})
        reply = te._handle_acknowledge_skill({
            "plan_item_id": "p1",
            "valuable_aspects": "too short",
            "kernel_application": _VALID_APPLICATION,
            "applicability": "apply",
        })
        assert "valuable_aspects is too short" in reply

    def test_rejects_too_long_kernel_application(self):
        te = _make_executor({"id": "p1", "backing_skill": "matmul"})
        reply = te._handle_acknowledge_skill({
            "plan_item_id": "p1",
            "valuable_aspects": _VALID_VALUABLE,
            "kernel_application": "x" * 600,
            "applicability": "apply",
        })
        assert "kernel_application is too long" in reply

    def test_apply_success_passes_to_feedback(self):
        captured = {}
        active = {"id": "p1", "backing_skill": "matmul"}
        te = _make_executor(active)
        te.feedback.record_skill_acknowledgement = lambda item_id, ack: (
            captured.update({"item_id": item_id, "ack": ack}) or False
        )
        reply = te._handle_acknowledge_skill({
            "plan_item_id": "p1",
            "valuable_aspects": _VALID_VALUABLE,
            "kernel_application": _VALID_APPLICATION,
            "applicability": "apply",
        })
        assert "Acknowledged" in reply
        assert captured["item_id"] == "p1"
        assert captured["ack"]["applicability"] == "apply"
        assert captured["ack"]["skill"] == "matmul"
        assert captured["ack"]["valuable_aspects"] == _VALID_VALUABLE
        assert captured["ack"]["kernel_application"] == _VALID_APPLICATION

    def test_unbind_reply_mentions_skill_stays_available(self):
        """The 'unbind' reply must tell the agent the skill is NOT
        permanently excluded — otherwise the agent may self-censor
        on later items the same way the old 'abandon' wording did."""
        active = {"id": "p1", "backing_skill": "matmul"}
        te = _make_executor(active)
        te.feedback.record_skill_acknowledgement = lambda *a, **kw: True
        reply = te._handle_acknowledge_skill({
            "plan_item_id": "p1",
            "valuable_aspects": _VALID_VALUABLE,
            "kernel_application": _VALID_APPLICATION,
            "applicability": "unbind",
        })
        assert "STAYS available" in reply or "stays available" in reply
        assert "future item" in reply


class TestFeedbackRecordAck:
    def test_apply_stores_ack_without_downgrade(self):
        from akg_agents.op.autoresearch.agent.feedback import FeedbackBuilder
        fb = FeedbackBuilder(config=SimpleNamespace(
            agent=SimpleNamespace(
                plan_item_rationale_max_chars=400,
                plan_item_rationale_min_chars=30,
                skill_keyword_max_per_item=5,
            )
        ))
        fb._plan_items = [{
            "id": "p1", "text": "do x", "status": "active",
            "backing_skill": "matmul", "keywords": [], "rationale": "r",
            "sketch": "",
        }]
        fb._active_item_id = "p1"
        unbound = fb.record_skill_acknowledgement("p1", {
            "skill": "matmul",
            "valuable_aspects": _VALID_VALUABLE,
            "kernel_application": _VALID_APPLICATION,
            "applicability": "apply",
        })
        assert unbound is False
        assert fb._plan_items[0]["backing_skill"] == "matmul"
        assert fb._plan_items[0]["skill_ack"]["applicability"] == "apply"

    def test_unbind_releases_binding_but_skill_stays_available(self):
        """'unbind' clears item.backing_skill + appends
        unbound_at_versions, but the skill remains in the registry
        and is still bindable (just demoted to tier 2)."""
        from akg_agents.op.autoresearch.agent.feedback import FeedbackBuilder
        fb = FeedbackBuilder(config=SimpleNamespace(
            agent=SimpleNamespace(
                plan_item_rationale_max_chars=400,
                plan_item_rationale_min_chars=30,
                skill_keyword_max_per_item=5,
            )
        ))
        fb.skill_builder = SkillBuilder()
        fb.skill_builder.register(
            SimpleNamespace(name="matmul", category="guide", content="c",
                            description="d", metadata={}),
            reason="r", plan_version=1,
        )
        fb._plan_items = [{
            "id": "p1", "text": "do x", "status": "active",
            "backing_skill": "matmul", "keywords": [], "rationale": "r",
            "sketch": "",
        }]
        fb._active_item_id = "p1"
        fb._plan_version = 1
        unbound = fb.record_skill_acknowledgement("p1", {
            "skill": "matmul",
            "valuable_aspects": _VALID_VALUABLE,
            "kernel_application": _VALID_APPLICATION,
            "applicability": "unbind",
        })
        assert unbound is True
        assert fb._plan_items[0]["backing_skill"] is None
        rec = fb.skill_builder.get("matmul")
        assert rec.unbound_at_versions == [1]
        # Still in the registry → still bindable for future items.
        assert "matmul" in fb.skill_builder
        assert rec.tier() == 2  # demoted, not excluded


class TestInjectBackingSkill:
    def test_synthetic_inject_tracked_and_unloaded(self):
        buf = ConversationBuffer()
        ok = buf.inject_backing_skill(
            "p1", "matmul", "skill body text",
            plan_version=1, max_chars=6_000,
        )
        assert ok is True
        msgs = buf.view()
        assert len(msgs) == 1
        # Must be a normal user message (not a raw tool_result) so the
        # Anthropic tool_use/tool_result pairing isn't broken.
        assert msgs[0]["role"] == "user"
        assert ConversationBuffer._SKILL_INJECT_PREFIX in msgs[0]["content"]
        assert "skill body text" in msgs[0]["content"]
        # Settle elides the injected content.
        n = buf.unload_item_reads("p1")
        assert n == 1
        assert buf.view()[0]["content"] == ConversationBuffer._SKILL_READ_ELIDED

    def test_inject_dedup_within_same_version(self):
        buf = ConversationBuffer()
        assert buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000) is True
        assert buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000) is False
        assert len(buf) == 1

    def test_empty_content_is_noop(self):
        buf = ConversationBuffer()
        assert buf.inject_backing_skill("p1", "matmul", "", 1, 6_000) is False
        assert len(buf) == 0

    def test_rebuild_keeps_dedup_when_marker_survives(self):
        """If auto_compact carries recent rounds forward, the inject
        marker is still in _msgs after rebuild, so re-injecting the
        same (plan_version, item, skill) must remain a no-op —
        otherwise the body would be duplicated in the buffer."""
        buf = ConversationBuffer()
        buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000)
        buf.on_buffer_rebuilt()
        assert buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000) is False
        # Still only one inject message in the buffer.
        assert sum(
            1 for m in buf.view()
            if isinstance(m.get("content"), str)
            and m["content"].startswith(ConversationBuffer._SKILL_INJECT_PREFIX)
        ) == 1

    def test_rebuild_releases_dedup_when_marker_compacted_out(self):
        """If the rebuild dropped the inject (e.g. summarized away),
        re-injecting must succeed so the active item's skill is
        re-populated in the fresh buffer."""
        buf = ConversationBuffer()
        buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000)
        # Simulate compact replacing messages with a summary-only buffer
        # — the inject marker is gone.
        buf.replace([{"role": "user", "content": "[SUMMARY] rounds compacted"}])
        buf.on_buffer_rebuilt()
        assert buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000) is True

    def test_truncation_applies(self):
        buf = ConversationBuffer()
        buf.inject_backing_skill("p1", "matmul", "x" * 1000, 1, max_chars=50)
        body = buf.view()[0]["content"]
        assert "...[truncated]" in body


# -- Pre-edit gate BLOCKED message cites the current ack schema --------------


class TestAckGateBlockedMessage:
    """When ``edit`` is blocked on a skill-bound item, the reply must
    instruct the agent with the CURRENT ack schema (valuable_aspects +
    kernel_application + applicability=apply|unbind), not the old
    summary + key_points + applicability={apply,partial,
    not_applicable} one. Otherwise the agent follows the message,
    emits a legacy-shaped ack, gets schema-rejected, and loops."""

    def _make_gate_executor(self):
        """Minimal TurnExecutor that reaches the ack-gate branch in
        ``_dispatch_tools``. We don't run the whole turn — we call
        ``_dispatch_tools`` directly with an ``edit`` tool call on a
        backing-skill-bound item that has no skill_ack yet."""
        import asyncio
        from akg_agents.op.autoresearch.agent.turn import TurnExecutor

        active = {
            "id": "p1", "status": "active", "text": "do x",
            "backing_skill": "matmul", "skill_ack": None,
            "keywords": [], "rationale": "r",
        }
        te = TurnExecutor.__new__(TurnExecutor)
        te.config = SimpleNamespace(
            name="op_test", dsl="", backend="", framework="",
            editable_files=["kernel.py"],
            agent=SimpleNamespace(
                max_no_edit_turns=3,
                cumulative_diff_truncate=4000,
                raw_output_tail=2048,
                smoke_output_limit=1000,
            ),
            primary_metric="latency_us",
            lower_is_better=True,
        )
        te.feedback = SimpleNamespace(
            get_active_item=lambda: active,
            validate_edit=lambda _id: (True, ""),
            _plan_items=[active],
            _active_item_id="p1",
            phase="active",
            must_replan=False,
            skill_builder=SkillBuilder(),
            plan_version=1,
        )
        te.verbose = False
        te._files = SimpleNamespace(
            is_stale=lambda path: False,
            mark_stale=lambda path: None,
            mark_fresh=lambda path: None,
            mark_touched=lambda path, tool_use_id: None,
        )
        te._tool_handlers = {
            "edit": lambda **kw: SimpleNamespace(ok=True, message="OK"),
        }
        te.task_dir = "."
        return te, active

    def test_blocked_message_instructs_current_ack_schema(self):
        import asyncio
        te, _active = self._make_gate_executor()
        tool_calls = [{
            "tool_name": "edit",
            "tool_use_id": "tu_1",
            "arguments": {
                "path": "kernel.py", "mode": "exact",
                "old_str": "a", "new_str": "b",
                "plan_item_id": "p1",
            },
        }]
        buffer = SimpleNamespace(append=lambda _m: None)
        counters = SimpleNamespace(
            total_api_calls=0, eval_calls_made=0,
            record_edit_attempt=lambda *, ok: None,
        )
        # Use a throwaway event loop rather than asyncio.run() — the
        # latter closes the default loop, and other tests in this
        # module use asyncio.get_event_loop() from sync code, which
        # would fail on a closed default loop.
        loop = asyncio.new_event_loop()
        try:
            results, edits, log, has_finish = loop.run_until_complete(
                te._dispatch_tools(
                    tool_calls, buffer, counters=counters,
                    eval_calls_made=0, max_rounds=10,
                )
            )
        finally:
            loop.close()
        assert len(results) == 1
        msg = results[0]["content"]
        assert "BLOCKED edit" in msg
        # Current schema tokens MUST appear:
        assert "valuable_aspects" in msg
        assert "kernel_application" in msg
        assert "apply" in msg and "unbind" in msg
        # Legacy-schema tokens MUST NOT appear — leaving them in the
        # reply teaches the agent to emit a call that the handler
        # will unconditionally reject.
        assert "summary" not in msg
        assert "key_points" not in msg
        assert "not_applicable" not in msg
        assert "partial" not in msg
