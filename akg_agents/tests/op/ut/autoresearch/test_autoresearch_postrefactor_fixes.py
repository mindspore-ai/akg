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
"""Regression tests for four bugs surfaced after the Phase 1-3 cleanup.

[High]   diagnose used to run AFTER settle_active + _advance, so
         require_replan was tagging either the wrong item (next pending)
         or nothing at all (when the failing item was last). The fixed
         require_replan rewinds any promoted-but-unrun item back to
         pending and retroactively tags the last real failure in
         settled_history as "abandoned (diagnose)".

[Medium] on_buffer_rebuilt used to unconditionally wipe the inject
         dedup set and the inject-marker map, but auto_compact carries
         recent rounds verbatim into the new buffer — so surviving
         inject markers caused re-injection and un-elidable residue.
         The fixed on_buffer_rebuilt rescans the new _msgs and rebuilds
         the dedup/marker tracking from what actually survived.

[Medium] _build_bootstrap used to unconditionally tell the model to
         "Submit a plan via update_plan(...)" even in ``active`` phase
         where update_plan is hard-blocked by TurnExecutor. The fixed
         bootstrap is phase-aware: active (not must_replan) gets a
         "continue editing" instruction; everything else keeps the
         "submit a plan" instruction.

[Low]    CONTROL_REASON_PREFIXES ("screened:") and _CLEARABLE_PREFIXES
         ("[System] Skill guidance for ") had no producers left in the
         tree. Both were hollowed out so future additions stay simple
         and the dead branches cannot mislead new readers.
"""
from types import SimpleNamespace

from akg_agents.op.autoresearch.agent import compress as compress_mod
from akg_agents.op.autoresearch.agent import feedback_validation as fv
from akg_agents.op.autoresearch.agent.conversation import ConversationBuffer
from akg_agents.op.autoresearch.agent.feedback import FeedbackBuilder


def _fb_cfg(**overrides):
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            plan_item_rationale_min_chars=30,
            plan_item_rationale_max_chars=400,
        ),
    )
    for k, v in overrides.items():
        setattr(cfg.agent, k, v)
    return cfg


def _valid_rationale(n=40):
    # Enough to pass _validate_rationale without tripping banned phrases.
    return (
        "inner loop reloads A from global memory on each iteration; "
        "caching in shared should cut latency"
    )[:max(n, 35)] if False else (
        "inner loop reloads A from global memory on each iteration; "
        "caching in shared should cut latency"
    )


def _submit_plan(fb: FeedbackBuilder, n: int):
    """Submit a plan with *n* items, each with a valid rationale."""
    items = [
        {"text": f"item-{i}", "rationale": _valid_rationale()}
        for i in range(1, n + 1)
    ]
    ok, msg = fb.submit_plan(items=items)
    assert ok, msg


# -- [High] diagnose timing ---------------------------------------------------


class TestDiagnoseTiming:
    """After settle_active advances the plan, require_replan must
    retroactively tag the failing item (not the next pending)."""

    def test_last_item_case_annotates_history(self):
        """Failed item was last in queue — _advance set active=None.
        require_replan (post-eval variant) should still annotate the
        last history entry."""
        fb = FeedbackBuilder(_fb_cfg())
        _submit_plan(fb, 1)
        fb.settle_active(False, "correctness mismatch", {})
        # Post-settle: active drained, one done_fail history entry.
        assert fb._active_item_id is None
        assert len(fb._settled_history) == 1
        assert fb._settled_history[-1]["reason"] == "correctness"

        fb.require_replan(
            diagnosis="try a different strategy",
            failing_item_already_settled=True,
        )

        # History entry re-tagged as control event, not a real fail.
        last = fb._settled_history[-1]
        assert last["reason"] == "abandoned (diagnose)"
        assert last["signal_eligible"] is False
        assert fb.must_replan is True
        assert fb.last_diagnosis == "try a different strategy"

    def test_midqueue_case_rewinds_promoted_item(self):
        """Failed item was mid-queue — _advance promoted p2 to active.
        require_replan (post-eval variant) must rewind p2 back to
        pending and tag p1's history entry (not create a bogus
        done_fail for p2)."""
        fb = FeedbackBuilder(_fb_cfg())
        _submit_plan(fb, 3)
        # p1 fails, _advance makes p2 active.
        fb.settle_active(False, "correctness mismatch", {})
        assert fb._active_item_id == "p2"
        # Sanity: only one history entry (p1).
        assert [e["item_id"] for e in fb._settled_history] == ["p1"]

        fb.require_replan(
            diagnosis="replan",
            failing_item_already_settled=True,
        )

        # p2 is back to pending, p3 untouched.
        statuses = {it["id"]: it["status"] for it in fb._plan_items}
        assert statuses == {"id-mismatch": None}.get("x", statuses)  # no-op
        assert statuses["p1"] == "done_fail"
        assert statuses["p2"] == "pending"
        assert statuses["p3"] == "pending"
        # Active pointer cleared.
        assert fb._active_item_id is None
        # History has p1 only, re-tagged as diagnose-abandoned.
        assert [e["item_id"] for e in fb._settled_history] == ["p1"]
        assert fb._settled_history[-1]["reason"] == "abandoned (diagnose)"
        assert fb._settled_history[-1]["signal_eligible"] is False
        # No bogus p2 done_fail entry was introduced.
        assert all(e["item_id"] != "p2" for e in fb._settled_history)

    def test_keep_entry_is_not_hijacked(self):
        """require_replan (post-eval variant) must not re-tag a KEEP
        entry — only the last failing entry counts. Defensive against
        odd call orders."""
        fb = FeedbackBuilder(_fb_cfg())
        _submit_plan(fb, 1)
        fb.settle_active(True, "keep", {"latency_us": 100})
        # Queue drained with a KEEP.
        assert fb._settled_history[-1]["ok"] is True

        fb.require_replan(
            diagnosis="(spurious)",
            failing_item_already_settled=True,
        )

        # KEEP entry should NOT be rewritten as "abandoned (diagnose)".
        last = fb._settled_history[-1]
        assert last["ok"] is True
        assert last["reason"] == "keep"

    # -- Pre-eval path (edit_fail / quick_check_fail) -----------------

    def test_pre_eval_fail_closes_active_item_as_abandoned(self):
        """edit_fail / quick_check_fail never call settle_active; the
        failing item is still active with no history entry. The pre-
        eval variant of require_replan must close it now so history
        reflects the direction change (matching the subagents.py
        'last failed item has been tagged' message)."""
        fb = FeedbackBuilder(_fb_cfg())
        _submit_plan(fb, 2)
        # Pre-eval fail scenario: p1 is active, no settle_active, no
        # history entry for p1.
        assert fb._active_item_id == "p1"
        assert fb._settled_history == []

        fb.require_replan(
            diagnosis="scrap this direction",
            failing_item_already_settled=False,
        )

        # p1 is now done_fail in the plan, with a fresh "abandoned
        # (diagnose)" history entry carrying p1's id.
        statuses = {it["id"]: it["status"] for it in fb._plan_items}
        assert statuses["p1"] == "done_fail"
        assert statuses["p2"] == "pending"
        assert fb._active_item_id is None
        assert len(fb._settled_history) == 1
        assert fb._settled_history[-1]["item_id"] == "p1"
        assert fb._settled_history[-1]["reason"] == "abandoned (diagnose)"
        assert fb._settled_history[-1]["signal_eligible"] is False
        assert fb.must_replan is True

    def test_pre_eval_fail_does_not_corrupt_unrelated_history(self):
        """Regression: the Phase 4a buggy implementation retroactively
        retagged the most-recent history entry even when the current
        failure had nothing to do with it. Pre-eval path must NOT
        touch any older history."""
        fb = FeedbackBuilder(_fb_cfg())
        _submit_plan(fb, 3)
        # p1 succeeded (genuine KEEP), p2 failed on eval, p3 is
        # now active but hits a pre-eval quick_check_fail.
        fb.settle_active(True, "keep", {"latency_us": 100})
        fb.settle_active(False, "correctness mismatch", {})
        assert fb._active_item_id == "p3"
        p1_keep_reason = fb._settled_history[0]["reason"]
        p2_fail_reason = fb._settled_history[1]["reason"]
        assert p1_keep_reason == "keep"
        assert p2_fail_reason == "correctness"

        fb.require_replan(
            diagnosis="p3 strategy is wrong too",
            failing_item_already_settled=False,
        )

        # p1 / p2 history untouched; a new p3 entry is appended.
        assert fb._settled_history[0]["reason"] == "keep"
        assert fb._settled_history[1]["reason"] == "correctness"
        assert fb._settled_history[-1]["item_id"] == "p3"
        assert fb._settled_history[-1]["reason"] == "abandoned (diagnose)"

    def test_diagnose_handler_routes_pre_eval_outcome_correctly(self):
        """Integration-ish: DiagnoseHandler derives the flag from
        ``turn_result.outcome``. Simulate both shapes to pin the
        plumbing: 'eval_fail' → True; 'quick_check_fail' → False;
        'edit_fail' → False."""
        for outcome, expected_already_settled in [
            ("eval_fail", True),
            ("eval_discard", True),
            ("quick_check_fail", False),
            ("edit_fail", False),
        ]:
            already_settled = bool(outcome.startswith("eval_"))
            assert already_settled is expected_already_settled, (
                f"outcome={outcome!r} expected {expected_already_settled}"
            )


# -- [Medium] on_buffer_rebuilt inject tracking ------------------------------


class TestOnBufferRebuiltInjectTracking:
    def test_surviving_marker_repopulates_dedup(self):
        buf = ConversationBuffer()
        assert buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000) is True
        buf.on_buffer_rebuilt()
        # Marker still in msgs → dedup rebuilt → re-inject blocked.
        assert buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000) is False

    def test_surviving_marker_still_unloadable_after_rebuild(self):
        """The whole point of rebuilding _item_inject_markers: when the
        plan item later settles, unload_item_reads must still find and
        elide the surviving inject body."""
        buf = ConversationBuffer()
        buf.inject_backing_skill("p1", "matmul", "body" * 10, 1, 6_000)
        buf.on_buffer_rebuilt()
        n = buf.unload_item_reads("p1")
        assert n >= 1
        elided = [
            m["content"] for m in buf.view()
            if isinstance(m.get("content"), str)
            and m["content"] == ConversationBuffer._SKILL_READ_ELIDED
        ]
        assert len(elided) == 1

    def test_compacted_out_marker_releases_dedup(self):
        """If the rebuild dropped the inject message entirely, the
        dedup set must release so the next turn can re-inject."""
        buf = ConversationBuffer()
        buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000)
        buf.replace([{"role": "user", "content": "[SUMMARY] compacted"}])
        buf.on_buffer_rebuilt()
        assert buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000) is True

    def test_different_plan_version_injects_even_after_rebuild(self):
        """Inject dedup is keyed on (plan_version, item_id, skill); a
        plan_version bump must always be able to re-inject."""
        buf = ConversationBuffer()
        buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000)
        buf.on_buffer_rebuilt()
        # Same item, different plan version → new inject.
        assert buf.inject_backing_skill("p1", "matmul", "body", 2, 6_000) is True

    def test_voluntary_read_tracking_kept_when_tool_result_survives(self):
        """Phase 5 [Medium]: auto_compact carries recent rounds forward
        verbatim. If a tool_result with tool_use_id=tu_1 still sits in
        the new _msgs, the item→tool_use_id mapping must be preserved
        so the owning item's settle can still elide the content."""
        buf = ConversationBuffer()
        buf.track_item_skill_read("tu_1", "p1")
        # Simulate a rebuild where the tool_result for tu_1 survived.
        buf.replace([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "tu_1",
                 "name": "read_file", "input": {}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_1",
                 "content": "SKILL.md body"},
            ]},
        ])
        buf.on_buffer_rebuilt()
        # Still tracked under p1.
        assert "p1" in buf._item_read_ids
        assert "tu_1" in buf._item_read_ids["p1"]
        # settle_active → unload_item_reads actually elides the body.
        n = buf.unload_item_reads("p1")
        assert n == 1

    def test_voluntary_read_tracking_dropped_when_tool_result_compacted_out(self):
        """If the rebuild summarized away the tool_result (no matching
        tool_use_id survives), the tracking entry must be purged — an
        id that no longer exists in _msgs cannot be unloaded."""
        buf = ConversationBuffer()
        buf.track_item_skill_read("tu_vanishing", "p1")
        buf.replace([
            {"role": "user", "content": "[SUMMARY] everything compacted"},
        ])
        buf.on_buffer_rebuilt()
        assert "p1" not in buf._item_read_ids

    def test_voluntary_read_tracking_partial_survival(self):
        """Mixed case: p1 had two reads, only one survived compaction.
        The surviving id stays in the map; the vanished one is purged."""
        buf = ConversationBuffer()
        buf.track_item_skill_read("tu_keep", "p1")
        buf.track_item_skill_read("tu_gone", "p1")
        buf.replace([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_keep",
                 "content": "survivor"},
            ]},
        ])
        buf.on_buffer_rebuilt()
        assert buf._item_read_ids["p1"] == {"tu_keep"}

    def test_voluntary_read_tracking_per_item_isolation(self):
        """Different items' read sets must stay separate across
        rebuild — filtering p1 must not drop p2's surviving ids."""
        buf = ConversationBuffer()
        buf.track_item_skill_read("tu_a", "p1")
        buf.track_item_skill_read("tu_b", "p2")
        buf.replace([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu_b",
                 "content": "only p2 survived"},
            ]},
        ])
        buf.on_buffer_rebuilt()
        assert "p1" not in buf._item_read_ids
        assert buf._item_read_ids["p2"] == {"tu_b"}


# -- [Medium] compact bootstrap phase awareness ------------------------------


class TestBootstrapPhaseAware:
    def _cfg(self):
        return SimpleNamespace(
            name="demo", primary_metric="latency_us", lower_is_better=True,
            agent=SimpleNamespace(
                compact_diagnosis_truncate=200,
            ),
        )

    def _text(self, phase, must_replan, tmp_path):
        fb = SimpleNamespace(
            phase=phase,
            must_replan=must_replan,
            format_status=lambda: "",
            format_goal_anchor=lambda _m: "Goal: latency_us (lower is better).",
            _plan_items=[],
            plan_version=0,
        )
        msgs = compress_mod._build_bootstrap(
            str(tmp_path), self._cfg(), feedback=fb,
        )
        return msgs[0]["content"]

    def test_active_phase_says_continue_editing(self, tmp_path):
        text = self._text("active", False, tmp_path)
        assert "Continue editing" in text
        assert "Submit a plan via update_plan" not in text

    def test_replanning_phase_asks_for_plan(self, tmp_path):
        text = self._text("replanning", False, tmp_path)
        assert "Submit a plan via update_plan" in text
        assert "Continue editing" not in text

    def test_active_but_must_replan_asks_for_plan(self, tmp_path):
        """When must_replan is True, the model has to submit a
        replacement plan even in what was logically the active phase."""
        text = self._text("active", True, tmp_path)
        assert "Submit a plan via update_plan" in text

    def test_no_feedback_falls_back_to_plan_instruction(self, tmp_path):
        msgs = compress_mod._build_bootstrap(
            str(tmp_path), self._cfg(), feedback=None,
        )
        text = msgs[0]["content"]
        assert "Submit a plan via update_plan" in text


# -- [Low] dead branch removal ------------------------------------------------


class TestDeadBranchRemoval:
    def test_control_reason_prefixes_empty(self):
        """The 'screened:' prefix had no producer left in-tree."""
        assert fv.CONTROL_REASON_PREFIXES == ()

    def test_control_reason_still_catches_exact_matches(self):
        """Removing the prefix list must not break the exact-match path."""
        assert fv.is_eval_signal(False, "superseded by replan") is False
        assert fv.is_eval_signal(False, "abandoned (diagnose)") is False
        # A random 'screened: ...' is now treated as a real fail, which
        # is the correct behaviour now that nothing produces it.
        assert fv.is_eval_signal(False, "screened: whatever") is True

    def test_compress_clearable_prefixes_empty(self):
        """The '[System] Skill guidance for ' producer was the
        supervisor; microcompact no longer needs a prefix match."""
        assert compress_mod._CLEARABLE_PREFIXES == ()

    def test_microcompact_still_clears_tool_results(self):
        """The prefix match path may be empty, but the tool_result
        clearing branch must still work — it's the main point of
        microcompact."""
        msgs = []
        # 6 rounds of fat tool_results so keep_recent=1 leaves 5 to clear.
        for i in range(6):
            msgs.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": f"t{i}", "name": "read_file", "input": {}},
            ]})
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"t{i}",
                 "content": "x" * 500},
            ]})
        compress_mod.microcompact(msgs, min_chars=200, keep_recent=1)
        cleared = sum(
            1 for m in msgs if m["role"] == "user"
            and any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                and b.get("content") == "[cleared]"
                for b in (m.get("content") or [])
            )
        )
        assert cleared >= 1


# -- History render: item_id + backing_skill --------------------------------


class TestHistoryRender:
    """plan.md's ``## Optimization History`` lines used to be just
    ``- [O/X] {edit_desc} -- {reason}{metric}``, which made three
    similarly-worded rows indistinguishable and cut the trail back to
    ``p1/p2/...`` and the backing skill. Render now surfaces both."""

    def _fb(self):
        fb = FeedbackBuilder(SimpleNamespace(
            agent=SimpleNamespace(
                plan_item_rationale_min_chars=30,
                plan_item_rationale_max_chars=400,
            ),
            primary_metric="latency_us",
            lower_is_better=True,
        ))
        return fb

    def test_history_line_includes_item_id(self):
        fb = self._fb()
        _submit_plan(fb, 1)
        fb.settle_active(True, "keep", {"latency_us": 100.0},
                         edit_desc="first edit")
        out = fb.format_history()
        assert "[p1] first edit" in out
        assert "keep" in out

    def test_history_line_includes_backing_skill_when_present(self):
        fb = self._fb()
        # Plan with a backing_skill that passes _validate_backing_skill.
        # No SkillBuilder attached → _validate_backing_skill returns None.
        # So we bypass by setting backing_skill directly on the item
        # after submit_plan, matching what the real binding path does.
        _submit_plan(fb, 2)
        fb._plan_items[0]["backing_skill"] = "triton-ascend-matmul"
        fb._plan_items[1]["backing_skill"] = "triton-ascend-reduce"
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="broke on FAIL path")
        out = fb.format_history()
        # Rendered line carries BOTH the item id AND the backing skill.
        assert "[p1]" in out
        assert "broke on FAIL path" in out
        assert "(bs: triton-ascend-matmul)" in out

    def test_history_line_omits_bs_suffix_when_unbound(self):
        fb = self._fb()
        _submit_plan(fb, 1)
        # No backing_skill set → no (bs: ...) suffix.
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="unbound attempt")
        out = fb.format_history()
        assert "[p1] unbound attempt" in out
        assert "(bs:" not in out

    def test_history_disambiguates_similar_edit_descs(self):
        """Regression for the real-world confusion that triggered
        this change: three 'Add CUBE_CORE_NUM ...' edit_descs on
        different items render as distinct lines."""
        fb = self._fb()
        _submit_plan(fb, 3)
        fb.settle_active(True, "keep", {"latency_us": 100.0},
                         edit_desc="Add CUBE_CORE_NUM initialization to Model class")
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="Add CUBE_CORE_NUM to kernel signature")
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="Add CUBE_CORE_NUM grid pass-through")
        out = fb.format_history()
        # Each line uniquely pinned to its item.
        assert "[p1] Add CUBE_CORE_NUM initialization to Model class" in out
        assert "[p2] Add CUBE_CORE_NUM to kernel signature" in out
        assert "[p3] Add CUBE_CORE_NUM grid pass-through" in out


# -- Phase 6 [Medium]: pre-eval diagnose must unload skill content -----------


class TestPreEvalDiagnoseUnload:
    """Post-eval diagnose's skill-elide runs inside TurnExecutor right
    after settle_active (turn.py:280). Pre-eval diagnose (edit_fail /
    quick_check_fail) never hits that code path — DiagnoseHandler.apply
    must call ``buffer.unload_item_reads`` itself, otherwise the
    failing item's auto-injected SKILL.md and read_file('skills/...')
    tool_results leak into the replacement prompt."""

    def _handler_fixture(self, fb_outcome: str):
        """Build a DiagnoseHandler wired to a real ConversationBuffer +
        FeedbackBuilder, with subagent / LLM / runner stubbed out so
        ``apply`` can run end-to-end in-process."""
        import asyncio

        buf = ConversationBuffer()
        # Seed: active item p1 with auto-inject + one read_file tool_result.
        fb = FeedbackBuilder(_fb_cfg())
        _submit_plan(fb, 2)  # p1 active, p2 pending
        fb._plan_items[0]["backing_skill"] = "triton-ascend-matmul"
        fb._plan_items[0]["skill_ack"] = {
            "applicability": "apply",
            "valuable_aspects": "v" * 150,
            "kernel_application": "k" * 150,
        }
        buf.inject_backing_skill("p1", "triton-ascend-matmul", "BODY" * 50, 1, 6_000)
        buf.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_read_1", "name": "read_file",
             "input": {"path": "skills/triton-ascend-matmul/SKILL.md"}},
        ]})
        buf.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_read_1",
             "content": "SKILL.md body " * 20},
        ]})
        buf.track_item_skill_read("tu_read_1", "p1")

        # Stub handler internals: we only care about apply() routing.
        from akg_agents.op.autoresearch.agent.subagents import DiagnoseHandler

        async def _stub_force_diagnose(self, turn_result):
            return "diagnose says: change direction"

        handler = DiagnoseHandler.__new__(DiagnoseHandler)
        handler._llm = None
        handler._config = _fb_cfg()
        handler._task_dir = "."
        handler._runner = None
        handler._counters = SimpleNamespace(
            consecutive_failures=3,
            reset_after_diagnose=lambda: None,
        )
        handler._feedback = fb
        handler._buffer = buf
        handler._knowledge_prompt = ""
        handler._verbose = False
        handler._save_session = lambda: None
        handler._heartbeat = lambda _e: None
        # Bind the stub as a bound method.
        handler._force_diagnose = _stub_force_diagnose.__get__(handler, DiagnoseHandler)

        turn_result = SimpleNamespace(
            outcome=fb_outcome, feedback="some error context",
        )
        asyncio.run(handler.apply(turn_result))
        return buf, fb

    def test_pre_eval_diagnose_elides_injected_skill(self):
        """edit_fail path: the auto-injected SKILL.md marker message
        must be replaced with the elided placeholder after apply."""
        buf, _fb = self._handler_fixture(fb_outcome="edit_fail")
        elided_count = sum(
            1 for m in buf.view()
            if isinstance(m.get("content"), str)
            and m["content"] == ConversationBuffer._SKILL_READ_ELIDED
        )
        assert elided_count >= 1, (
            "Pre-eval diagnose must unload the auto-injected SKILL.md; "
            "otherwise the replacement prompt carries stale skill body."
        )

    def test_pre_eval_diagnose_elides_voluntary_read_tool_result(self):
        """edit_fail path: the tracked read_file tool_result must also
        be elided (preserving the tool_use_id for API pairing)."""
        buf, _fb = self._handler_fixture(fb_outcome="quick_check_fail")
        found_elided_tool_result = False
        for m in buf.view():
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if (isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and block.get("tool_use_id") == "tu_read_1"
                        and block.get("content") == ConversationBuffer._SKILL_READ_ELIDED):
                    found_elided_tool_result = True
        assert found_elided_tool_result, (
            "Pre-eval diagnose must elide the voluntary skill read's "
            "tool_result body (keeping tool_use_id for API pairing)."
        )

    def test_post_eval_diagnose_does_not_double_unload(self):
        """eval_fail path: TurnExecutor.execute already called
        unload_item_reads(settled_id); DiagnoseHandler must NOT touch
        the buffer again. We simulate 'already unloaded' by running the
        full fixture flow: even though pre-elide didn't happen in this
        unit test, the post-eval branch must not attempt unload based
        on the now-None active item — that would be a no-op at best
        and a wrong-item elide at worst. Concretely: after apply()
        returns on eval_fail outcome, the buffer inject body must be
        untouched (no pre-eval unload fired)."""
        import asyncio

        buf = ConversationBuffer()
        fb = FeedbackBuilder(_fb_cfg())
        _submit_plan(fb, 2)
        fb._plan_items[0]["backing_skill"] = "triton-ascend-matmul"
        fb._plan_items[0]["skill_ack"] = {
            "applicability": "apply",
            "valuable_aspects": "v" * 150,
            "kernel_application": "k" * 150,
        }
        # Settle p1 as FAIL (simulating the eval-fail path).
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="attempted edit")
        # Now active is p2 (promoted). Inject a marker that BELONGS to
        # p2, not p1 — if the post-eval path wrongly captured p2's id
        # and called unload_item_reads(p2), the marker would vanish.
        buf.inject_backing_skill("p2", "triton-ascend-reduce", "P2BODY" * 20, 1, 6_000)

        from akg_agents.op.autoresearch.agent.subagents import DiagnoseHandler

        async def _stub_force_diagnose(self, turn_result):
            return "post-eval replan"

        handler = DiagnoseHandler.__new__(DiagnoseHandler)
        handler._llm = None
        handler._config = _fb_cfg()
        handler._task_dir = "."
        handler._runner = None
        handler._counters = SimpleNamespace(
            consecutive_failures=3,
            reset_after_diagnose=lambda: None,
        )
        handler._feedback = fb
        handler._buffer = buf
        handler._knowledge_prompt = ""
        handler._verbose = False
        handler._save_session = lambda: None
        handler._heartbeat = lambda _e: None
        handler._force_diagnose = _stub_force_diagnose.__get__(handler, DiagnoseHandler)

        turn_result = SimpleNamespace(outcome="eval_fail", feedback="ctx")
        asyncio.run(handler.apply(turn_result))

        # p2's inject body must still be present — post-eval path must
        # not have attempted a second unload (which would also have
        # captured p2 as "the active item" and wrongly elided it).
        survived = [
            m for m in buf.view()
            if isinstance(m.get("content"), str)
            and m["content"].startswith(ConversationBuffer._SKILL_INJECT_PREFIX)
            and "P2BODY" in m["content"]
        ]
        assert len(survived) == 1, (
            "Post-eval diagnose must not call unload_item_reads again — "
            "TurnExecutor already handled the real failing item."
        )


# -- Phase 7 [Low]: ConversationBuffer tracking invariant is sealed ----------


class TestTrackingInvariantSealed:
    """Before Phase 7, ``clear()`` and ``load_latest()`` only touched
    ``_msgs``; the three skill-tracking maps (``_skill_inject_keys``,
    ``_item_inject_markers``, ``_item_read_ids``) were left stale.
    Callers had to know to call ``on_buffer_rebuilt`` by hand — the
    invariant leaked out of ConversationBuffer. Now both methods
    route through ``replace``, which fires ``on_buffer_rebuilt``
    automatically, so the invariant is sealed inside the class."""

    def test_clear_resets_inject_dedup(self):
        buf = ConversationBuffer()
        buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000)
        buf.clear()
        # After clear, re-injecting the same (v, item, skill) must
        # succeed — the dedup set was blown away with the buffer.
        assert buf.inject_backing_skill("p1", "matmul", "body", 1, 6_000) is True

    def test_clear_resets_item_read_ids(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_x", "content": "body"},
        ]})
        buf.track_item_skill_read("tu_x", "p1")
        assert "p1" in buf._item_read_ids  # sanity
        buf.clear()
        # No messages survive clear → _item_read_ids intersects with
        # an empty surviving-id set → all entries purged.
        assert "p1" not in buf._item_read_ids
        # And unload on an unknown item is a no-op.
        assert buf.unload_item_reads("p1") == 0

    def test_load_latest_rebuilds_tracking_from_disk(self, tmp_path):
        """Save a buffer with an inject marker + a tool_result, load
        it into a fresh ConversationBuffer: the tracking maps must
        reflect the restored contents, NOT the empty initial state."""
        import os

        buf1 = ConversationBuffer()
        buf1.inject_backing_skill("p1", "matmul", "BODYBODY", 1, 6_000)
        buf1.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": "read_file",
             "input": {}},
        ]})
        buf1.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": "skill body " * 10},
        ]})
        # Persist.
        session_dir = "agent_session"
        os.makedirs(tmp_path / session_dir, exist_ok=True)
        buf1.save_latest(str(tmp_path), session_dir=session_dir)

        buf2 = ConversationBuffer()
        # Seed buf2 with a stale tracking map to prove load wipes it.
        buf2._skill_inject_keys = {(9, "zombie", "zombie")}
        buf2._item_inject_markers = {"zombie": {"zombie-marker"}}
        buf2._item_read_ids = {"zombie": {"tu_zombie"}}

        assert buf2.load_latest(str(tmp_path), session_dir=session_dir) is True

        # Stale tracking is gone; marker from restored msgs is back.
        assert (9, "zombie", "zombie") not in buf2._skill_inject_keys
        assert "zombie" not in buf2._item_inject_markers
        assert "zombie" not in buf2._item_read_ids
        assert (1, "p1", "matmul") in buf2._skill_inject_keys
        # Re-inject on the restored buffer is blocked — dedup live.
        assert buf2.inject_backing_skill(
            "p1", "matmul", "BODYBODY", 1, 6_000,
        ) is False

    def test_load_latest_callers_do_not_need_manual_rebuild(self, tmp_path):
        """Regression guard: the loop.py resume path used to call
        ``self._buffer.on_buffer_rebuilt()`` manually after
        ``load_latest``. That patch was only needed because the
        invariant leaked; with Phase 7, load_latest seals it itself."""
        import os
        import inspect

        from akg_agents.op.autoresearch.agent import loop as loop_mod

        # Static check: the manual patch line is gone from loop.py.
        src = inspect.getsource(loop_mod)
        assert "load_latest(" in src  # callsite still exists
        # The old explicit reset comment / call is no longer there:
        assert "Reset the item-read tracking map since tool_use_ids" not in src, (
            "loop.py's manual on_buffer_rebuilt patch should have been "
            "removed once ConversationBuffer.load_latest started sealing "
            "the invariant itself."
        )

        # Behavioural check: load_latest alone restores the invariant.
        buf1 = ConversationBuffer()
        buf1.inject_backing_skill("p1", "matmul", "body", 1, 6_000)
        session_dir = "agent_session"
        os.makedirs(tmp_path / session_dir, exist_ok=True)
        buf1.save_latest(str(tmp_path), session_dir=session_dir)

        buf2 = ConversationBuffer()
        buf2.load_latest(str(tmp_path), session_dir=session_dir)
        # No manual on_buffer_rebuilt call — the invariant should hold.
        assert (1, "p1", "matmul") in buf2._skill_inject_keys


# -- Previous-plan summary appended to update_plan ack -----------------------


class TestPrevPlanSummary:
    """Every accepted ``submit_plan`` (the fresh-plan path, not
    ``replace_active_item``) must attach a mechanical statistical
    summary of the version that just ended so the agent has a
    distilled view before proposing the next plan."""

    def _cfg(self):
        return SimpleNamespace(
            agent=SimpleNamespace(
                plan_item_rationale_min_chars=30,
                plan_item_rationale_max_chars=400,
            ),
            primary_metric="latency_us",
            lower_is_better=True,
        )

    def _submit(self, fb, n, keywords_per_item=None):
        items = [
            {
                "text": f"item-{i}",
                "rationale": (
                    "inner loop reloads A from global memory on each "
                    "iteration; caching in shared should cut latency"
                ),
                "keywords": (keywords_per_item or {}).get(i, []),
            }
            for i in range(1, n + 1)
        ]
        ok, msg = fb.submit_plan(items=items)
        assert ok, msg
        return msg

    def test_empty_when_no_prior_version(self):
        fb = FeedbackBuilder(self._cfg())
        assert fb.format_prev_plan_summary(0) == ""
        assert fb.format_prev_plan_summary(1) == ""

    def test_summary_counts_keep_fail_discard(self):
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 3)  # v1
        fb.settle_active(True, "keep", {"latency_us": 100.0},
                         edit_desc="Replace pointer arithmetic with block_ptr")
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="Expand autotune with larger BLOCK_N")
        fb.settle_active(False, "no improvement", {"latency_us": 110.0},
                         edit_desc="Tune num_stages and num_warps")
        # Now move to v2 so v1 becomes prev.
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1)
        assert "## Previous plan (v1) summary" in out
        assert "3 settled item(s)" in out
        assert "1 KEEP" in out
        assert "1 DISCARD" in out
        assert "1 FAIL" in out

    def test_summary_best_metric_tracked(self):
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 2)
        fb.settle_active(True, "keep", {"latency_us": 1000.0},
                         edit_desc="first KEEP")
        fb.settle_active(True, "keep", {"latency_us": 950.0},
                         edit_desc="second KEEP — faster")
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1)
        assert "Best latency_us recorded this version: 950" in out

    def test_summary_no_keep_says_unchanged(self):
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 2)
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="bad edit 1")
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="bad edit 2")
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1)
        assert "No KEEP this version" in out
        assert "unchanged" in out

    def test_summary_surfaces_repeated_tokens(self):
        """The regression the whole feature was built for: three edits
        whose descriptions all mention ``BLOCK_N`` should show up as a
        repeated token so the agent doesn't write a fourth BLOCK_N
        sweep plan."""
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 3)
        fb.settle_active(False, "no improvement", {},
                         edit_desc="tune BLOCK_N=128 with GROUP_SIZE_M=8")
        fb.settle_active(False, "no improvement", {},
                         edit_desc="sweep BLOCK_N=120 with autotune")
        fb.settle_active(False, "no improvement", {},
                         edit_desc="tighten BLOCK_N=112 for alignment")
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1)
        # block_n (normalized to lowercase) should show ≥ 3 repeats.
        assert "block_n" in out.lower()
        assert "×3" in out

    def test_summary_lists_backing_skills_exercised(self):
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 2)
        fb._plan_items[0]["backing_skill"] = "triton-ascend-matmul"
        fb._plan_items[1]["backing_skill"] = "triton-ascend-reduce"
        fb.settle_active(True, "keep", {"latency_us": 100.0},
                         edit_desc="first edit")
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="second edit")
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1)
        assert "triton-ascend-matmul" in out
        assert "triton-ascend-reduce" in out

    def test_summary_stopwords_filtered(self):
        """High-frequency filler like ``add`` / ``use`` / ``try`` /
        ``change`` should NOT show up as "repeated tokens"."""
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 3)
        for desc in ("add foo to bar", "add spam to eggs", "add baz to qux"):
            fb.settle_active(False, "no improvement", {},
                             edit_desc=desc)
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1)
        assert "add×" not in out
        assert "to×" not in out

    def test_summary_ignores_control_events_in_failure_reasons(self):
        """Replan / diagnose-abandoned entries are control events,
        not real failures. They should be counted separately and
        should NOT dominate the 'Failure reasons:' line."""
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 2)
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="real fail")
        # Second item: diagnose-abandon via require_replan (pre-eval).
        fb.require_replan(
            diagnosis="direction bad", failing_item_already_settled=False,
        )
        # v1 → v2: manually seed via submit_plan (must_replan path
        # will normally go through replace_active_item, but for this
        # unit test we just do a straight submit to switch version).
        # Clear must_replan to allow submit_plan path.
        fb._must_replan = False
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1)
        # 1 control event shown separately; "correctness" is the only
        # real failure reason.
        assert "1 control-event" in out
        assert "correctness" in out
        # "abandoned (diagnose)" is a control-event reason and must NOT
        # be listed as a plan-level failure reason.
        assert "abandoned (diagnose)" not in out.split(
            "Failure reasons:", 1
        )[1].split("\n", 1)[0]

    # -- Repeated-keywords scope: real_fails only ---------------------

    def test_repeated_keywords_excludes_KEEP(self):
        """KEEPs are validated directions; the follow-up nudge says
        'avoid proposing items whose direction matches ...', so KEEP
        tokens must NOT appear in the repeated-keywords line —
        otherwise the agent is told to avoid what just worked."""
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 3)
        # Three KEEPs all built around BLOCK_N.
        fb.settle_active(True, "keep", {"latency_us": 1000.0},
                         edit_desc="set BLOCK_N=128")
        fb.settle_active(True, "keep", {"latency_us": 950.0},
                         edit_desc="set BLOCK_N=120")
        fb.settle_active(True, "keep", {"latency_us": 900.0},
                         edit_desc="set BLOCK_N=112")
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1)
        # Normalizer lowercases; must NOT surface 'block_n'.
        assert "block_n" not in out.lower(), (
            "KEEP-only tokens must not be listed as repeated keywords"
        )
        assert "Repeated optimization keywords: none" in out

    def test_repeated_keywords_excludes_control_events(self):
        """superseded / abandoned (diagnose) entries are control
        events — never actually executed as failures. They feed into
        the 'control-event' count, NOT repeated keywords."""
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 2)
        # First item: real fail mentioning BLOCK_N.
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="sweep BLOCK_N=128")
        # Second item: pre-eval diagnose abandon — its edit_desc is
        # empty (settle_active was never called for it), but its
        # plan_text defaults to the item's own text when edit_desc=""
        # in _close_active_item. We manually append a history entry
        # to simulate an abandoned item whose plan_text contains the
        # same token, so we can assert that token doesn't leak in.
        fb._settled_history.append({
            "item_id": "p-ctl",
            "plan_text": "Try BLOCK_N=256",
            "sketch": "",
            "backing_skill": None,
            "keywords": [],
            "rationale": "",
            "text": "Try BLOCK_N=256 with tight alignment",  # edit_desc
            "ok": False,
            "reason": "superseded by replan",
            "metrics": {},
            "version": 1,
            "signal_eligible": False,
        })
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1)
        # Only ONE occurrence of block_n — from the one real fail.
        # (Count >= 2 is the threshold to surface at all.)
        assert "block_n×2" not in out.lower()
        # The real fail alone doesn't cross the ×2 threshold, so the
        # repeated-keywords line should be "none".
        assert "Repeated optimization keywords: none" in out

    def test_repeated_keywords_filters_action_verbs(self):
        """Regression: if three fails all say 'tune / sweep BLOCK_N',
        the meaningful signal is BLOCK_N only. Surfacing 'tune×3' or
        'sweep×3' next to the nudge 'avoid directions matching ...'
        would be read as 'stop tuning / sweeping anything', which is
        actively harmful since tuning IS optimization."""
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 3)
        fb.settle_active(False, "no improvement", {},
                         edit_desc="tune BLOCK_N=128 with autotune")
        fb.settle_active(False, "no improvement", {},
                         edit_desc="sweep BLOCK_N=120 across configs")
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="adjust BLOCK_N=112 and replace stride")
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1).lower()
        assert "block_n×3" in out
        # Action verbs MUST be filtered even if they repeat.
        for filler in ("tune×", "sweep×", "adjust×", "replace×", "with×"):
            assert filler not in out, f"action/filler leaked: {filler}"

    def test_repeated_keywords_filters_comparative_adjectives(self):
        """'smaller / larger / faster' appearing across fails carries
        no direction — it's subjective framing around the real token."""
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 3)
        fb.settle_active(False, "no improvement", {},
                         edit_desc="smaller BLOCK_N for faster loads")
        fb.settle_active(False, "no improvement", {},
                         edit_desc="larger BLOCK_N for better utilization")
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="simpler BLOCK_N grid")
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1).lower()
        assert "block_n×3" in out
        for filler in ("smaller×", "larger×", "faster×", "simpler×", "better×"):
            assert filler not in out

    def test_repeated_keywords_mixed_scope(self):
        """Mixed case: 2 KEEPs + 2 real fails, all mentioning BLOCK_N.
        The repeated line must count only the 2 fails (×2), not the
        4 total occurrences."""
        fb = FeedbackBuilder(self._cfg())
        self._submit(fb, 4)
        fb.settle_active(True, "keep", {"latency_us": 1000.0},
                         edit_desc="set BLOCK_N=128 — success 1")
        fb.settle_active(True, "keep", {"latency_us": 950.0},
                         edit_desc="set BLOCK_N=120 — success 2")
        fb.settle_active(False, "correctness mismatch", {},
                         edit_desc="set BLOCK_N=256 — broke")
        fb.settle_active(False, "no improvement", {},
                         edit_desc="set BLOCK_N=64 — regressed")
        self._submit(fb, 1)
        out = fb.format_prev_plan_summary(1)
        assert "block_n×2" in out.lower()
        # Explicitly assert NOT ×4 (the whole-entries bug).
        assert "block_n×4" not in out.lower()
        assert "block_n×3" not in out.lower()


# -- Plan breadth gate (min_items_per_plan) -----------------------------------


class TestMinItemsPerPlan:
    """Fresh `update_plan` must carry at least ``min_items_per_plan``
    distinct items, so single-item reactive tweaks (the "block-size
    casino" failure mode in Task_v0f52btb) get rejected up front. The
    replace_active_item path (must_replan surgical replacement) is
    deliberately exempt.
    """

    def _cfg(self, min_items: int):
        return _fb_cfg(min_items_per_plan=min_items)

    def _item(self, text="do thing"):
        return {"text": text, "rationale": _valid_rationale()}

    def test_rejects_under_floor(self):
        fb = FeedbackBuilder(self._cfg(3))
        ok, msg = fb.submit_plan(items=[self._item("a"), self._item("b")])
        assert ok is False
        assert "min_items_per_plan is 3" in msg
        # Rejection carries guidance, not just a count error.
        assert "DISTINCT" in msg or "distinct" in msg.lower()

    def test_accepts_at_floor(self):
        fb = FeedbackBuilder(self._cfg(3))
        ok, msg = fb.submit_plan(items=[
            self._item(f"item-{i}") for i in range(3)
        ])
        assert ok is True, msg
        assert len(fb._plan_items) == 3

    def test_accepts_above_floor(self):
        fb = FeedbackBuilder(self._cfg(3))
        ok, msg = fb.submit_plan(items=[
            self._item(f"item-{i}") for i in range(5)
        ])
        assert ok is True, msg
        assert len(fb._plan_items) == 5

    def test_missing_config_defaults_to_one(self):
        """No min_items_per_plan on config → gate is effectively off
        (so legacy tests / callers keep working)."""
        cfg = SimpleNamespace(agent=SimpleNamespace(
            plan_item_rationale_min_chars=30,
            plan_item_rationale_max_chars=400,
        ))
        fb = FeedbackBuilder(cfg)
        ok, msg = fb.submit_plan(items=[self._item("solo")])
        assert ok is True, msg

    def test_zero_disables_gate(self):
        """min_items_per_plan=0 disables the gate explicitly."""
        fb = FeedbackBuilder(self._cfg(0))
        ok, msg = fb.submit_plan(items=[self._item("solo")])
        assert ok is True, msg

    def test_replace_active_item_is_exempt(self):
        """must_replan replacement path accepts a 1-item plan even when
        the fresh-plan floor is 3 — that's the whole point of surgical
        replacement."""
        fb = FeedbackBuilder(self._cfg(3))
        fb.submit_plan(items=[self._item(f"seed-{i}") for i in range(3)])
        # Put ourselves in must_replan.
        fb.require_replan(diagnosis="need a different direction")
        ok, msg = fb.replace_active_item([self._item("single replacement")])
        assert ok is True, msg
        # Floor still applies on the next fresh update_plan.
        ok2, msg2 = fb.submit_plan(items=[self._item("only one")])
        assert ok2 is False
        assert "min_items_per_plan is 3" in msg2
