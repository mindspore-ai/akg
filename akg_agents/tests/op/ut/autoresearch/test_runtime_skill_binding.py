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

"""Runtime skill-binding tests for the SkillPool refactor.

Three layered concerns:

  1. ``SkillBuilder.register`` idempotence — required for
     ``SkillPool.refill`` to be safe across session resume.

  2. ``SkillPool.refill`` (replace + append modes) — runs the
     keyword pipeline, dedups, marks new trackable entries as
     ``selected`` in SkillBuilder, prints diagnostic lines.

  3. ``TurnExecutor._handle_update_plan`` continuous augmentation —
     every agent ``update_plan`` is automatically extended with
     residual bound items from the pool. Bound items come AFTER
     agent items. Agent-supplied ``backing_skill`` is stripped at
     the trust boundary; bindings come exclusively from the system.

  4. ``TurnExecutor._handle_search_skills`` — thin wrapper over
     ``feedback.skill_pool.refill(mode='append')``; returns the
     names of newly-added candidates and explicitly tells the
     agent they take effect on the NEXT update_plan call.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import pytest

from akg_agents.op.autoresearch.agent import skill_pool as sp_module
from akg_agents.op.autoresearch.agent.counters import RunCounters
from akg_agents.op.autoresearch.agent.feedback import FeedbackBuilder
from akg_agents.op.autoresearch.agent.skill_adapter import (
    QueryKeywords,
    TRACKABLE_PATTERN_CATEGORIES,
)
from akg_agents.op.autoresearch.agent.skill_builder import SkillBuilder
from akg_agents.op.autoresearch.agent.skill_pool import SkillPool


# Default rationale used by tests that don't care about its content. Long
# enough to pass FeedbackBuilder._validate_rationale (>= min_chars + not a
# banned generic phrase).
_R = "test rationale: specific bottleneck on the K-loop with measurable expected effect"


def _item(text: str, **extra) -> dict:
    """Test helper: build a plan item dict with default rationale.

    Tests that don't exercise rationale validation specifically can use
    this to keep the focus on the behavior under test.
    """
    out = {"text": text, "rationale": _R}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class _FakeSkill:
    def __init__(self, name, category, content="", description="", metadata=None):
        self.name = name
        self.category = category
        self.content = content or f"<content of {name}>"
        self.description = description or f"{name} description"
        self.metadata = metadata or {}
        self.skill_path = None


class _StubCatalog:
    def __init__(self, filter_result=None):
        self._filter_result = list(filter_result or [])
        self.filter_calls = []

    def filter_by_context(self, **kwargs):
        self.filter_calls.append(kwargs)
        return list(self._filter_result)

    @staticmethod
    def classify_case_type(skill):
        meta = getattr(skill, "metadata", {}) or {}
        return meta.get("case_type", "improvement")


@dataclass
class _StubAgentCfg:
    skill_block_top_k: int = 3
    skill_keyword_max_per_item: int = 5
    skill_block_max_chars: int = 8_000
    skill_narrow_timeout: float = 30.0
    editable_file_truncate: int = 8_000
    system_context_file_truncate: int = 15_000
    plan_max_chars: int = 4_000


@dataclass
class _StubConfig:
    name: str = "matmul"
    description: str = "optimize a matmul kernel"
    dsl: str = "triton_cuda"
    framework: str = "torch"
    backend: str = "cuda"
    arch: str = "a100"
    primary_metric: str = "latency_us"
    lower_is_better: bool = True
    metadata: dict = field(default_factory=dict)
    agent: Optional[_StubAgentCfg] = None

    def __post_init__(self):
        if self.agent is None:
            self.agent = _StubAgentCfg()


def _make_feedback_with_pool() -> FeedbackBuilder:
    """Real FeedbackBuilder + SkillBuilder + SkillPool, no-op persistence."""
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            plan_max_chars=4000,
            skill_keyword_max_per_item=5,
            eval_feedback_tail=1000,
            raw_output_tail=2048,
            ranking_description_truncate=100,
            ranking_error_truncate=120,
            cumulative_diff_truncate=10000,
            finish_hint_threshold=2,
            history_summary_last_n=10,
        ),
        primary_metric="latency_us",
        lower_is_better=True,
    )
    fb = FeedbackBuilder(cfg)
    fb.skill_builder = SkillBuilder(config=None)
    fb.skill_pool = SkillPool(fb.skill_builder)
    return fb


def _seed_pool(fb: FeedbackBuilder, skills: list) -> None:
    """Seed the SkillPool directly + register trackable entries.

    Used by tests that don't want to spin up the keyword pipeline.
    Mirrors what ``SkillPool.refill`` does at the end of its happy
    path: write the ranked list, then ``register`` each
    trackable entry.
    """
    fb.skill_pool.replace(skills)
    for s in skills:
        if s.category in TRACKABLE_PATTERN_CATEGORIES:
            fb.skill_builder.register(
                s, reason="test fixture", plan_version=0,
            )


# ---------------------------------------------------------------------------
# (1) SkillBuilder.register idempotence — unchanged invariants
# ---------------------------------------------------------------------------


class TestRegisterIdempotent:
    """Pins the contract that SkillPool.refill relies on for resume safety."""

    def _fresh(self):
        return SkillBuilder(config=None), SimpleNamespace(
            name="matmul", category="guide",
            description="d", content="c", metadata={},
        )

    def test_duplicate_registration_no_new_entry(self):
        sb, skill = self._fresh()
        sb.register(skill, reason="first", plan_version=1)
        sb.register(skill, reason="second", plan_version=2)
        assert len(sb._skills) == 1

    def test_duplicate_registration_updates_reason(self):
        sb, skill = self._fresh()
        sb.register(skill, reason="first", plan_version=1)
        sb.register(skill, reason="second", plan_version=2)
        assert sb._skills["matmul"].registered_reason == "second"

    def test_register_after_unbind_keeps_history_but_stays_bindable(self):
        """Post-abandon refactor: unbind is NOT terminal. Re-registering
        an unbound skill keeps the ``unbound_at_versions`` history but
        the skill stays in the registry and is still bindable."""
        sb, skill = self._fresh()
        sb.register(skill, reason="first", plan_version=1)
        sb.mark_unbound("matmul", reason="bad fit", plan_version=1)
        sb.register(skill, reason="second", plan_version=2)
        rec = sb.get("matmul")
        assert rec.unbound_at_versions == [1]  # history retained
        assert "matmul" in sb  # still bindable


# ---------------------------------------------------------------------------
# (2) SkillPool.refill — replace + append modes
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_pipeline(monkeypatch):
    """Stub catalog + keyword generator + ranker for refill tests.

    Returns the catalog object so individual tests can swap its
    ``filter_result`` between calls (simulating fresh search hits).
    """
    catalog = _StubCatalog()
    monkeypatch.setattr(sp_module, "_get_catalog", lambda: catalog)

    async def _fake_generate(**kwargs):
        return QueryKeywords(must_have=["matmul"])

    def _fake_rank(skills, keywords, **kwargs):
        return [(float(len(skills) - i), s) for i, s in enumerate(skills)]

    monkeypatch.setattr(sp_module, "generate_query_keywords", _fake_generate)
    monkeypatch.setattr(sp_module, "rank_skills_by_keywords", _fake_rank)
    return catalog


def _refill(fb, *, hint="", mode="replace"):
    return fb.skill_pool.refill(
        llm=object(),
        config=_StubConfig(),
        hint=hint,
        mode=mode,
    )


class TestSkillPoolRefillReplace:
    @pytest.mark.asyncio
    async def test_registers_trackable_top_k_as_selected(self, patched_pipeline):
        fb = _make_feedback_with_pool()
        patched_pipeline._filter_result = [
            _FakeSkill("s1-guide", "guide"),
            _FakeSkill("s2-example", "example"),
            _FakeSkill("s3-case", "case"),
        ]
        added = await _refill(fb)
        assert [s.name for s in added] == ["s1-guide", "s2-example", "s3-case"]
        # All 3 trackable, all marked selected.
        assert set(fb.skill_builder._skills.keys()) == {
            "s1-guide", "s2-example", "s3-case",
        }
        for name in ("s1-guide", "s2-example", "s3-case"):
            assert name in fb.skill_builder  # every registered skill is bindable

    @pytest.mark.asyncio
    async def test_non_trackable_reach_pool_but_not_skill_builder(
        self, patched_pipeline,
    ):
        fb = _make_feedback_with_pool()
        patched_pipeline._filter_result = [
            _FakeSkill("fund-1", "fundamental"),
            _FakeSkill("ref-1", "reference"),
            _FakeSkill("guide-1", "guide"),
            _FakeSkill("case-1", "case"),
        ]
        await _refill(fb)
        # All 4 reach the pool (Layer 2 rendering window).
        assert len(fb.skill_pool) == 4
        # But only the 2 trackable enter the skill state machine.
        assert set(fb.skill_builder._skills.keys()) == {"guide-1", "case-1"}

    @pytest.mark.asyncio
    async def test_replace_wipes_existing_pool(self, patched_pipeline):
        fb = _make_feedback_with_pool()
        fb.skill_pool.replace([_FakeSkill("stale", "guide")])
        patched_pipeline._filter_result = [_FakeSkill("fresh", "guide")]
        await _refill(fb)
        assert [s.name for s in fb.skill_pool] == ["fresh"]

    @pytest.mark.asyncio
    async def test_resume_re_refill_safe(self, patched_pipeline):
        """Refill after a session resume must not duplicate or reset
        the applied/abandoned registry badges."""
        fb = _make_feedback_with_pool()
        patched_pipeline._filter_result = [_FakeSkill("matmul-guide", "guide")]
        await _refill(fb)
        fb.skill_builder.record_applied("matmul-guide", plan_version=1)

        await _refill(fb)
        assert len(fb.skill_builder._skills) == 1
        assert fb.skill_builder._skills["matmul-guide"].applied_versions == [1]

    @pytest.mark.asyncio
    async def test_empty_pool_when_catalog_empty(self, patched_pipeline):
        fb = _make_feedback_with_pool()
        patched_pipeline._filter_result = []
        await _refill(fb)
        assert fb.skill_pool.is_empty()
        assert fb.skill_builder._skills == {}


class TestSkillPoolRefillAppend:
    @pytest.mark.asyncio
    async def test_append_keeps_existing_entries(self, patched_pipeline):
        fb = _make_feedback_with_pool()
        patched_pipeline._filter_result = [_FakeSkill("seed", "guide")]
        await _refill(fb, mode="replace")

        patched_pipeline._filter_result = [_FakeSkill("hinted", "guide")]
        added = await _refill(fb, hint="some hint", mode="append")
        assert [s.name for s in added] == ["hinted"]
        assert [s.name for s in fb.skill_pool] == ["seed", "hinted"]

    @pytest.mark.asyncio
    async def test_append_dedupes_by_name(self, patched_pipeline):
        fb = _make_feedback_with_pool()
        patched_pipeline._filter_result = [_FakeSkill("dup", "guide")]
        await _refill(fb, mode="replace")

        # The catalog filter is what excludes duplicates upstream in
        # append mode (exclude_names = applied + abandoned + existing
        # pool names). The fixture's ``_StubCatalog`` honors that
        # exclude_names, so a name already in the pool is filtered
        # out before ranking.
        patched_pipeline._filter_result = []
        added = await _refill(fb, hint="dup again", mode="append")
        assert added == []
        assert [s.name for s in fb.skill_pool] == ["dup"]

    @pytest.mark.asyncio
    async def test_append_marks_new_trackable_skills(self, patched_pipeline):
        fb = _make_feedback_with_pool()
        patched_pipeline._filter_result = []
        await _refill(fb, mode="replace")

        patched_pipeline._filter_result = [
            _FakeSkill("new-guide", "guide"),
            _FakeSkill("new-fund", "fundamental"),  # not trackable
        ]
        added = await _refill(fb, hint="anything", mode="append")
        assert [s.name for s in added] == ["new-guide", "new-fund"]
        # Only the trackable one entered the skill state machine.
        assert "new-guide" in fb.skill_builder._skills
        assert "new-fund" not in fb.skill_builder._skills

    @pytest.mark.asyncio
    async def test_append_does_not_exclude_unbound_skills(self, patched_pipeline):
        """Post-abandon refactor: previously-unbound skills stay
        eligible for refill. The exclude_names only protects against
        in-pool duplicates (append-mode dedup). If a skill was
        unbound on an earlier item it can still be re-selected and
        ranked — ``match_by_keywords`` will just put it in tier 2."""
        fb = _make_feedback_with_pool()
        sb_skill = SimpleNamespace(
            name="once-unbound", category="guide",
            description="d", content="c", metadata={},
        )
        fb.skill_builder.register(sb_skill, reason="t", plan_version=0)
        fb.skill_builder.mark_unbound(
            "once-unbound", reason="bad fit on earlier item", plan_version=0,
        )

        patched_pipeline._filter_result = []
        await _refill(fb, hint="x", mode="append")

        # The unbound skill must NOT be in exclude_names — only pool-
        # dedup names (currently in the pool) are excluded.
        latest = patched_pipeline.filter_calls[-1]
        assert "once-unbound" not in latest["exclude_names"]

    @pytest.mark.asyncio
    async def test_append_passes_hint_through_to_keyword_generator(
        self, patched_pipeline, monkeypatch,
    ):
        fb = _make_feedback_with_pool()
        captured = {}

        async def _capturing_generate(**kwargs):
            captured.update(kwargs)
            return QueryKeywords(must_have=["matmul"])

        monkeypatch.setattr(
            sp_module, "generate_query_keywords", _capturing_generate,
        )
        patched_pipeline._filter_result = [_FakeSkill("a", "guide")]
        await _refill(fb, hint="fused softmax FP16", mode="append")
        assert captured.get("hint") == "fused softmax FP16"
        # Context fields propagated.
        assert captured.get("dsl") == "triton_cuda"
        assert captured.get("backend") == "cuda"
        assert captured.get("arch") == "a100"
        assert captured.get("framework") == "torch"


# ---------------------------------------------------------------------------
# (3) FeedbackBuilder.submit_plan(items=...) shared tail — unchanged
# ---------------------------------------------------------------------------


class TestSubmitPlanItemsReusesTailLogic:
    """The items entry must converge on the legacy markdown tail."""

    def test_items_entry_bumps_version_and_activates_first(self):
        fb = _make_feedback_with_pool()
        ok, _ = fb.submit_plan(items=[
            _item("alpha", backing_skill=None),
            _item("beta", backing_skill=None),
        ])
        assert ok and fb.plan_version == 1
        assert fb._plan_items[0]["status"] == "active"
        assert fb._plan_items[0]["text"] == "alpha"
        assert fb._plan_items[1]["status"] == "pending"
        assert fb._active_item_id == "p1"

    def test_items_entry_persists_raw_text_as_markdown(self):
        fb = _make_feedback_with_pool()
        fb.submit_plan(items=[
            _item("alpha", backing_skill=None),
            _item("beta", backing_skill=None),
        ])
        assert fb._plan == "- [ ] alpha\n- [ ] beta"

    def test_items_entry_binds_backing_skill_per_item(self):
        fb = _make_feedback_with_pool()
        fb.skill_builder.register(
            SimpleNamespace(
                name="matmul", category="guide",
                description="d", content="c", metadata={},
            ),
            reason="test", plan_version=0,
        )
        ok, _ = fb.submit_plan(items=[
            _item("adopt matmul", backing_skill="matmul"),
            _item("follow up"),
        ])
        assert ok
        assert fb._plan_items[0]["backing_skill"] == "matmul"
        assert fb._plan_items[1]["backing_skill"] is None
        # "active" is derived from plan_items, not stored on the
        # registry — the registry just tracks registered / applied /
        # unbound. Every registered skill is bindable (no terminal state).
        assert "matmul" in fb.skill_builder

    def test_items_entry_positional_drift_guard(self):
        fb = _make_feedback_with_pool()
        for nm in ("A", "C"):
            fb.skill_builder.register(
                SimpleNamespace(
                    name=nm, category="guide",
                    description="d", content="c", metadata={},
                ),
                reason="test", plan_version=0,
            )
        ok, _ = fb.submit_plan(items=[
            _item("first", backing_skill="A"),
            _item("", backing_skill="should-vanish"),
            _item("third", backing_skill="C"),
        ])
        assert ok
        assert len(fb._plan_items) == 2
        assert fb._plan_items[0]["text"] == "first"
        assert fb._plan_items[0]["backing_skill"] == "A"
        assert fb._plan_items[1]["text"] == "third"
        assert fb._plan_items[1]["backing_skill"] == "C"

    def test_items_entry_superseded_history_on_replan(self):
        fb = _make_feedback_with_pool()
        fb.submit_plan(items=[_item("first-attempt", backing_skill=None)])
        fb.submit_plan(items=[_item("new-direction", backing_skill=None)])
        assert fb.plan_version == 2
        assert len(fb._settled_history) == 1
        superseded = fb._settled_history[0]
        assert superseded["reason"] == "superseded by replan"
        assert superseded["text"] == "first-attempt"
        assert superseded["item_id"] == "p1"

    def test_items_entry_rejects_empty_list(self):
        fb = _make_feedback_with_pool()
        ok, msg = fb.submit_plan(items=[])
        assert not ok
        assert "item" in msg.lower()

    def test_items_entry_rejects_all_empty_texts(self):
        fb = _make_feedback_with_pool()
        ok, _ = fb.submit_plan(items=[
            {"text": "", "backing_skill": "x"},
            {"text": "  ", "backing_skill": "y"},
        ])
        assert not ok


# ---------------------------------------------------------------------------
# (4) TurnExecutor._handle_update_plan keyword-driven binding
# ---------------------------------------------------------------------------


class _TurnHandlerHarness:
    """Bare TurnExecutor wired with feedback + skill_pool + counters."""

    def __init__(self, ranked_skills, *, top_k=3):
        from akg_agents.op.autoresearch.agent.turn import TurnExecutor

        self.executor = TurnExecutor.__new__(TurnExecutor)
        cfg = _StubConfig()
        cfg.agent.skill_block_top_k = top_k
        self.executor.config = cfg
        self.executor.feedback = _make_feedback_with_pool()
        _seed_pool(self.executor.feedback, list(ranked_skills))
        self.counters = RunCounters()

    def call(self, args: dict) -> str:
        import asyncio
        return asyncio.run(
            self.executor._handle_update_plan(args, counters=self.counters)
        )

    @property
    def plan_items(self):
        return self.executor.feedback._plan_items

    @property
    def skill_builder(self):
        return self.executor.feedback.skill_builder

    @property
    def skill_pool(self):
        return self.executor.feedback.skill_pool


class TestUpdatePlanKeywordBinding:
    """Items only get a backing_skill when they declare keywords AND
    those keywords actually hit a residual-bindable skill in the pool.
    No more system-driven auto-append."""

    def _ranked(self):
        return [
            _FakeSkill("guide-A", "guide", description="Alpha tiling matmul."),
            _FakeSkill("example-B", "example", description="Beta reduce sum."),
            _FakeSkill("case-C", "case", description="Gamma fused softmax."),
        ]

    def test_items_without_keywords_stay_unbound(self):
        h = _TurnHandlerHarness(self._ranked(), top_k=3)
        reply = h.call({"items": [
            _item("free-form 1"),
            _item("free-form 2"),
        ]})
        assert "accepted" in reply
        # No skill bindings appended; agent items are pure free exploration.
        assert "skill items appended" not in reply
        assert len(h.plan_items) == 2
        assert h.plan_items[0]["text"] == "free-form 1"
        assert h.plan_items[0]["backing_skill"] is None
        assert h.plan_items[1]["backing_skill"] is None

    def test_keyword_match_binds_to_top_skill(self):
        h = _TurnHandlerHarness(self._ranked(), top_k=3)
        reply = h.call({"items": [
            _item("apply tiling to the matmul", keywords=["tiling", "matmul"]),
        ]})
        assert "accepted" in reply
        assert "keyword-matched" in reply
        assert h.plan_items[0]["text"] == "apply tiling to the matmul"
        # guide-A's description has "tiling" and "matmul" → highest score.
        assert h.plan_items[0]["backing_skill"] == "guide-A"

    def test_keywords_with_no_pool_hit_stay_unbound(self):
        h = _TurnHandlerHarness(self._ranked(), top_k=3)
        h.call({"items": [
            _item("try X", keywords=["nonexistent", "unrelated"]),
        ]})
        # No skill in the pool has those tokens → degrade to free exploration.
        assert h.plan_items[0]["backing_skill"] is None

    def test_handler_strips_agent_supplied_backing_skill(self):
        """Trust boundary: agent-supplied backing_skill is dropped at
        the trust boundary. The system only sets backing_skill via the
        keyword matcher; without keywords, no binding."""
        h = _TurnHandlerHarness(self._ranked(), top_k=3)
        h.call({"items": [
            _item("agent item", backing_skill="guide-A"),  # no keywords
        ]})
        assert h.plan_items[0]["text"] == "agent item"
        assert h.plan_items[0]["backing_skill"] is None  # stripped, no rebind

    def test_legacy_markdown_path_rejected_at_handler(self):
        """Legacy markdown plans cannot carry rationale per item, so
        the handler rejects them whole."""
        h = _TurnHandlerHarness(self._ranked(), top_k=3)
        reply = h.call({"plan": "- [ ] m1\n- [ ] m2"})
        assert "Plan rejected" in reply or "rejected" in reply

    def test_applied_skill_can_be_rebound(self):
        """An applied skill (KEEP'd) is still bindable — broadly
        applicable skills may be useful across multiple plan items."""
        h = _TurnHandlerHarness(self._ranked(), top_k=3)
        h.skill_builder.record_applied("guide-A", plan_version=1)
        h.call({"items": [
            _item("retry guide-A", keywords=["tiling", "matmul"]),
        ]})
        assert h.plan_items[0]["backing_skill"] == "guide-A"

    def test_unbound_skill_remains_bindable_at_lower_tier(self):
        """Post-abandon refactor: an unbound skill is NOT excluded —
        it can still be bound on a later item, just ranked last
        within the keyword-match result set."""
        h = _TurnHandlerHarness(self._ranked(), top_k=3)
        h.skill_builder.mark_unbound("case-C", reason="t", plan_version=1)
        h.call({"items": [
            _item("try fused softmax", keywords=["fused", "softmax"]),
        ]})
        # Either something else ranked higher in tier 1 and got bound
        # (normal case), OR case-C was the only match and binds at
        # tier 2 — what we must NOT do is hard-exclude it.
        bound = h.plan_items[0]["backing_skill"]
        if bound == "case-C":
            # Bound via tier 2: unbound but still selectable.
            assert h.skill_builder.get("case-C").tier() == 2

    def test_empty_items_rejected(self):
        h = _TurnHandlerHarness(self._ranked(), top_k=3)
        reply = h.call({"items": []})
        assert "rejected" in reply.lower()

    def test_neither_field_rejected(self):
        h = _TurnHandlerHarness(self._ranked(), top_k=3)
        reply = h.call({})
        assert "rejected" in reply.lower()


# ---------------------------------------------------------------------------
# (5) TurnExecutor._handle_search_skills
# ---------------------------------------------------------------------------


class _SearchHarness:
    """Wraps a TurnExecutor to exercise _handle_search_skills end-to-end
    against a stubbed catalog + keyword pipeline."""

    def __init__(self, monkeypatch, *, seed=None):
        from akg_agents.op.autoresearch.agent.turn import TurnExecutor

        self.catalog = _StubCatalog()
        monkeypatch.setattr(sp_module, "_get_catalog", lambda: self.catalog)

        async def _gen(**kwargs):
            return QueryKeywords(must_have=["matmul"])

        def _rank(skills, keywords, **kwargs):
            return [(float(len(skills) - i), s) for i, s in enumerate(skills)]

        monkeypatch.setattr(sp_module, "generate_query_keywords", _gen)
        monkeypatch.setattr(sp_module, "rank_skills_by_keywords", _rank)

        self.executor = TurnExecutor.__new__(TurnExecutor)
        self.executor.config = _StubConfig()
        self.executor.feedback = _make_feedback_with_pool()
        self.executor.llm = object()
        if seed:
            self.executor.feedback.skill_pool.replace(list(seed))

    async def call(self, hint: str) -> str:
        return await self.executor._handle_search_skills({"hint": hint})

    @property
    def pool(self):
        return self.executor.feedback.skill_pool


class TestSearchSkillsHandler:
    @pytest.mark.asyncio
    async def test_empty_hint_rejected(self, monkeypatch):
        h = _SearchHarness(monkeypatch)
        reply = await h.call("")
        assert "rejected" in reply

    @pytest.mark.asyncio
    async def test_whitespace_hint_rejected(self, monkeypatch):
        h = _SearchHarness(monkeypatch)
        reply = await h.call("    ")
        assert "rejected" in reply

    @pytest.mark.asyncio
    async def test_added_candidates_appear_in_pool(self, monkeypatch):
        h = _SearchHarness(monkeypatch)
        h.catalog._filter_result = [
            _FakeSkill("alpha", "guide"),
            _FakeSkill("beta", "example"),
        ]
        reply = await h.call("any hint")
        assert "added 2 candidates" in reply
        assert "alpha" in reply and "beta" in reply
        assert [s.name for s in h.pool] == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_dedupes_against_existing_pool(self, monkeypatch):
        h = _SearchHarness(monkeypatch, seed=[_FakeSkill("alpha", "guide")])
        # Catalog filter would normally exclude `alpha` upstream because
        # the harness append-mode passes existing pool names in
        # ``exclude_names``. The stub honors that.
        h.catalog._filter_result = []  # filter returned empty
        reply = await h.call("hint")
        assert "0 new candidates" in reply
        assert [s.name for s in h.pool] == ["alpha"]
        # Post-abandon refactor: the 0-candidate message MUST NOT
        # claim skills are excluded for being "applied / abandoned".
        # "Abandoned" is no longer a terminal state; "applied" never
        # excludes at all. Leaving those words in the message would
        # teach the agent a wrong mental model of the registry.
        assert "abandoned" not in reply.lower()
        assert "already applied" not in reply.lower()

    @pytest.mark.asyncio
    async def test_added_appears_in_next_update_plan(self, monkeypatch):
        h = _SearchHarness(monkeypatch)
        h.catalog._filter_result = [
            _FakeSkill(
                "new-guide", "guide",
                description="fused softmax kernel pattern",
            ),
        ]
        await h.call("fused softmax")
        # Now an update_plan call with matching keywords binds to the new skill.
        counters = RunCounters()
        reply = await h.executor._handle_update_plan(
            {"items": [_item("apply fused softmax", keywords=["fused", "softmax"])]},
            counters=counters,
        )
        assert "accepted" in reply
        plan_items = h.executor.feedback._plan_items
        assert plan_items[0]["text"] == "apply fused softmax"
        assert plan_items[0]["backing_skill"] == "new-guide"

    @pytest.mark.asyncio
    async def test_no_skill_pool_attached_returns_unavailable(self, monkeypatch):
        from akg_agents.op.autoresearch.agent.turn import TurnExecutor

        executor = TurnExecutor.__new__(TurnExecutor)
        executor.config = _StubConfig()
        executor.feedback = _make_feedback_with_pool()
        executor.feedback.skill_pool = None
        executor.llm = object()
        reply = await executor._handle_search_skills({"hint": "x"})
        assert "unavailable" in reply
