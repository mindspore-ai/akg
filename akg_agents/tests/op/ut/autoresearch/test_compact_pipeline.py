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
"""Tests for the Phase 8 multi-step compact rewrite.

The new compact pipeline replaces the single ``_llm_summarize`` call
with TWO independent LLM calls that run concurrently:

  * LLM #1 ``_summarize_operator_from_keywords`` — consumes historical
    keywords + reference.py head, produces operator summary; landed on
    disk as ``agent_session/op_summary.md``.
  * LLM #2 ``_analyze_plan_md`` — consumes full plan.md, produces a
    strict 5-section structured analysis; landed on disk as
    ``agent_session/plan_analysis.md``.

The rebuilt buffer layout is now five distinct messages:
  [COMPACT_BOUNDARY]
  [BOOTSTRAP]                   — current plan vN + operator summary
  [STATE_ATTACHMENT:KERNEL]     — editable files, full content
  [STATE_ATTACHMENT:PLAN]       — structured analysis (LLM or fallback)
  [STATE_ATTACHMENT:RANKING]    — ranking.md full

When either LLM call fails, the pipeline degrades to a fallback string
(keyword dump / truncated raw plan.md) — the overall compact never
raises and always returns a valid new buffer.
"""
import os
from collections import Counter
from types import SimpleNamespace

import pytest

from akg_agents.op.autoresearch.agent import compress as cm
from akg_agents.op.autoresearch.agent.compress import (
    BOOTSTRAP_MARKER,
    COMPACT_BOUNDARY,
    OPERATOR_SUMMARY_MARKER,
    STATE_KERNEL,
    STATE_PLAN,
    STATE_RANKING,
    auto_compact,
    force_rebuild_minimal_context,
    _analyze_plan_md,
    _build_kernel_attachment,
    _collect_historical_keywords,
    _op_summary_fallback,
    _plan_analysis_fallback,
    _summarize_operator_from_keywords,
)


# -- test helpers ------------------------------------------------------------


class _FakeLLM:
    """LLM stub. Returns ``reply`` for every .call(); raises if
    ``raise_exc`` is set."""

    def __init__(self, reply="stub-llm-response", raise_exc=None):
        self.reply = reply
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def call(self, *, system_prompt, messages, tools=None,
                   compact=False, max_tokens=None, **kw):
        self.calls.append({
            "system": system_prompt, "messages": messages,
            "max_tokens": max_tokens,
        })
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"content": self.reply}

    def get_response_text(self, response):
        return response.get("content") or ""

    def get_stop_reason(self, response):
        return "end_turn"


def _make_config(editable_files=("kernel.py",), **agent_overrides):
    agent = SimpleNamespace(
        session_dir="agent_session",
        compact_op_summary_max_tokens=500,
        compact_plan_analysis_max_tokens=1500,
        compact_kernel_sanity_cap=80_000,
        compact_plan_raw_fallback_chars=6_000,
        compact_max_retries=1,
        compact_diagnosis_truncate=2_000,
        compact_keep_recent_rounds=2,
        compact_min_messages=4,
        chars_per_token=3,
        context_limit=150_000,
    )
    for k, v in agent_overrides.items():
        setattr(agent, k, v)
    return SimpleNamespace(
        name="demo_op",
        primary_metric="latency_us",
        lower_is_better=True,
        editable_files=list(editable_files),
        agent=agent,
    )


def _make_feedback(items=(), settled_keywords=()):
    """Feedback stub with the attributes compress.py actually reads."""
    plan_items = [
        {"id": f"p{i+1}", "text": f"item {i+1}", "status": "active" if i == 0 else "pending",
         "backing_skill": None, "keywords": list(kws)}
        for i, kws in enumerate(items)
    ]
    history = [
        {"item_id": f"h{i}", "plan_text": f"hist {i}", "text": f"edit {i}",
         "ok": False, "reason": "fail", "keywords": list(kws),
         "backing_skill": None, "version": 1, "signal_eligible": True}
        for i, kws in enumerate(settled_keywords)
    ]
    return SimpleNamespace(
        must_replan=False,
        last_diagnosis=None,
        phase="active" if items else "no_plan",
        plan_version=1,
        format_status=lambda: "",
        format_goal_anchor=lambda m: f"Goal: latency_us (current: {m}).",
        skill_builder=None,
        skill_pool=None,
        _plan_items=plan_items,
        _settled_history=history,
    )


def _seed_plan_md(task_dir: str, content: str):
    with open(os.path.join(task_dir, "plan.md"), "w", encoding="utf-8") as f:
        f.write(content)


def _seed_file(task_dir: str, name: str, content: str):
    path = os.path.join(task_dir, name)
    os.makedirs(os.path.dirname(path) or task_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# 1. Keyword aggregation
# ---------------------------------------------------------------------------


class TestKeywordAggregation:
    def test_dedups_and_counts_across_plan_items_and_history(self):
        fb = _make_feedback(
            items=[["matmul", "block_size"], ["matmul"]],
            settled_keywords=[["matmul", "grid"], ["autotune"]],
        )
        kws = _collect_historical_keywords(fb)
        # Counter view: matmul=3, autotune/block_size/grid=1 each.
        as_dict = dict(kws)
        assert as_dict["matmul"] == 3
        assert as_dict["block_size"] == 1
        assert as_dict["grid"] == 1
        assert as_dict["autotune"] == 1

    def test_sorts_by_frequency_then_alpha(self):
        fb = _make_feedback(items=[["z", "a"], ["a"]])
        kws = _collect_historical_keywords(fb)
        # 'a' appears 2× → first; 'z' appears 1× → second.
        assert kws[0] == ("a", 2)
        assert kws[1] == ("z", 1)

    def test_empty_feedback_returns_empty(self):
        assert _collect_historical_keywords(None) == []
        fb = _make_feedback()
        assert _collect_historical_keywords(fb) == []


# ---------------------------------------------------------------------------
# 2. LLM call #1 — operator summary
# ---------------------------------------------------------------------------


class TestOperatorSummary:
    @pytest.mark.asyncio
    async def test_success_persists_to_disk_and_returns_text(self, tmp_path):
        _seed_file(str(tmp_path), "reference.py", "def ref(): return 1")
        fb = _make_feedback(items=[["matmul"]])
        llm = _FakeLLM(reply="## Operator Shape\nMatmul-heavy.")
        config = _make_config()

        text = await _summarize_operator_from_keywords(
            llm, config, fb, str(tmp_path),
        )
        assert "Matmul-heavy" in text
        # Persisted under agent_session/op_summary.md
        path = tmp_path / "agent_session" / "op_summary.md"
        assert path.exists()
        assert "Matmul-heavy" in path.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_keyword_dump(self, tmp_path):
        fb = _make_feedback(
            items=[["matmul", "block_size"]],
            settled_keywords=[["matmul"]],
        )
        llm = _FakeLLM(raise_exc=RuntimeError("llm gone"))
        text = await _summarize_operator_from_keywords(
            llm, _make_config(), fb, str(tmp_path),
        )
        # Fallback string includes the keyword dump.
        assert "Keywords seen so far" in text
        assert "matmul" in text
        # Even the fallback is persisted.
        path = tmp_path / "agent_session" / "op_summary.md"
        assert path.exists()


# ---------------------------------------------------------------------------
# 3. LLM call #2 — plan.md structured analysis
# ---------------------------------------------------------------------------


class TestPlanAnalysis:
    @pytest.mark.asyncio
    async def test_success_persists_structured_analysis(self, tmp_path):
        _seed_plan_md(str(tmp_path),
                      "# Plan v3\n- [O] [p1] change X\n- [X] [p2] change Y")
        reply = (
            "## Current Status\nstatus text\n"
            "## What's Working\nworking text\n"
            "## High-ROI Operations\nROI text\n"
            "## Repeated Failures\nfailures text\n"
            "## Dead Directions\ndead text"
        )
        llm = _FakeLLM(reply=reply)
        text = await _analyze_plan_md(llm, _make_config(), str(tmp_path))
        assert "## Current Status" in text
        assert "## Dead Directions" in text
        # Persisted.
        path = tmp_path / "agent_session" / "plan_analysis.md"
        assert path.exists()
        body = path.read_text(encoding="utf-8")
        assert "## What's Working" in body

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_truncated_raw(self, tmp_path):
        _seed_plan_md(str(tmp_path), "# Plan v1\n" + "X" * 200)
        llm = _FakeLLM(raise_exc=RuntimeError("boom"))
        text = await _analyze_plan_md(
            llm, _make_config(compact_plan_raw_fallback_chars=100),
            str(tmp_path),
        )
        assert "analysis unavailable" in text
        # Truncated at fallback_chars.
        assert "[truncated]" in text

    @pytest.mark.asyncio
    async def test_missing_plan_md_returns_empty_placeholder(self, tmp_path):
        # No plan.md on disk.
        llm = _FakeLLM()
        text = await _analyze_plan_md(llm, _make_config(), str(tmp_path))
        assert "empty" in text.lower()


# ---------------------------------------------------------------------------
# 4. Attachment builders
# ---------------------------------------------------------------------------


class TestKernelAttachment:
    def test_full_content_passes_through_when_below_cap(self, tmp_path):
        _seed_file(str(tmp_path), "kernel.py", "x = 1\n" * 100)
        msgs = _build_kernel_attachment(str(tmp_path), _make_config())
        assert len(msgs) == 1
        content = msgs[0]["content"]
        assert STATE_KERNEL in content
        assert "## kernel.py" in content
        # No truncation marker.
        assert "WARNING: file truncated" not in content

    def test_sanity_cap_triggers_warning_on_huge_file(self, tmp_path):
        huge = "A" * 200_000  # > 80k default
        _seed_file(str(tmp_path), "kernel.py", huge)
        msgs = _build_kernel_attachment(str(tmp_path), _make_config())
        content = msgs[0]["content"]
        assert "WARNING: file truncated at sanity cap" in content
        # Cap respected: message body ≤ cap + a bit of overhead
        assert len(content) < 81_000 + 500

    def test_missing_file_drops_attachment(self, tmp_path):
        # No kernel.py on disk.
        msgs = _build_kernel_attachment(str(tmp_path), _make_config())
        assert msgs == []


# ---------------------------------------------------------------------------
# 5. auto_compact end-to-end
# ---------------------------------------------------------------------------


class TestAutoCompactEndToEnd:
    def _dummy_rounds(self, n_rounds):
        """Build n rounds of (assistant text + user tool result) pairs."""
        msgs = []
        for i in range(n_rounds):
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": f"round {i}"}]})
            msgs.append({"role": "user",
                         "content": [{"type": "tool_result",
                                      "tool_use_id": f"tu{i}",
                                      "content": f"result {i}"}]})
        return msgs

    @pytest.mark.asyncio
    async def test_buffer_layout_is_five_messages_plus_recent(self, tmp_path):
        _seed_file(str(tmp_path), "kernel.py", "k = 1")
        _seed_plan_md(str(tmp_path), "# Plan v1\n- [>>>] [p1] do X")
        _seed_file(str(tmp_path), "ranking.md", "## Performance Ranking\n1. R1")
        fb = _make_feedback(items=[["matmul"]])
        config = _make_config()
        llm = _FakeLLM(reply="## Operator Shape\nhello")
        old = self._dummy_rounds(6)

        new = await auto_compact(
            old, llm, str(tmp_path),
            config=config, tools=None, feedback=fb,
            keep_recent_rounds=2, best_metric_str="123",
        )

        # Must differ from input (=actually compacted).
        assert new is not old
        # First 5 marker messages in order.
        contents = [m["content"] for m in new[:5]]
        assert COMPACT_BOUNDARY in contents[0]
        assert BOOTSTRAP_MARKER in contents[1]
        assert STATE_KERNEL in contents[2]
        assert STATE_PLAN in contents[3]
        assert STATE_RANKING in contents[4]
        # Recent 2 rounds (4 msgs) tailing after.
        assert len(new) == 5 + 4

    @pytest.mark.asyncio
    async def test_noop_when_not_enough_rounds(self, tmp_path):
        fb = _make_feedback()
        config = _make_config()
        llm = _FakeLLM()
        old = self._dummy_rounds(2)
        new = await auto_compact(
            old, llm, str(tmp_path),
            config=config, tools=None, feedback=fb,
            keep_recent_rounds=2, best_metric_str="",
        )
        # Identity signal: same list returned unchanged.
        assert new is old

    @pytest.mark.asyncio
    async def test_both_llm_failures_still_returns_valid_buffer(self, tmp_path):
        _seed_file(str(tmp_path), "kernel.py", "k = 1")
        _seed_plan_md(str(tmp_path), "# Plan v1\nbody")
        fb = _make_feedback(items=[["matmul"]])
        config = _make_config()
        # Both LLM calls raise.
        llm = _FakeLLM(raise_exc=RuntimeError("llm down"))
        old = self._dummy_rounds(6)

        new = await auto_compact(
            old, llm, str(tmp_path),
            config=config, tools=None, feedback=fb,
            keep_recent_rounds=2, best_metric_str="",
        )
        # Did not raise. Buffer has the 5-section shape (ranking absent
        # since ranking.md wasn't seeded, so 4 leading messages).
        assert COMPACT_BOUNDARY in new[0]["content"]
        assert BOOTSTRAP_MARKER in new[1]["content"]
        # Plan attachment contains the fallback marker.
        plan_msgs = [m for m in new if STATE_PLAN in m["content"]]
        assert plan_msgs
        assert "analysis unavailable" in plan_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_bootstrap_contains_operator_summary_marker(self, tmp_path):
        _seed_file(str(tmp_path), "kernel.py", "k")
        _seed_plan_md(str(tmp_path), "# Plan v1\n- [>>>] [p1] do X")
        fb = _make_feedback(items=[["matmul"]])
        config = _make_config()
        llm = _FakeLLM(reply="Operator summary body here.")
        old = self._dummy_rounds(6)
        new = await auto_compact(
            old, llm, str(tmp_path),
            config=config, tools=None, feedback=fb,
            keep_recent_rounds=2, best_metric_str="",
        )
        bootstrap_content = new[1]["content"]
        assert OPERATOR_SUMMARY_MARKER in bootstrap_content


# ---------------------------------------------------------------------------
# 6. force_rebuild — no LLM
# ---------------------------------------------------------------------------


class TestForceRebuild:
    def test_force_rebuild_uses_keyword_fallback_no_llm_call(self, tmp_path):
        _seed_file(str(tmp_path), "kernel.py", "k = 1")
        _seed_plan_md(str(tmp_path), "# Plan v1\nbody")
        fb = _make_feedback(items=[["matmul", "block_size"]])
        config = _make_config()

        new = force_rebuild_minimal_context(
            str(tmp_path), config, feedback=fb,
            best_metric_str="42",
        )
        # Five-message layout (ranking.md missing → 4).
        assert COMPACT_BOUNDARY in new[0]["content"]
        bootstrap = new[1]["content"]
        # Operator section uses the keyword dump fallback.
        assert OPERATOR_SUMMARY_MARKER in bootstrap
        assert "Keywords seen so far" in bootstrap
        # Plan attachment uses the "analysis unavailable" fallback.
        plan_msgs = [m for m in new if STATE_PLAN in m["content"]]
        assert plan_msgs
        assert "analysis unavailable" in plan_msgs[0]["content"]
        # No LLM was invoked — we constructed without one.
        # (covered by not passing an llm at all)


# ---------------------------------------------------------------------------
# 7. Fallbacks (pure helpers)
# ---------------------------------------------------------------------------


class TestFallbacks:
    def test_op_summary_fallback_includes_keywords(self):
        out = _op_summary_fallback(
            "demo", [("matmul", 3), ("block", 1)],
        )
        assert "demo" in out
        assert "matmul×3" in out
        assert "block" in out

    def test_op_summary_fallback_handles_empty_keywords(self):
        out = _op_summary_fallback("demo", [])
        assert "No keywords recorded" in out

    def test_plan_analysis_fallback_truncates_and_labels(self):
        raw = "a" * 100
        out = _plan_analysis_fallback(raw, fallback_chars=30)
        assert "analysis unavailable" in out
        assert "[truncated]" in out


# -- Post-refactor fixes (rebuild-marker match, rebuild caps, ref_file) ------


class TestIsRebuildMessage:
    """``_is_rebuild_message`` must recognise every ``[STATE_ATTACHMENT:*]``
    variant. The earlier check used ``"[STATE_ATTACHMENT]"`` (bracketed)
    as a substring, which does NOT match ``"[STATE_ATTACHMENT:KERNEL]"``
    — meaning the ``strip previous rebuild markers`` pass in
    ``auto_compact`` leaked old attachments into the next compaction."""

    def _msg(self, content):
        return {"role": "user", "content": content}

    def test_kernel_attachment_is_rebuild(self):
        from akg_agents.op.autoresearch.agent.compress import (
            STATE_KERNEL, _is_rebuild_message,
        )
        assert _is_rebuild_message(self._msg(f"{STATE_KERNEL}\n## k.py\n..."))

    def test_plan_attachment_is_rebuild(self):
        from akg_agents.op.autoresearch.agent.compress import (
            STATE_PLAN, _is_rebuild_message,
        )
        assert _is_rebuild_message(self._msg(f"{STATE_PLAN}\n## plan"))

    def test_ranking_attachment_is_rebuild(self):
        from akg_agents.op.autoresearch.agent.compress import (
            STATE_RANKING, _is_rebuild_message,
        )
        assert _is_rebuild_message(self._msg(f"{STATE_RANKING}\n## ranking"))

    def test_bootstrap_and_boundary_still_recognised(self):
        from akg_agents.op.autoresearch.agent.compress import (
            BOOTSTRAP_MARKER, COMPACT_BOUNDARY, _is_rebuild_message,
        )
        assert _is_rebuild_message(self._msg(f"{COMPACT_BOUNDARY}\nfoo"))
        assert _is_rebuild_message(self._msg(f"{BOOTSTRAP_MARKER}\nfoo"))

    def test_normal_user_message_not_rebuild(self):
        from akg_agents.op.autoresearch.agent.compress import (
            _is_rebuild_message,
        )
        assert not _is_rebuild_message(self._msg("regular agent feedback"))

    def test_auto_compact_strips_all_variants_of_prior_rebuild(self):
        """End-to-end: the ``strip previous rebuild markers`` line
        in ``auto_compact`` must drop KERNEL / PLAN / RANKING
        attachments from the previous compact so they don't double up
        in the next rebuilt buffer."""
        from akg_agents.op.autoresearch.agent.compress import (
            BOOTSTRAP_MARKER, COMPACT_BOUNDARY, STATE_KERNEL, STATE_PLAN,
            STATE_RANKING, _find_last_boundary, _is_rebuild_message,
        )
        msgs = [
            {"role": "user", "content": "initial task"},
            {"role": "user", "content": f"{COMPACT_BOUNDARY}\nprev"},
            {"role": "user", "content": f"{BOOTSTRAP_MARKER}\nboot"},
            {"role": "user", "content": f"{STATE_KERNEL}\n## k.py\nold"},
            {"role": "user", "content": f"{STATE_PLAN}\n## plan\nold"},
            {"role": "user", "content": f"{STATE_RANKING}\n## r.md\nold"},
            {"role": "assistant", "content": "r1"},
            {"role": "user", "content": "eval 1"},
        ]
        bi = _find_last_boundary(msgs)
        incremental = msgs[bi + 1:] if bi is not None else msgs
        live = [m for m in incremental if not _is_rebuild_message(m)]
        assert [m["role"] for m in live] == ["assistant", "user"]
        assert not any(
            "[STATE_ATTACHMENT" in m.get("content", "")
            for m in live if isinstance(m.get("content"), str)
        )


class TestForceRebuildCaps:
    """force_rebuild is the PTL escape hatch. Its rebuilt buffer MUST
    be strictly smaller than the one that tripped PTL, otherwise the
    recovery loops. Normal auto_compact path keeps the full 80k
    kernel cap + uncapped ranking.md; force_rebuild overrides both."""

    def test_ranking_attachment_uncapped_by_default(self, tmp_path):
        from akg_agents.op.autoresearch.agent.compress import (
            _build_ranking_attachment,
        )
        (tmp_path / "ranking.md").write_text("X" * 50_000, encoding="utf-8")
        msgs = _build_ranking_attachment(str(tmp_path))
        assert len(msgs) == 1
        assert "force_rebuild" not in msgs[0]["content"]
        assert msgs[0]["content"].count("X") == 50_000

    def test_ranking_attachment_honours_max_chars(self, tmp_path):
        from akg_agents.op.autoresearch.agent.compress import (
            _build_ranking_attachment,
        )
        (tmp_path / "ranking.md").write_text("Y" * 50_000, encoding="utf-8")
        msgs = _build_ranking_attachment(str(tmp_path), max_chars=1_000)
        assert len(msgs) == 1
        body = msgs[0]["content"]
        assert body.count("Y") <= 1_000
        assert "truncated at 1000 chars during force_rebuild" in body

    def test_kernel_attachment_honours_override(self, tmp_path):
        from types import SimpleNamespace
        from akg_agents.op.autoresearch.agent.compress import (
            _build_kernel_attachment,
        )
        (tmp_path / "k.py").write_text("Z" * 50_000, encoding="utf-8")
        cfg = SimpleNamespace(
            editable_files=["k.py"],
            agent=SimpleNamespace(compact_kernel_sanity_cap=80_000),
        )
        msgs = _build_kernel_attachment(
            str(tmp_path), cfg, max_chars_override=5_000,
        )
        assert len(msgs) == 1
        body = msgs[0]["content"]
        assert body.count("Z") <= 5_000


class TestOpSummaryRespectsRefFile:
    """AgentConfig.ref_file lets non-scaffolder tasks place the
    reference code somewhere other than ``reference.py``. The
    operator-summary helper used to hardcode ``reference.py`` and
    silently drop the reference context for those tasks."""

    def _run(self, tmp_path, cfg, captured):
        import asyncio
        import os
        from types import SimpleNamespace
        from akg_agents.op.autoresearch.agent.compress import (
            _summarize_operator_from_keywords,
        )

        class _LLM:
            async def call(self, *, system_prompt, messages, tools,
                           compact, max_tokens):
                captured["user_prompt"] = messages[0]["content"]
                return {"content": "## Operator Shape\nok"}

            @staticmethod
            def get_response_text(r):
                return r.get("content") or ""

        fb = SimpleNamespace(_plan_items=[], _settled_history=[])
        os.makedirs(tmp_path / "agent_session", exist_ok=True)
        asyncio.run(_summarize_operator_from_keywords(
            _LLM(), cfg, fb, str(tmp_path),
        ))

    def test_reads_configured_ref_file(self, tmp_path):
        """``ref_file`` lives on ``TaskConfig`` (config root), NOT on
        ``AgentConfig``. This test pins the correct surface so a
        future regression that reads it off ``config.agent`` fails."""
        from types import SimpleNamespace
        (tmp_path / "custom_ref.py").write_text(
            "# CUSTOM REF MARKER\nx = 1\n", encoding="utf-8",
        )
        captured = {}
        cfg = SimpleNamespace(
            name="probe",
            ref_file="custom_ref.py",   # <-- top level, TaskConfig surface
            agent=SimpleNamespace(
                session_dir="agent_session",
                compact_op_summary_max_tokens=500,
                compact_max_retries=0,
            ),
        )
        self._run(tmp_path, cfg, captured)
        assert "CUSTOM REF MARKER" in captured.get("user_prompt", "")

    def test_ignores_ref_file_on_agent_surface(self, tmp_path):
        """Regression guard: if ref_file is mistakenly placed on
        ``config.agent`` (old mis-fix), the code must NOT pick it up.
        The reference.py fallback should fire instead."""
        from types import SimpleNamespace
        # Only reference.py exists on disk.
        (tmp_path / "reference.py").write_text(
            "# FALLBACK REF\n", encoding="utf-8",
        )
        # ``custom_ref.py`` exists too but it's only referenced off the
        # wrong (agent) surface — should be ignored.
        (tmp_path / "custom_ref.py").write_text(
            "# MISPLACED MARKER\n", encoding="utf-8",
        )
        captured = {}
        cfg = SimpleNamespace(
            name="probe",
            # ref_file deliberately NOT set on top level
            agent=SimpleNamespace(
                ref_file="custom_ref.py",   # <-- wrong surface; must be ignored
                session_dir="agent_session",
                compact_op_summary_max_tokens=500,
                compact_max_retries=0,
            ),
        )
        self._run(tmp_path, cfg, captured)
        assert "FALLBACK REF" in captured.get("user_prompt", "")
        assert "MISPLACED MARKER" not in captured.get("user_prompt", "")

    def test_falls_back_to_reference_py_when_ref_file_unset(self, tmp_path):
        from types import SimpleNamespace
        (tmp_path / "reference.py").write_text(
            "# LEGACY REF\n", encoding="utf-8",
        )
        captured = {}
        cfg = SimpleNamespace(
            name="probe",
            # ref_file unset anywhere
            agent=SimpleNamespace(
                session_dir="agent_session",
                compact_op_summary_max_tokens=500,
                compact_max_retries=0,
            ),
        )
        self._run(tmp_path, cfg, captured)
        assert "LEGACY REF" in captured.get("user_prompt", "")
