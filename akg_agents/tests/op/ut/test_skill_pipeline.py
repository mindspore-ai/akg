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

"""Unit tests for the pure skill-pipeline primitives in ``skill_adapter``.

This file covers the stateless, side-effect-free building blocks:

  - ``QueryKeywords`` dataclass
  - ``extract_fallback_keywords``
  - ``rank_skills_by_keywords``
  - ``render_skills_markdown`` / ``render_ranked_skill_block``
  - ``_parse_and_validate_query_keywords``
  - ``generate_query_keywords`` (with a stub LLM)
  - ``ConversationAdapter._retry_with_backoff`` anti-regression
  - source-level invariants (no kernel_gen import) + the module-level
    catalog cache

The hint-adapter flow tests (``select_and_render_skills_for_hint`` end-to-end,
stub catalog, fixtures) live in ``test_hint_skill_adapter.py`` — they will
be renamed to ``test_supervisor.py`` when the hint path is removed.
"""

import inspect
from types import SimpleNamespace

import pytest

from akg_agents.op.autoresearch.agent import skill_adapter as sa
from akg_agents.op.autoresearch.agent import skill_rendering as sr
from akg_agents.op.autoresearch.agent.llm_client import ConversationAdapter
from akg_agents.op.autoresearch.agent.skill_adapter import (
    QueryKeywords,
    extract_fallback_keywords,
    generate_query_keywords,
    rank_skills_by_keywords,
)
from akg_agents.op.autoresearch.agent.skill_pool import SkillPool
from akg_agents.op.autoresearch.agent.skill_rendering import (
    render_ranked_skill_block,
    render_skills_markdown,
)


class _FakeSkillBuilder:
    """Minimal SkillBuilder stand-in for SkillPool tests.

    Post-abandon refactor: every registered skill is bindable — no
    terminal-state predicate, no exclude-set query. SkillPool only
    needs ``register`` (refill write side) and ``get`` (for
    ``_tier_of`` when match_by_keywords sorts results). This stub
    returns a minimal record-like object with a stable tier() so the
    tests that don't care about tiering stay readable.
    """

    def __init__(self):
        self.marked = []
        self._records: dict[str, object] = {}

    def register(self, skill, *, reason, plan_version):
        name = getattr(skill, "name", "")
        self.marked.append((name, reason, plan_version))
        self._records[name] = type("_Rec", (), {
            "tier": staticmethod(lambda: 1),
            "applied_versions": [],
            "unbound_at_versions": [],
        })

    def get(self, name):
        return self._records.get(name)

    def __contains__(self, name):
        return name in self._records


class _FakeSkill:
    def __init__(
        self,
        name,
        category,
        content="",
        metadata=None,
        description=None,
        skill_path=None,
    ):
        self.name = name
        self.category = category
        self.content = content or f"<content of {name}>"
        self.metadata = metadata or {}
        self.description = description or f"{name} description"
        self.skill_path = skill_path


class _StubLLM:
    def __init__(self, *, text="", exc=None, delay=0.0):
        self.text = text
        self.exc = exc
        self.delay = delay
        self.calls = []

    async def call(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            import asyncio
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        return {"content": self.text}

    def get_response_text(self, response):
        return response.get("content", "")


class TestQueryKeywords:
    def test_defaults(self):
        keywords = QueryKeywords()
        assert keywords.must_have == []
        assert keywords.nice_to_have == []
        assert keywords.avoid == []
        assert keywords.reasoning == ""

    def test_is_empty(self):
        assert QueryKeywords().is_empty() is True
        assert QueryKeywords(must_have=["matmul"]).is_empty() is False
        assert QueryKeywords(nice_to_have=["tiling"]).is_empty() is False
        assert QueryKeywords(avoid=["scan"]).is_empty() is True

    def test_merged_with_dedups(self):
        left = QueryKeywords(
            must_have=["matmul", "tl.dot"],
            nice_to_have=["tiling"],
            avoid=["scan"],
            reasoning="left",
        )
        right = QueryKeywords(
            must_have=["matmul", "BLOCK_M"],
            nice_to_have=["tiling", "swizzle"],
            avoid=["scan", "reduce"],
            reasoning="right",
        )
        merged = left.merged_with(right)
        assert merged.must_have == ["matmul", "tl.dot", "BLOCK_M"]
        assert merged.nice_to_have == ["tiling", "swizzle"]
        assert merged.avoid == ["scan", "reduce"]
        assert merged.reasoning == "left"


class TestRenderSkillsMarkdown:
    def test_empty_input_returns_empty_string(self):
        assert render_skills_markdown([]) == ""

    def test_method_and_implementation_render_in_full_mode(self):
        skills = [
            _FakeSkill("m", "method", content="method body"),
            _FakeSkill("impl", "implementation", content="impl body"),
        ]
        out = render_skills_markdown(skills)
        assert "method body" in out
        assert "impl body" in out

    def test_index_mode_includes_description_and_path(self):
        skill = _FakeSkill(
            "matmul",
            "implementation",
            description="first line\nsecond line",
            skill_path=sr._PROJECT_ROOT / "foo" / "bar" / "SKILL.md",
        )
        out = render_skills_markdown([skill], mode="index")
        assert "## Skill Index" in out
        assert "matmul [implementation]" in out
        # Paths are now task-dir-relative (skills/<name>/SKILL.md);
        # the project-root catalog path is no longer surfaced.
        assert "skills/matmul/SKILL.md" in out
        assert "foo/bar/SKILL.md" not in out
        assert "first line" in out
        assert "second line" not in out

    def test_max_chars_truncates(self):
        skill = _FakeSkill("g", "guide", content="x" * 200)
        out = render_skills_markdown([skill], max_chars=80)
        assert len(out) <= 80
        assert out.endswith("[truncated]")


class TestExtractFallbackKeywords:
    def test_op_name_seeds_must_have(self):
        result = extract_fallback_keywords(
            "MatMulKernel",
            "Use tl.dot and block tiling for the matmul kernel",
        )
        assert result.must_have
        assert "mat" in result.must_have or "matmulkernel" in result.must_have

    def test_filters_stopwords_and_banned_keywords(self):
        result = extract_fallback_keywords(
            "matmul",
            "performance optimization performance tiling tiling tiling 优化",
        )
        assert "performance" not in result.must_have
        assert "optimization" not in result.must_have
        assert "优化" not in result.must_have
        assert "tiling" in result.must_have or "tiling" in result.nice_to_have

    def test_handles_chinese_and_english(self):
        result = extract_fallback_keywords(
            "矩阵乘",
            "矩阵乘 matmul tile swizzle",
        )
        assert "矩阵乘" in result.must_have
        assert any(
            token in {"matmul", "tile", "swizzle"}
            for token in result.must_have + result.nice_to_have
        )


class TestRankSkillsByKeywords:
    def test_must_have_beats_nice_to_have(self):
        skills = [
            _FakeSkill("matmul-guide", "guide", description="matmul tl.dot tiling"),
            _FakeSkill("generic-guide", "guide", description="tiling only"),
        ]
        ranked = rank_skills_by_keywords(
            skills,
            QueryKeywords(must_have=["matmul"], nice_to_have=["tiling"]),
            op_name="matmul",
        )
        assert ranked[0][1].name == "matmul-guide"
        assert ranked[0][0] > ranked[1][0]

    def test_avoid_penalty_applies(self):
        skills = [
            _FakeSkill("bad", "guide", description="scan reduction"),
            _FakeSkill("good", "guide", description="matmul tiling"),
        ]
        ranked = rank_skills_by_keywords(
            skills,
            QueryKeywords(must_have=["matmul"], avoid=["scan"]),
            op_name="matmul",
        )
        assert ranked[0][1].name == "good"

    def test_debug_stage_prefers_cases(self):
        skills = [
            _FakeSkill("case-fix", "case", description="compile error mask"),
            _FakeSkill("guide", "guide", description="compile error mask"),
        ]
        ranked = rank_skills_by_keywords(
            skills,
            QueryKeywords(must_have=["mask"]),
            stage="debug",
            op_name="mask",
        )
        assert ranked[0][1].name == "case-fix"

    def test_empty_keywords_falls_back_to_prior(self):
        skills = [
            _FakeSkill("guide", "guide"),
            _FakeSkill("fundamental", "fundamental"),
        ]
        ranked = rank_skills_by_keywords(skills, QueryKeywords())
        assert ranked[0][1].name == "guide"


def _pool(*skills):
    """Construct a SkillPool whose internal ranked list is exactly
    the given skills. Used by every read-side test below.

    Post-abandon refactor: the terminal ``abandoned`` exclusion is
    gone, so there's no need for the old ``abandoned=`` parameter.
    Callers that used to rely on 'excluded by terminal state' should
    instead assert on tier() behaviour via ``match_by_keywords``.
    """
    pool = SkillPool(_FakeSkillBuilder())
    pool.replace(list(skills))
    return pool


class TestSkillPoolResidualBindable:
    """Pin the sliding-window bind-eligibility query."""

    def test_keeps_only_trackable_categories(self):
        pool = _pool(
            _FakeSkill("guide-A", "guide"),
            _FakeSkill("example-B", "example"),
            _FakeSkill("fund-C", "fundamental"),
        )
        bindable = pool.residual_bindable(top_k=3)
        assert [s.name for s in bindable] == ["guide-A", "example-B"]

    def test_sliding_window_walks_past_non_trackable_prefix(self):
        """Old behavior was ``ranked[:top_k]`` then filter, so a
        non-trackable prefix would shrink the result. New sliding-
        window walks the full list and stops at K matches."""
        pool = _pool(
            _FakeSkill("fund-A", "fundamental"),
            _FakeSkill("ref-B", "reference"),
            _FakeSkill("fund-C", "fundamental"),
            _FakeSkill("guide-D", "guide"),
            _FakeSkill("example-E", "example"),
            _FakeSkill("guide-F", "guide"),
        )
        bindable = pool.residual_bindable(top_k=2)
        assert [s.name for s in bindable] == ["guide-D", "example-E"]

    def test_stops_at_top_k_matches(self):
        pool = _pool(
            _FakeSkill("guide-A", "guide"),
            _FakeSkill("example-B", "example"),
            _FakeSkill("guide-C", "guide"),
        )
        bindable = pool.residual_bindable(top_k=2)
        assert [s.name for s in bindable] == ["guide-A", "example-B"]
        assert "guide-C" not in [s.name for s in bindable]

    def test_residual_bindable_keeps_every_trackable_post_refactor(self):
        """Post-abandon refactor: there is no terminal state.
        ``residual_bindable`` returns every trackable-category entry —
        previously-unbound skills are still selectable here. The
        tiering happens downstream in ``match_by_keywords``."""
        pool = _pool(
            _FakeSkill("guide-A", "guide"),
            _FakeSkill("example-B", "example"),
            _FakeSkill("case-C", "case"),
        )
        bindable = pool.residual_bindable(top_k=5)
        names = [s.name for s in bindable]
        assert set(names) == {"guide-A", "example-B", "case-C"}

    def test_dedupes_duplicate_names(self):
        dup = _FakeSkill("guide-A", "guide")
        pool = _pool(dup, dup)
        bindable = pool.residual_bindable(top_k=5)
        assert [s.name for s in bindable] == ["guide-A"]

    def test_empty_pool_returns_empty(self):
        pool = _pool()
        assert pool.residual_bindable(top_k=5) == []

    def test_top_k_zero_returns_empty(self):
        pool = _pool(_FakeSkill("guide-A", "guide"))
        assert pool.residual_bindable(top_k=0) == []

    def test_preserves_rank_order(self):
        pool = _pool(
            _FakeSkill("zzz-guide", "guide"),
            _FakeSkill("aaa-example", "example"),
            _FakeSkill("mmm-case", "case"),
        )
        bindable = pool.residual_bindable(top_k=5)
        assert [s.name for s in bindable] == [
            "zzz-guide", "aaa-example", "mmm-case",
        ]


class TestSkillPoolContainerProtocol:
    """`__iter__`, `__len__`, `top_k(n)`, `is_empty` are read by
    rendering / supervisor / compress code."""

    def test_iterate_yields_ranked_order(self):
        a = _FakeSkill("a", "guide")
        b = _FakeSkill("b", "example")
        pool = _pool(a, b)
        assert list(pool) == [a, b]

    def test_len_matches_ranked(self):
        pool = _pool(_FakeSkill("a", "guide"), _FakeSkill("b", "example"))
        assert len(pool) == 2

    def test_top_k_unfiltered_slice(self):
        """``top_k`` returns the raw rank window with NO eligibility
        filter — used by rendering. ``residual_bindable`` is the
        bind-side query."""
        pool = _pool(
            _FakeSkill("a", "guide"),
            _FakeSkill("fund-b", "fundamental"),
            _FakeSkill("c", "example"),
        )
        # `top_k(3)` keeps all 3 — no category filter, no exclude.
        names = [s.name for s in pool.top_k(3)]
        assert names == ["a", "fund-b", "c"]

    def test_is_empty(self):
        assert _pool().is_empty() is True
        assert _pool(_FakeSkill("a", "guide")).is_empty() is False

    def test_replace_then_append_new(self):
        pool = SkillPool(_FakeSkillBuilder())
        pool.replace([_FakeSkill("a", "guide")])
        assert len(pool) == 1
        added = pool.append_new([
            _FakeSkill("a", "guide"),  # dup, dropped
            _FakeSkill("b", "example"),
        ])
        assert [s.name for s in added] == ["b"]
        assert [s.name for s in pool] == ["a", "b"]


class TestParseAndValidateQueryKeywords:
    def test_clean_xml(self):
        parsed = sa._parse_and_validate_query_keywords(
            "<keywords>"
            "<must_have>matmul</must_have>"
            "<nice_to_have>tiling</nice_to_have>"
            "<avoid>scan</avoid>"
            "<reasoning>ok</reasoning>"
            "</keywords>"
        )
        assert parsed == QueryKeywords(
            must_have=["matmul"],
            nice_to_have=["tiling"],
            avoid=["scan"],
            reasoning="ok",
        )

    def test_comma_separated_tokens_within_tag(self):
        parsed = sa._parse_and_validate_query_keywords(
            "<must_have>matmul, tl.dot, BLOCK_M</must_have>"
            "<nice_to_have>tiling, swizzle</nice_to_have>"
        )
        assert parsed.must_have == ["matmul", "tl.dot", "block_m"]
        assert parsed.nice_to_have == ["tiling", "swizzle"]

    def test_fenced_xml(self):
        parsed = sa._parse_and_validate_query_keywords(
            "```xml\n<keywords><must_have>matmul</must_have></keywords>\n```"
        )
        assert parsed.must_have == ["matmul"]

    def test_prose_wrapped_xml(self):
        parsed = sa._parse_and_validate_query_keywords(
            "Here is the block: "
            "<must_have>matmul</must_have>"
            "<nice_to_have>tiling</nice_to_have>"
            " thanks"
        )
        assert parsed.must_have == ["matmul"]
        assert parsed.nice_to_have == ["tiling"]

    def test_repeated_tags_concatenate(self):
        parsed = sa._parse_and_validate_query_keywords(
            "<must_have>matmul</must_have>"
            "<must_have>tl.dot</must_have>"
        )
        assert parsed.must_have == ["matmul", "tl.dot"]

    def test_no_xml_tags_returns_none(self):
        assert sa._parse_and_validate_query_keywords("just prose, no tags") is None

    def test_missing_must_have_returns_none(self):
        assert sa._parse_and_validate_query_keywords(
            "<nice_to_have>tiling</nice_to_have>"
        ) is None

    def test_empty_must_have_returns_none(self):
        assert sa._parse_and_validate_query_keywords(
            "<must_have></must_have><nice_to_have>tiling</nice_to_have>"
        ) is None

    def test_optional_tags_default_empty(self):
        parsed = sa._parse_and_validate_query_keywords(
            "<must_have>matmul</must_have>"
        )
        assert parsed.must_have == ["matmul"]
        assert parsed.nice_to_have == []
        assert parsed.avoid == []
        assert parsed.reasoning == ""

    def test_filters_duplicates_banned_and_empty_results(self):
        assert sa._parse_and_validate_query_keywords(
            "<must_have>performance, performance</must_have>"
        ) is None


class TestGenerateQueryKeywords:
    @pytest.mark.asyncio
    async def test_valid_llm_response(self):
        llm = _StubLLM(
            text=(
                "<keywords>"
                "<must_have>matmul</must_have>"
                "<nice_to_have>tiling</nice_to_have>"
                "</keywords>"
            )
        )
        result = await generate_query_keywords(
            "matmul",
            "optimize matmul",
            llm=llm,
            timeout=0.5,
        )
        assert result.must_have == ["matmul"]
        assert result.nice_to_have == ["tiling"]
        assert llm.calls[0]["max_retries"] == 0
        assert llm.calls[0]["tools"] == []

    @pytest.mark.asyncio
    async def test_exception_falls_back(self):
        llm = _StubLLM(exc=RuntimeError("boom"))
        expected = extract_fallback_keywords("matmul", "tile tile tile")
        result = await generate_query_keywords(
            "matmul",
            "tile tile tile",
            llm=llm,
            timeout=0.5,
        )
        assert result == expected

    @pytest.mark.asyncio
    async def test_invalid_response_falls_back(self):
        llm = _StubLLM(text="no tags here, just prose")
        expected = extract_fallback_keywords("matmul", "tile tile tile")
        result = await generate_query_keywords(
            "matmul",
            "tile tile tile",
            llm=llm,
            timeout=0.5,
        )
        assert result == expected

    @pytest.mark.asyncio
    async def test_timeout_falls_back(self):
        llm = _StubLLM(
            text="<must_have>matmul</must_have>",
            delay=0.2,
        )
        expected = extract_fallback_keywords("matmul", "tile tile tile")
        result = await generate_query_keywords(
            "matmul",
            "tile tile tile",
            llm=llm,
            timeout=0.01,
        )
        assert result == expected


class TestConversationAdapterRetry:
    @pytest.mark.asyncio
    async def test_max_retries_zero_still_invokes_fn_once(self):
        try:
            import openai  # noqa: F401
        except ImportError:
            pytest.skip("openai package not installed")

        adapter = ConversationAdapter.__new__(ConversationAdapter)
        adapter.provider = "openai"
        adapter.verbose = False
        adapter.model = "test-model"
        adapter._max_retries = 5
        adapter._retry_initial_backoff = 0.0
        adapter._retry_max_backoff_rate_limit = 0.0
        adapter._retry_max_backoff_other = 0.0

        call_count = [0]
        sentinel = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"must_have":["matmul"]}'))]
        )

        async def fake_fn():
            call_count[0] += 1
            return sentinel

        result = await adapter._retry_with_backoff(fake_fn, max_retries=0)
        assert call_count[0] == 1
        assert result is sentinel


class TestRenderRankedSkillBlock:
    def test_full_plus_index_appendix(self):
        skills = [
            _FakeSkill("a", "guide"),
            _FakeSkill("b", "guide"),
            _FakeSkill("c", "guide"),
        ]
        out = render_ranked_skill_block(
            skills,
            top_k=1,
            max_chars=200,
        )
        assert "<content of a>" in out
        assert "## Skill Index" in out
        assert "b [guide]" in out
        assert "c [guide]" in out


class TestSourceLevelInvariants:
    def test_skill_adapter_does_not_import_kernel_gen(self):
        src = inspect.getsource(sa)
        assert "from akg_agents.op.agents.kernel_gen" not in src
        assert "KernelGen" not in src

    def test_old_overlap_helpers_are_gone(self):
        assert not hasattr(sa, "_tokenize_for_overlap")
        assert not hasattr(sa, "_score_case_by_overlap")


class TestModuleLevelCatalogCache:
    def test_get_catalog_lazy_construct(self):
        """_get_catalog is a functools.lru_cache; repeated calls share
        the same OperatorSkillCatalog instance."""
        sa._get_catalog.cache_clear()
        try:
            c1 = sa._get_catalog()
            c2 = sa._get_catalog()
            assert c1 is c2
        finally:
            sa._get_catalog.cache_clear()
