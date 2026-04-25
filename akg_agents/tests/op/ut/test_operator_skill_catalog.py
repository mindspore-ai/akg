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
Tests for ``op.skill.OperatorSkillCatalog`` — the shared per-DSL skill
loader + filter + cache + classify + render facade above ``SkillLoader``
and ``OperatorSkillSelector``.

This module is the missing public component the autoresearch hint
path now uses (instead of subclassing kernel_gen). These tests pin
its API and behavioral contracts so any future migration of
kernel_gen / kernel_designer to the same catalog has a stable target.
"""

from pathlib import Path

import pytest

from akg_agents.op.skill import OperatorSkillCatalog


# ---------------------------------------------------------------------------
# Fakes — minimal Skill stand-in
# ---------------------------------------------------------------------------


class _FakeSkill:
    """Minimal stand-in mimicking ``core_v2.skill.SkillMetadata``.

    Catalog reads ``name``, ``category``, ``content``, ``metadata``,
    ``skill_path``. We populate just those.
    """

    def __init__(self, name, category, content="", metadata=None,
                 skill_path=None):
        self.name = name
        self.category = category
        self.content = content or f"<content of {name}>"
        self.metadata = metadata or {}
        self.skill_path = skill_path
        self.description = f"{name} description"


# ---------------------------------------------------------------------------
# classify_case_type — pure static, no setup needed
# ---------------------------------------------------------------------------


class TestClassifyCaseType:
    """Mirrors kernel_gen._infer_case_type's priority order:
    metadata.case_type > metadata.source > skill_path > default."""

    def test_explicit_metadata_case_type_fix(self):
        s = _FakeSkill("c1", "case", metadata={"case_type": "fix"})
        assert OperatorSkillCatalog.classify_case_type(s) == "fix"

    def test_explicit_metadata_case_type_improvement(self):
        s = _FakeSkill("c2", "case", metadata={"case_type": "improvement"})
        assert OperatorSkillCatalog.classify_case_type(s) == "improvement"

    def test_invalid_case_type_falls_through(self):
        """metadata.case_type values other than 'fix'/'improvement'
        should not be honored — fall through to next signal."""
        s = _FakeSkill("c3", "case",
                       metadata={"case_type": "weird", "source": "error_fix"})
        assert OperatorSkillCatalog.classify_case_type(s) == "fix"

    def test_metadata_source_error_fix(self):
        s = _FakeSkill("c4", "case", metadata={"source": "error_fix"})
        assert OperatorSkillCatalog.classify_case_type(s) == "fix"

    def test_path_based_fix(self):
        s = _FakeSkill("c5", "case",
                       skill_path=Path("/skills/evolved-fix/foo.md"))
        assert OperatorSkillCatalog.classify_case_type(s) == "fix"

    def test_path_based_improvement(self):
        s = _FakeSkill("c6", "case",
                       skill_path=Path("/skills/evolved-improvement/bar.md"))
        assert OperatorSkillCatalog.classify_case_type(s) == "improvement"

    def test_default_is_improvement(self):
        s = _FakeSkill("c7", "case")
        assert OperatorSkillCatalog.classify_case_type(s) == "improvement"

    def test_handles_missing_metadata_attribute(self):
        """Defensive: if a skill object doesn't have a metadata attr at
        all, classifier should default to 'improvement', not raise."""
        class _Bare:
            pass
        assert OperatorSkillCatalog.classify_case_type(_Bare()) == "improvement"


# ---------------------------------------------------------------------------
# render_as_markdown — pure class method, no instance state
# ---------------------------------------------------------------------------


class TestRenderAsMarkdown:

    def test_empty_input_returns_empty_string(self):
        assert OperatorSkillCatalog.render_as_markdown([]) == ""

    def test_single_fundamental_renders_basics_section(self):
        s = _FakeSkill("basics", "fundamental", content="basics body")
        out = OperatorSkillCatalog.render_as_markdown([s])
        assert "### 基础知识与规范" in out
        assert "basics body" in out

    def test_single_guide_renders_guide_section(self):
        s = _FakeSkill("g", "guide", content="guide body")
        out = OperatorSkillCatalog.render_as_markdown([s])
        assert "### 算子优化指南" in out
        assert "guide body" in out

    def test_single_example_renders_example_section(self):
        s = _FakeSkill("e", "example", content="ex body")
        out = OperatorSkillCatalog.render_as_markdown([s])
        assert "### 代码示例参考" in out
        assert "ex body" in out

    def test_single_case_renders_case_section(self):
        s = _FakeSkill("c", "case", content="case body")
        out = OperatorSkillCatalog.render_as_markdown([s])
        assert "### 优化/修复案例" in out
        assert "case body" in out

    def test_section_order_matches_category_layer(self):
        """Sections appear in the order: fundamental/reference →
        guide → example → case, regardless of input order."""
        skills = [
            _FakeSkill("c1", "case", content="case content"),
            _FakeSkill("g1", "guide", content="guide content"),
            _FakeSkill("f1", "fundamental", content="fund content"),
            _FakeSkill("e1", "example", content="ex content"),
        ]
        out = OperatorSkillCatalog.render_as_markdown(skills)
        # Check the relative order of section headers
        pos_fund = out.index("### 基础知识与规范")
        pos_guide = out.index("### 算子优化指南")
        pos_ex = out.index("### 代码示例参考")
        pos_case = out.index("### 优化/修复案例")
        assert pos_fund < pos_guide < pos_ex < pos_case

    def test_fundamental_and_reference_share_basics_section(self):
        """Both fundamental and reference categories render under one
        '基础知识与规范' header (matches kernel_gen's behavior)."""
        skills = [
            _FakeSkill("f", "fundamental", content="F"),
            _FakeSkill("r", "reference", content="R"),
        ]
        out = OperatorSkillCatalog.render_as_markdown(skills)
        # Only one basics header
        assert out.count("### 基础知识与规范") == 1
        # Both contents present
        assert "F" in out
        assert "R" in out

    def test_unknown_category_dropped_from_render(self):
        """Categories not in CATEGORY_LAYER (e.g., 'workflow') are
        excluded from the rendered output."""
        skills = [
            _FakeSkill("g", "guide", content="kept"),
            _FakeSkill("w", "workflow", content="dropped"),
        ]
        out = OperatorSkillCatalog.render_as_markdown(skills)
        assert "kept" in out
        assert "dropped" not in out

    def test_multiple_in_same_category_joined_with_separator(self):
        skills = [
            _FakeSkill("g1", "guide", content="A"),
            _FakeSkill("g2", "guide", content="B"),
        ]
        out = OperatorSkillCatalog.render_as_markdown(skills)
        # Both contents present, separated by ---
        assert "A" in out
        assert "B" in out
        assert "---" in out


# ---------------------------------------------------------------------------
# load_by_dsl — caching + missing-dir handling
# ---------------------------------------------------------------------------


class TestLoadByDsl:

    def test_missing_dsl_directory_returns_empty_and_caches(self, monkeypatch):
        """A DSL directory that doesn't exist on disk returns [] and
        caches the empty result so subsequent calls don't re-stat."""
        from akg_agents.op.skill import operator_skill_catalog as osc

        # Point SKILLS_DIR at a tmp_path-ish nowhere
        monkeypatch.setattr(osc, "SKILLS_DIR", Path("/nonexistent/skills/root"))

        catalog = OperatorSkillCatalog()
        result1 = catalog.load_by_dsl("triton_cuda")
        assert result1 == []

        # Cache populated
        from akg_agents.core_v2.skill.metadata import dsl_to_dir_key
        dsl_key = dsl_to_dir_key("triton_cuda")
        assert dsl_key in catalog._raw_cache
        assert catalog._raw_cache[dsl_key] == []

        # Second call returns same empty list (no re-load)
        result2 = catalog.load_by_dsl("triton_cuda")
        assert result2 == []

    def test_load_by_dsl_caches_per_dsl(self, monkeypatch):
        """Two distinct DSLs cache independently; calling load_by_dsl
        on one doesn't poison the other's cache slot."""
        from akg_agents.op.skill import operator_skill_catalog as osc

        monkeypatch.setattr(osc, "SKILLS_DIR", Path("/nonexistent/skills/root"))
        catalog = OperatorSkillCatalog()

        catalog.load_by_dsl("triton_cuda")
        catalog.load_by_dsl("triton_ascend")

        from akg_agents.core_v2.skill.metadata import dsl_to_dir_key
        assert dsl_to_dir_key("triton_cuda") in catalog._raw_cache
        assert dsl_to_dir_key("triton_ascend") in catalog._raw_cache

    def test_load_by_dsl_uses_cache_on_hit(self, monkeypatch):
        """If the cache already has an entry for a DSL key, we don't
        touch SkillLoader at all."""
        catalog = OperatorSkillCatalog()
        sentinel = [object(), object(), object()]

        from akg_agents.core_v2.skill.metadata import dsl_to_dir_key
        catalog._raw_cache[dsl_to_dir_key("triton_cuda")] = sentinel

        # Replace the loader with one that raises if called
        class _BoomLoader:
            def load_from_directory(self, *a, **kw):
                raise AssertionError("loader called on cache hit")

        catalog._loader = _BoomLoader()
        assert catalog.load_by_dsl("triton_cuda") is sentinel


# ---------------------------------------------------------------------------
# filter_by_context — load + coarse_filter + exclude
# ---------------------------------------------------------------------------


class TestFilterByContext:

    def _seed_catalog(self, dsl, skills):
        """Create a catalog with a pre-seeded cache for one DSL key."""
        from akg_agents.core_v2.skill.metadata import dsl_to_dir_key
        catalog = OperatorSkillCatalog()
        catalog._raw_cache[dsl_to_dir_key(dsl)] = skills
        return catalog

    def test_empty_cache_returns_empty(self, monkeypatch):
        """Filter on a DSL with no loaded skills returns empty."""
        from akg_agents.op.skill import operator_skill_catalog as osc
        monkeypatch.setattr(osc, "SKILLS_DIR", Path("/nonexistent/skills/root"))

        catalog = OperatorSkillCatalog()
        result = catalog.filter_by_context(
            dsl="triton_cuda", backend="cuda",
            framework="torch", hardware="a100",
        )
        assert result == []

    def test_filter_passes_through_when_metadata_matches(self):
        """A skill whose metadata matches the (backend, dsl) context
        survives coarse_filter."""
        skill = _FakeSkill(
            "guide-1", "guide",
            metadata={
                "backend": "cuda",
                "dsl": "triton_cuda",
                "operator_patterns": "all",
            },
        )
        catalog = self._seed_catalog("triton_cuda", [skill])

        result = catalog.filter_by_context(
            dsl="triton_cuda", backend="cuda",
            framework="torch", hardware="a100",
        )
        assert len(result) == 1
        assert result[0].name == "guide-1"

    def test_filter_drops_wrong_backend(self):
        """A skill whose metadata.backend mismatches is dropped."""
        skill = _FakeSkill(
            "ascend-only", "guide",
            metadata={
                "backend": "ascend",
                "dsl": "triton_ascend",
                "operator_patterns": "all",
            },
        )
        catalog = self._seed_catalog("triton_cuda", [skill])

        result = catalog.filter_by_context(
            dsl="triton_cuda", backend="cuda",
            framework="torch", hardware="a100",
        )
        # ascend-only skill filtered out
        assert result == []

    def test_exclude_names_drops_after_coarse_filter(self):
        skills = [
            _FakeSkill("a", "guide",
                       metadata={"backend": "cuda", "dsl": "triton_cuda",
                                 "operator_patterns": "all"}),
            _FakeSkill("b", "guide",
                       metadata={"backend": "cuda", "dsl": "triton_cuda",
                                 "operator_patterns": "all"}),
        ]
        catalog = self._seed_catalog("triton_cuda", skills)

        result = catalog.filter_by_context(
            dsl="triton_cuda", backend="cuda",
            framework="torch", hardware="a100",
            exclude_names={"a"},
        )
        assert len(result) == 1
        assert result[0].name == "b"

    def test_exclude_names_empty_set_is_noop(self):
        skill = _FakeSkill(
            "kept", "guide",
            metadata={"backend": "cuda", "dsl": "triton_cuda",
                      "operator_patterns": "all"},
        )
        catalog = self._seed_catalog("triton_cuda", [skill])
        result = catalog.filter_by_context(
            dsl="triton_cuda", backend="cuda",
            framework="torch", hardware="a100",
            exclude_names=set(),
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# loader / selector property access
# ---------------------------------------------------------------------------


class TestPropertyAccess:

    def test_loader_property_returns_skill_loader(self):
        from akg_agents.core_v2.skill import SkillLoader
        catalog = OperatorSkillCatalog()
        assert isinstance(catalog.loader, SkillLoader)

    def test_selector_property_returns_operator_skill_selector(self):
        from akg_agents.op.skill.operator_selector import OperatorSkillSelector
        catalog = OperatorSkillCatalog()
        assert isinstance(catalog.selector, OperatorSkillSelector)

    def test_loader_and_selector_stable_across_calls(self):
        """Properties return the same instance, not freshly constructed."""
        catalog = OperatorSkillCatalog()
        assert catalog.loader is catalog.loader
        assert catalog.selector is catalog.selector


# ---------------------------------------------------------------------------
# _normalize_hardware
# ---------------------------------------------------------------------------


class TestNormalizeHardware:
    """Pin the hardware alias normalisation contract."""

    def setup_method(self):
        from akg_agents.op.skill.operator_skill_catalog import _normalize_hardware
        self.normalize = _normalize_hardware

    # -- exact alias hits --

    def test_ascend910b4_maps_to_atlas_a2(self):
        assert self.normalize("ascend910b4") == "Atlas A2"

    def test_ascend910_9391_maps_to_atlas_a3(self):
        assert self.normalize("ascend910_9391") == "Atlas A3"

    def test_alias_is_case_insensitive(self):
        assert self.normalize("Ascend910B4") == "Atlas A2"

    # -- known-prefix unknown-chip → "" (skip filter) --

    def test_unknown_ascend_chip_returns_empty(self):
        assert self.normalize("ascend_future_chip") == ""

    def test_unknown_nvidia_chip_returns_empty(self):
        assert self.normalize("nvidia_h200") == ""

    def test_unknown_sm_chip_returns_empty(self):
        assert self.normalize("sm_120") == ""

    # -- no prefix match → passthrough --

    def test_a100_passthrough(self):
        assert self.normalize("a100") == "a100"

    def test_v100_passthrough(self):
        assert self.normalize("v100") == "v100"

    def test_empty_string_passthrough(self):
        assert self.normalize("") == ""

    def test_unknown_vendor_passthrough(self):
        assert self.normalize("amd_mi300") == "amd_mi300"
