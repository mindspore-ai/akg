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
"""Behavioural tests for the Phase 3 autoresearch refactor.

Three things are pinned here:

  1. ``execute_quick_check`` split — the orchestrator now delegates to
     per-stage helpers (_qc_syntax / _qc_dsl_compliance / _qc_import /
     _qc_smoke). The helpers exist with the right signatures, short-
     circuit correctly, and the public ToolResult behaviour matches
     the old monolithic implementation for the common cases.

  2. ``FeedbackBuilder`` accepts ``skill_builder`` / ``skill_pool`` at
     construction time instead of via post-hoc attribute assignment.
     Callers see a fully-initialized object, not a half-wired one.

  3. ``feedback_validation`` module exposes the pure helpers that used
     to live on FeedbackBuilder as private methods, and the thin
     FeedbackBuilder wrappers still compute the same results.
"""
from types import SimpleNamespace

import pytest

from akg_agents.op.autoresearch.agent import feedback_validation as fv
from akg_agents.op.autoresearch.agent import tools as tools_mod
from akg_agents.op.autoresearch.agent.feedback import FeedbackBuilder
from akg_agents.op.autoresearch.agent.skill_builder import SkillBuilder


# -- (1) execute_quick_check split -------------------------------------------


class TestQuickCheckSplit:
    """The orchestrator delegates to named per-stage helpers."""

    def test_per_stage_helpers_exist(self):
        for name in ("_qc_syntax", "_qc_dsl_compliance", "_qc_import",
                     "_qc_smoke", "_qc_resolve_smoke_cmd"):
            assert hasattr(tools_mod, name), (
                f"Phase 3a expected per-stage helper {name} in tools.py"
            )

    def test_syntax_ok_returns_none(self, tmp_path):
        f = tmp_path / "clean.py"
        f.write_text("x = 1\n")
        assert tools_mod._qc_syntax(str(f), "clean.py") is None

    def test_syntax_error_returns_string(self, tmp_path):
        f = tmp_path / "broken.py"
        f.write_text("def x(:\n")  # deliberate syntax error
        err = tools_mod._qc_syntax(str(f), "broken.py")
        assert err is not None
        assert "SyntaxError in broken.py" in err

    def test_dsl_compliance_empty_when_dsl_missing(self, tmp_path):
        f = tmp_path / "k.py"
        f.write_text("x = 1\n")
        cfg = SimpleNamespace(dsl="", backend="")
        assert fv is not None  # sanity, avoid unused import
        assert tools_mod._qc_dsl_compliance(str(f), "k.py", cfg) == []

    def test_quick_check_syntax_only_mode_skips_smoke(self, tmp_path):
        """When ``import_timeout <= 0`` the orchestrator short-circuits
        to a syntax-only success — it must not reach the smoke stage."""
        f = tmp_path / "k.py"
        f.write_text("x = 1\n")
        cfg = SimpleNamespace(
            editable_files=["k.py"],
            dsl="",
            backend="",
            framework="",
            import_timeout=0,          # => syntax-only mode
            smoke_test_script="",
            smoke_test_timeout=10,
            agent=SimpleNamespace(smoke_output_limit=1000),
        )
        result = tools_mod.execute_quick_check(str(tmp_path), cfg, device_id=None)
        assert result.ok is True
        assert "syntax only" in result.message

    def test_quick_check_syntax_error_fails(self, tmp_path):
        f = tmp_path / "k.py"
        f.write_text("def x(:\n")
        cfg = SimpleNamespace(
            editable_files=["k.py"],
            dsl="",
            backend="",
            framework="",
            import_timeout=0,
            smoke_test_script="",
            smoke_test_timeout=10,
            agent=SimpleNamespace(smoke_output_limit=1000),
        )
        result = tools_mod.execute_quick_check(str(tmp_path), cfg, device_id=None)
        assert result.ok is False
        assert "SyntaxError in k.py" in result.message

    def test_resolve_smoke_cmd_no_smoke_when_no_config(self, tmp_path):
        cfg = SimpleNamespace(
            smoke_test_script="",
            dsl="",
            framework="",
            backend="",
        )
        assert tools_mod._qc_resolve_smoke_cmd(str(tmp_path), cfg) is None


# -- (2) FeedbackBuilder constructor injection -------------------------------


class TestFeedbackBuilderConstructorInjection:
    """skill_builder / skill_pool are constructor arguments."""

    def _cfg(self):
        return SimpleNamespace(
            agent=SimpleNamespace(
                plan_item_rationale_min_chars=30,
                plan_item_rationale_max_chars=400,
            ),
        )

    def test_constructor_accepts_skill_builder(self):
        sb = SkillBuilder()
        fb = FeedbackBuilder(self._cfg(), skill_builder=sb)
        assert fb.skill_builder is sb

    def test_constructor_accepts_skill_pool(self):
        sb = SkillBuilder()
        sentinel = object()
        fb = FeedbackBuilder(self._cfg(), skill_builder=sb, skill_pool=sentinel)
        assert fb.skill_pool is sentinel

    def test_defaults_to_none_when_not_provided(self):
        """Standalone / test fixtures still work without skill wiring."""
        fb = FeedbackBuilder(self._cfg())
        assert fb.skill_builder is None
        assert fb.skill_pool is None


# -- (3) feedback_validation module ------------------------------------------


class TestFeedbackValidationModule:
    """The pure helpers exist at module scope and produce the same
    results the old private methods produced."""

    def test_sanitize_text_basic(self):
        assert fv.sanitize_text("  hello  ", 100) == "hello"
        assert fv.sanitize_text("", 100) == ""
        assert fv.sanitize_text(None, 100) == ""
        assert fv.sanitize_text(123, 100) == ""

    def test_sanitize_text_truncation(self):
        out = fv.sanitize_text("a" * 200, 10)
        assert out.endswith("\u2026")
        assert len(out) <= 11

    def test_validate_rationale_rejects_empty(self):
        cleaned, err = fv.validate_rationale("", min_chars=30, max_chars=400)
        assert cleaned is None
        assert "required" in err

    def test_validate_rationale_rejects_too_short(self):
        cleaned, err = fv.validate_rationale(
            "too short", min_chars=30, max_chars=400,
        )
        assert cleaned is None
        assert "too short" in err

    def test_validate_rationale_rejects_generic_phrase(self):
        # Long enough to pass the min-chars check, but still generic
        # enough (banned phrase present, length below 2 * min_chars)
        # to trip the generic filter.
        cleaned, err = fv.validate_rationale(
            "improve performance around the loop",
            min_chars=30, max_chars=400,
        )
        assert cleaned is None
        assert "too generic" in err

    def test_validate_rationale_accepts_specific(self):
        text = (
            "inner loop reloads A from global memory on each iteration; "
            "caching in shared should cut latency"
        )
        cleaned, err = fv.validate_rationale(
            text, min_chars=30, max_chars=400,
        )
        assert cleaned == text
        assert err == ""

    @pytest.mark.parametrize("reason,label", [
        ("", "fail"),
        ("no improvement", "no improvement"),
        ("the test timed out", "timeout"),
        ("CompilationError: foo", "compile error"),
        ("x is not defined", "compile error"),
        ("correctness mismatch", "correctness"),
        ("UB overflow!", "ub overflow"),
        ("syntax invalid", "syntax error"),
        ("some unknown thing", "fail"),
    ])
    def test_classify_reason(self, reason, label):
        assert fv.classify_reason(reason) == label

    @pytest.mark.parametrize("ok,reason,expected", [
        (True, "", True),                              # KEEP always signal
        (True, "no improvement", True),
        (False, "correctness", True),                  # real fail
        (False, "superseded by replan", False),        # control event
        (False, "abandoned (diagnose)", False),        # control event
    ])
    def test_is_eval_signal(self, ok, reason, expected):
        assert fv.is_eval_signal(ok, reason) is expected

    def test_render_item_line_active(self):
        item = {"id": "p1", "status": "active", "text": "do X",
                "backing_skill": None, "keywords": []}
        line = fv.render_item_line(item)
        assert line.startswith("- >>> [p1] do X")
        assert "unbound" in line

    def test_render_item_line_done_ok_with_skill_and_keywords(self):
        item = {"id": "p2", "status": "done_ok", "text": "did Y",
                "backing_skill": "vectorize", "keywords": ["loop", "tile"]}
        line = fv.render_item_line(item)
        assert "- [O] [p2] did Y" in line
        assert "backing: vectorize" in line
        assert "keywords: loop, tile" in line


# -- Thin wrappers on FeedbackBuilder still produce identical output --------


class TestFeedbackBuilderDelegation:
    """Delegation wrappers must not drift from the underlying pure helper."""

    def _fb(self):
        return FeedbackBuilder(SimpleNamespace(
            agent=SimpleNamespace(
                plan_item_rationale_min_chars=30,
                plan_item_rationale_max_chars=400,
            ),
        ))

    def test_sanitize_text_wrapper_matches(self):
        fb = self._fb()
        assert fb._sanitize_text("  hi  ", 100) == fv.sanitize_text("  hi  ", 100)

    def test_classify_reason_wrapper_matches(self):
        fb = self._fb()
        for r in ("", "timed out", "correctness", "weird"):
            assert fb._classify_reason(r) == fv.classify_reason(r)

    def test_is_eval_signal_wrapper_matches(self):
        fb = self._fb()
        for ok, reason in [(True, ""), (False, "superseded by replan"),
                           (False, "correctness")]:
            assert fb._is_eval_signal(ok, reason) == fv.is_eval_signal(ok, reason)

    def test_render_item_line_wrapper_matches(self):
        fb = self._fb()
        item = {"id": "p1", "status": "pending", "text": "X",
                "backing_skill": None, "keywords": []}
        assert fb._render_item_line(item) == fv.render_item_line(item)
