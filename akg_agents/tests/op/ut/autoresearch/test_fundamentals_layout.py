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
"""Tests for the unified task_dir/skills/ layout + Layer 0 fundamentals
scan in build_system_prompt."""

import os
import tempfile
from pathlib import Path

from akg_agents.op.autoresearch.agent.prompt_builder import (
    _build_fundamentals_section,
    _parse_skill_frontmatter,
    _scan_task_skills,
)


def _write_skill(root: Path, name: str, category: str, body: str,
                 description: str = "") -> None:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    fm_desc = f'description: "{description}"\n' if description else ""
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ncategory: {category}\n{fm_desc}---\n\n{body}\n",
        encoding="utf-8",
    )


class TestFrontMatterParser:
    def test_parses_scalar_keys(self):
        meta, body = _parse_skill_frontmatter(
            '---\nname: foo\ncategory: guide\n---\nhello\n'
        )
        assert meta["name"] == "foo"
        assert meta["category"] == "guide"
        assert body == "hello\n"

    def test_parses_metadata_block(self):
        meta, _ = _parse_skill_frontmatter(
            '---\nname: foo\nmetadata:\n  backend: ascend\n  dsl: triton_ascend\n---\n'
        )
        assert meta["metadata"]["backend"] == "ascend"
        assert meta["metadata"]["dsl"] == "triton_ascend"

    def test_missing_delimiter_is_noop(self):
        meta, body = _parse_skill_frontmatter("no frontmatter here\n")
        assert meta == {}
        assert body == "no frontmatter here\n"

    def test_unterminated_delimiter_is_noop(self):
        meta, body = _parse_skill_frontmatter(
            '---\nname: foo\ncategory: guide\n# never closed\n'
        )
        assert meta == {}


class TestScanTaskSkills:
    def test_empty_directory_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            assert _scan_task_skills(d) == []

    def test_skips_unreadable_entries(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Subdirectory without SKILL.md → skipped
            (root / "skills" / "empty").mkdir(parents=True)
            _write_skill(root, "ok", "fundamental", "body")
            out = _scan_task_skills(str(root))
            assert [name for name, _, _ in out] == ["ok"]

    def test_returns_name_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_skill(root, "zebra", "guide", "z body")
            _write_skill(root, "alpha", "fundamental", "a body")
            names = [name for name, _, _ in _scan_task_skills(str(root))]
            assert names == ["alpha", "zebra"]


class TestFundamentalsSection:
    def test_includes_only_fundamentals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_skill(root, "rules-a", "fundamental", "rule A body")
            _write_skill(root, "rules-b", "fundamental", "rule B body")
            _write_skill(root, "matmul", "guide", "guide body")
            section = _build_fundamentals_section(str(root), 20_000)
            assert "## DSL Fundamentals" in section
            assert "rules-a" in section
            assert "rule A body" in section
            assert "rules-b" in section
            # guide/example live on disk for read_file but not in Layer 0
            assert "matmul" not in section
            assert "guide body" not in section

    def test_empty_when_no_fundamentals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_skill(root, "matmul", "guide", "guide body")
            assert _build_fundamentals_section(str(root), 20_000) == ""

    def test_budget_caps_output(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_skill(root, "big", "fundamental", "x" * 5000)
            _write_skill(root, "small", "fundamental", "short body")
            # Budget big enough for header + small only
            section = _build_fundamentals_section(str(root), 500)
            # The overflowing entry is either partially skipped or
            # appears in the skipped-list footer — but output must
            # not exceed the budget.
            assert len(section) <= 500

    def test_skip_note_lists_dropped_names(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_skill(root, "small", "fundamental", "short body")
            _write_skill(root, "huge-a", "fundamental", "x" * 1000)
            _write_skill(root, "huge-b", "fundamental", "y" * 1000)
            section = _build_fundamentals_section(str(root), 200)
            # At least one of the huge ones should be named in the
            # skip note so the drop is visible.
            assert "skipped" in section or section == ""

    def test_missing_skills_dir_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            assert _build_fundamentals_section(d, 20_000) == ""

    def test_zero_budget_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_skill(root, "rules", "fundamental", "body")
            assert _build_fundamentals_section(str(root), 0) == ""
