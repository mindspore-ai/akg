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

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from akg_agents.op.autoresearch.agent.compress import _build_bootstrap
from akg_agents.op.autoresearch.agent.prompt_builder import (
    build_initial_message,
    build_system_prompt,
)
from akg_agents.op.autoresearch.agent.skill_pool import SkillPool


class _FakeSkill:
    def __init__(self, name, category, content="", description="", skill_path=None):
        self.name = name
        self.category = category
        self.content = content or f"<content of {name}>"
        self.description = description or f"{name} description"
        self.metadata = {}
        self.skill_path = skill_path


@dataclass
class _StubAgentCfg:
    editable_file_truncate: int = 8_000
    system_context_file_truncate: int = 15_000
    skill_block_top_k: int = 2
    skill_block_max_chars: int = 500
    compact_diagnosis_truncate: int = 2_000


@dataclass
class _StubConfig:
    name: str = "matmul"
    primary_metric: str = "latency_us"
    lower_is_better: bool = True
    agent: _StubAgentCfg = None

    def __post_init__(self):
        if self.agent is None:
            self.agent = _StubAgentCfg()


class _StubRunner:
    def __init__(self):
        self.best_result = SimpleNamespace(metrics={"latency_us": 123.0})

    @staticmethod
    def get_editable_contents():
        return {"kernel.py": "print('hello')"}


class _EmptySkillBuilder:
    @staticmethod
    def is_empty():
        return True

    @staticmethod
    def get(_name):
        return None

    def __contains__(self, _name):
        return False


def _make_pool(skills) -> SkillPool:
    pool = SkillPool(_EmptySkillBuilder())
    pool.replace(list(skills))
    return pool


@dataclass
class _SystemPromptAgentCfg:
    system_context_file_truncate: int = 15_000
    system_context_total_truncate: int = 40_000


@dataclass
class _SystemPromptConfig:
    name: str = "matmul"
    description: str = "optimize matmul on CUDA"
    primary_metric: str = "latency_us"
    lower_is_better: bool = True
    editable_files: list = field(default_factory=lambda: ["kernel.py"])
    metadata: dict = field(default_factory=dict)
    program_file: str = ""
    ref_file: str = ""
    context_files: list = field(default_factory=list)
    constraints: dict = field(default_factory=dict)
    forbidden_patterns: dict = field(default_factory=lambda: {
        "content": [], "diff": [], "diff_any": [],
    })
    agent: _SystemPromptAgentCfg = None

    def __post_init__(self):
        if self.agent is None:
            self.agent = _SystemPromptAgentCfg()


def _feedback(*, phase, preselected_skills, status_text="") -> SimpleNamespace:
    skill_builder = _EmptySkillBuilder()
    return SimpleNamespace(
        phase=phase,
        skill_pool=_make_pool(preselected_skills),
        format_status=lambda: status_text,
        must_replan=False,
        skill_builder=skill_builder,
    )


def test_build_system_prompt_renders_literal_braces_in_template():
    workdir = Path(tempfile.mkdtemp(dir=os.getcwd()))
    try:
        full, knowledge = build_system_prompt(_SystemPromptConfig(), str(workdir))
        assert "latency_us" in full
        assert "kernel.py" in full
        # The example item dict in the PLANNING section should render
        # with literal braces (template uses {{ }} escapes).
        assert '{"text": "concrete change"' in full
        assert '"rationale": "why"' in full
        # Tool protocol stays out of the knowledge-only base.
        assert "update_plan" not in knowledge
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_build_initial_message_includes_preselected_skills():
    workdir = Path(tempfile.mkdtemp(dir=os.getcwd()))
    try:
        config = _StubConfig()
        runner = _StubRunner()
        feedback = _feedback(
            phase="no_plan",
            preselected_skills=[
                _FakeSkill("matmul", "implementation", content="matmul body"),
                _FakeSkill("tiling", "method", content="tiling body"),
                _FakeSkill("generic", "guide", content="generic body"),
            ],
        )

        message = build_initial_message(
            config=config,
            task_dir=str(workdir),
            runner=runner,
            feedback=feedback,
            max_rounds=10,
        )

        assert "## Pre-selected Skills" in message
        # Initial prompt is now index-only: full skill body is NOT
        # inlined; it is injected into the conversation when a
        # skill-bound item activates.
        assert "matmul body" not in message
        assert "tiling body" not in message
        assert "## Skill Index" in message
        # Each skill rendered as a one-line index entry.
        assert "matmul [implementation]" in message
        assert "tiling [method]" in message
        assert "generic [guide]" in message
        # Agent never sees "backing_skill" — bindings are system-only.
        assert "backing_skill" not in message
        # The no_plan start guidance points the agent at update_plan
        # with rationale + keywords.
        assert "search_skills" in message
        assert "Submit `update_plan" in message
        assert "rationale" in message
        # Old auto-augment language MUST NOT appear.
        assert "appends bound items" not in message
        assert "pre-bound" not in message
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_build_initial_message_includes_top_guide_exploration_hint():
    workdir = Path(tempfile.mkdtemp(dir=os.getcwd()))
    try:
        config = _StubConfig()
        runner = _StubRunner()
        feedback = _feedback(
            phase="no_plan",
            preselected_skills=[
                _FakeSkill("opt-basics", "fundamental", description="fundamental desc"),
                _FakeSkill(
                    "matmul-guide",
                    "guide",
                    description="matmul guide first line\nextra line should be ignored",
                ),
                _FakeSkill("matmul-example", "example", description="example desc"),
            ],
        )

        message = build_initial_message(
            config=config,
            task_dir=str(workdir),
            runner=runner,
            feedback=feedback,
            max_rounds=10,
        )

        assert "## Exploration Hint" in message
        assert "Top-ranked guide to consider before defaulting to parameter tuning:" in message
        assert "- matmul-guide: matmul guide first line" in message
        assert "extra line should be ignored" not in message
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_build_initial_message_omits_exploration_hint_without_guide():
    workdir = Path(tempfile.mkdtemp(dir=os.getcwd()))
    try:
        config = _StubConfig()
        runner = _StubRunner()
        feedback = _feedback(
            phase="no_plan",
            preselected_skills=[
                _FakeSkill("opt-basics", "fundamental", description="fundamental desc"),
                _FakeSkill("matmul-example", "example", description="example desc"),
            ],
        )

        message = build_initial_message(
            config=config,
            task_dir=str(workdir),
            runner=runner,
            feedback=feedback,
            max_rounds=10,
        )

        assert "## Exploration Hint" not in message
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_build_initial_message_resumed_branch_renders_plan_status():
    """Resumed sessions get the resumed branch with a Plan Status
    block; fresh runs always start at no_plan in the new design."""
    workdir = Path(tempfile.mkdtemp(dir=os.getcwd()))
    try:
        config = _StubConfig()
        runner = _StubRunner()
        feedback = _feedback(
            phase="active",
            preselected_skills=[_FakeSkill("matmul", "implementation", content="matmul body")],
            status_text="## Plan Status\n>>> [p1] resume me",
        )

        message = build_initial_message(
            config=config,
            task_dir=str(workdir),
            runner=runner,
            feedback=feedback,
            max_rounds=10,
            session_restored=True,
        )

        assert "## Plan Status" in message
        assert ">>> [p1]" in message
        assert "Resumed session" in message
        assert "Continue the plan" in message
        # No_plan-branch start prompt absent in resumed branch.
        assert "Each item needs `text` + `rationale`" not in message
        assert "## Exploration Hint" not in message
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_build_initial_message_fundamentals_excluded_from_pool_index():
    """Fundamentals live in the system prompt (Layer 0) and MUST NOT
    appear in the initial user message's Pre-selected Skills index.
    The index now contains only bindable guides/examples/cases, and
    the read_file hint points the agent at task_dir/skills/<name>/
    SKILL.md."""
    workdir = Path(tempfile.mkdtemp(dir=os.getcwd()))
    try:
        config = _StubConfig()
        runner = _StubRunner()
        feedback = _feedback(
            phase="no_plan",
            preselected_skills=[
                _FakeSkill("matmul-guide", "guide", content="guide body"),
                _FakeSkill("basics", "fundamental", content="basics body"),
            ],
        )

        message = build_initial_message(
            config=config,
            task_dir=str(workdir),
            runner=runner,
            feedback=feedback,
            max_rounds=10,
        )

        # Bindable guide renders under Pre-selected Skills.
        assert "## Pre-selected Skills" in message
        assert "matmul-guide [guide]" in message
        # Fundamentals are gone from the initial user message.
        assert "## Reference Skills" not in message
        assert "basics [fundamental]" not in message
        # The read_file hint tells the agent where to fetch SKILL.md.
        assert "skills/matmul-guide/SKILL.md" in message
        assert "read_file" in message
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_build_bootstrap_no_longer_includes_skill_index():
    """Phase 8 (multi-step compact rewrite): the bootstrap no longer
    renders a ``Bindable skills`` / ``Skill Index`` block. That
    content was moved out — operator identity goes into the
    [OPERATOR_SUMMARY] section (from LLM call #1), and per-skill
    bindability is implicit in the plan analysis."""
    workdir = Path(tempfile.mkdtemp(dir=os.getcwd()))
    try:
        config = _StubConfig()
        feedback = SimpleNamespace(
            must_replan=False,
            format_status=lambda: "",
            skill_builder=_EmptySkillBuilder(),
            skill_pool=_make_pool([
                _FakeSkill(
                    "matmul", "implementation",
                    description="matmul desc",
                    skill_path=workdir / "skills" / "matmul" / "SKILL.md",
                )
            ]),
            _plan_items=[],
            plan_version=0,
        )

        bootstrap = _build_bootstrap(str(workdir), config, feedback=feedback)
        content = bootstrap[0]["content"]

        assert "Bindable skills" not in content
        assert "## Skill Index" not in content
        assert "matmul [implementation]" not in content
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_build_bootstrap_renders_operator_summary_when_provided():
    """When the caller passes ``op_summary``, it is rendered under the
    ``[OPERATOR_SUMMARY]`` marker in the bootstrap body."""
    workdir = Path(tempfile.mkdtemp(dir=os.getcwd()))
    try:
        config = _StubConfig()
        feedback = SimpleNamespace(
            must_replan=False,
            format_status=lambda: "",
            skill_builder=_EmptySkillBuilder(),
            skill_pool=_make_pool([]),
            _plan_items=[],
            plan_version=0,
        )
        bootstrap = _build_bootstrap(
            str(workdir), config, feedback=feedback,
            op_summary="## Operator Shape\nMatmul with elementwise tail.",
        )
        content = bootstrap[0]["content"]
        assert "[OPERATOR_SUMMARY]" in content
        assert "Matmul with elementwise tail" in content
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_build_bootstrap_renders_current_plan_items():
    """Current plan items (v{N}) are rendered directly in the bootstrap
    — no more duplicate Plan Status block (that used to come from
    feedback.format_status)."""
    workdir = Path(tempfile.mkdtemp(dir=os.getcwd()))
    try:
        config = _StubConfig()
        items = [
            {"id": "p1", "text": "change X", "status": "active",
             "backing_skill": "matmul", "keywords": ["matmul"]},
            {"id": "p2", "text": "change Y", "status": "pending",
             "backing_skill": None, "keywords": []},
        ]
        feedback = SimpleNamespace(
            must_replan=False,
            format_status=lambda: "",
            skill_builder=_EmptySkillBuilder(),
            skill_pool=_make_pool([]),
            _plan_items=items,
            plan_version=7,
        )
        bootstrap = _build_bootstrap(str(workdir), config, feedback=feedback)
        content = bootstrap[0]["content"]
        assert "## Current Plan v7" in content
        assert "[p1] change X" in content
        assert "[p2] change Y" in content
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
