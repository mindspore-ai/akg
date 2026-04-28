# Copyright 2026 Huawei Technologies Co., Ltd
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
Autoresearch skill rendering — markdown assembly for prompt injection.

Two rendering primitives:

  - ``render_skills_markdown(skills, *, max_chars, mode)`` — pure
    single-mode render. ``mode="full"`` defers to
    ``OperatorSkillCatalog.render_as_markdown`` (with lightweight
    category normalization so "method" / "implementation" can piggyback
    on the legacy "guide" / "example" branches of the catalog renderer).
    ``mode="index"`` produces the compact one-line-per-skill index
    with project-relative path hints.

  - ``render_ranked_skill_block(skills, *, top_k, max_chars)`` — budget-
    aware split render: the top-K skills get the full body, the tail
    gets the compact index, with the full section capped at
    ``max_chars * _SKILL_BLOCK_FULL_BUDGET_RATIO`` so one very long
    skill cannot starve the rest of the block.

Shared category taxonomy (``_CATEGORY_LAYER``, ``_canonical_category``)
is imported from ``skill_adapter`` because ranking and rendering both
need the same sort order and "method → guide / implementation →
example" collapse rules. Keyword generation and ranking stay in
``skill_adapter.py``; this module is strictly the prompt-facing
display layer.
"""

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

from akg_agents import get_project_root
from akg_agents.op.skill import OperatorSkillCatalog

from .skill_adapter import _CATEGORY_LAYER, _canonical_category

logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(get_project_root()).resolve()

_SKILL_BLOCK_FULL_BUDGET_RATIO = 0.75
_SKILL_BLOCK_SEPARATOR = "\n\n"


def _sort_key(skill) -> Tuple[int, int, str]:
    category = getattr(skill, "category", "") or ""
    layer, sublayer = _CATEGORY_LAYER.get(category, (9, 9))
    return (layer, sublayer, getattr(skill, "name", "") or "")


def _skill_path_for_index(skill) -> str:
    """Return the task-dir-relative path to this skill's SKILL.md.

    Every skill in the pool is mirrored into ``task_dir/skills/<name>/
    SKILL.md`` by the workflow scaffolder, so the read_file path
    shown in the index is stable and local. Falls back to the
    project-relative catalog path when the skill has no name (never
    expected in practice).
    """
    name = (getattr(skill, "name", "") or "").strip()
    if name:
        return f"skills/{name}/SKILL.md"
    raw_path = getattr(skill, "skill_path", None)
    if not raw_path:
        return ""
    try:
        path = Path(raw_path).resolve()
        return str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")
    except Exception:
        return str(raw_path).replace("\\", "/")


def _coerce_for_full_render(skill):
    category = getattr(skill, "category", "") or ""
    family = _canonical_category(category)
    if family == category:
        return skill
    # Only name/category/content are fed to the legacy catalog render.
    # Any new field read by OperatorSkillCatalog.render_as_markdown()
    # must be mirrored here as well.
    return SimpleNamespace(
        name=getattr(skill, "name", ""),
        category=family,
        content=getattr(skill, "content", ""),
    )


def _render_index(skills: List) -> str:
    lines = ["## Skill Index", ""]
    for skill in sorted(skills, key=_sort_key):
        category = getattr(skill, "category", "") or "?"
        description = getattr(skill, "description", "") or ""
        first_line = description.strip().splitlines()[0] if description.strip() else ""
        path_hint = _skill_path_for_index(skill)
        if path_hint:
            line = f"- {skill.name} [{category}] @ {path_hint}: {first_line}"
        else:
            line = f"- {skill.name} [{category}]: {first_line}"
        lines.append(line.rstrip())
    return "\n".join(lines).rstrip()


def render_skills_markdown(
    skills: List,
    *,
    max_chars: Optional[int] = None,
    mode: str = "full",
) -> str:
    if not skills:
        return ""
    if max_chars is not None and max_chars <= 0:
        return ""

    if mode == "index":
        rendered = _render_index(skills)
    else:
        if mode != "full":
            logger.warning(
                "render_skills_markdown: unknown mode=%r, falling back to full",
                mode,
            )
        normalized = [_coerce_for_full_render(skill) for skill in skills]
        rendered = OperatorSkillCatalog.render_as_markdown(normalized)

    if max_chars is not None and len(rendered) > max_chars:
        tail = "\n...[truncated]"
        if max_chars <= len(tail):
            return tail[:max_chars]
        return rendered[:max_chars - len(tail)] + tail
    return rendered


def render_ranked_skill_block(
    skills: List,
    *,
    top_k: int,
    max_chars: Optional[int],
) -> str:
    if not skills:
        return ""
    if max_chars is not None and max_chars <= 0:
        return ""

    top_k = max(top_k, 0)
    full_skills = skills[:top_k]
    remainder = skills[top_k:]

    if max_chars is not None and remainder:
        full_budget = max(int(max_chars * _SKILL_BLOCK_FULL_BUDGET_RATIO), 0)
    else:
        full_budget = max_chars

    full_text = render_skills_markdown(
        full_skills,
        max_chars=full_budget,
        mode="full",
    ) if full_skills else ""

    if max_chars is None:
        index_budget = None
    else:
        spent = len(full_text)
        join_cost = len(_SKILL_BLOCK_SEPARATOR) if full_text and remainder else 0
        index_budget = max(max_chars - spent - join_cost, 0)

    index_text = render_skills_markdown(
        remainder,
        max_chars=index_budget,
        mode="index",
    ) if remainder and (index_budget is None or index_budget > 0) else ""

    if full_text and index_text:
        return full_text + _SKILL_BLOCK_SEPARATOR + index_text
    return full_text or index_text
