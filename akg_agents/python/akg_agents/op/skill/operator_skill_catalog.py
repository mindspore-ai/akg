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
OperatorSkillCatalog — per-DSL skill loading with metadata filtering
and caching, the missing facade above SkillLoader / OperatorSkillSelector.

The pattern "load all skills for a DSL once, cache the raw list, then
coarse_filter on demand by (backend, dsl, framework, hardware) context"
was previously duplicated in three places:

  - kernel_gen.KernelGen._init_skills + _load_skills_by_dsl
  - kernel_designer.KernelDesigner._init_skills + _load_skills_by_dsl
  - autoresearch.agent.skill_adapter._load_and_filter_skills (no cache)

This module is the single home for that pattern. The current commit
migrates only ``autoresearch.agent.skill_adapter`` to use it; kernel_gen
and kernel_designer continue to maintain their own inline copies and
will be migrated in a separate commit if/when that cleanup is decided.

Why this exists in ``op/skill/`` and not in ``autoresearch``:

  - It is a generic operator-domain primitive (load + filter + cache),
    not autoresearch-specific. Other op-domain agents (kernel_gen,
    kernel_designer) can adopt it later by importing from the same
    ``op/skill/`` namespace they already use for ``OperatorSkillSelector``.
  - The "filtering skill loader" capability didn't exist anywhere as
    a reusable component — it was always inlined inside whichever
    agent needed it. This module is the missing public facade above
    the raw ``SkillLoader`` / ``OperatorSkillSelector`` primitives.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from akg_agents import get_project_root
from akg_agents.core_v2.skill import SkillLoader
from akg_agents.core_v2.skill.metadata import dsl_to_dir_key
from akg_agents.op.skill.operator_selector import (
    OperatorSelectionContext,
    OperatorSkillSelector,
)

logger = logging.getLogger(__name__)

# Resolved at import time; cheap (just a Path join, no I/O).
SKILLS_DIR = Path(get_project_root()) / "op" / "resources" / "skills"

# ---------------------------------------------------------------------------
# Hardware name normalisation — loaded from hardware_aliases.yaml
# ---------------------------------------------------------------------------

_HW_ALIASES_FILE = os.path.join(os.path.dirname(__file__), "hardware_aliases.yaml")


def _load_hardware_aliases() -> tuple[dict[str, str], list[str]]:
    """Parse hardware_aliases.yaml into (alias_map, fallback_prefixes).

    Returns empty structures if the file is missing or malformed so the
    normaliser degrades to passthrough rather than crashing.
    """
    alias_map: dict[str, str] = {}
    prefixes: list[str] = []
    if not os.path.exists(_HW_ALIASES_FILE):
        return alias_map, prefixes
    try:
        with open(_HW_ALIASES_FILE, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception:
        return alias_map, prefixes
    for vendor in raw.values():
        if not isinstance(vendor, dict):
            continue
        for p in vendor.get("prefixes") or []:
            if isinstance(p, str):
                prefixes.append(p.lower())
        for chip, family in (vendor.get("aliases") or {}).items():
            if isinstance(chip, str) and isinstance(family, str):
                alias_map[chip.lower()] = family
    return alias_map, prefixes


_ALIAS_MAP, _FALLBACK_PREFIXES = _load_hardware_aliases()


def _normalize_hardware(hw: str) -> str:
    """Map a runtime arch identifier to the skill-metadata hardware name.

    Returns the mapped name when an exact alias exists, ``""`` (skip
    hardware filter) when the id matches a known vendor prefix but has
    no alias, or the original value otherwise.
    """
    if not hw:
        return hw
    key = hw.lower()
    mapped = _ALIAS_MAP.get(key)
    if mapped is not None:
        return mapped
    if any(key.startswith(p) for p in _FALLBACK_PREFIXES):
        return ""
    return hw


class OperatorSkillCatalog:
    """Per-DSL skill loading with metadata filtering, caching, classification,
    and rendering.

    A single instance can serve multiple callers; the per-DSL cache is
    additive and read-mostly, so concurrent shares are safe in CPython
    (dict insertion is atomic under the GIL, and existing entries are
    never overwritten).

    Public API:
      - ``load_by_dsl(dsl)``                                → list[Skill]
      - ``filter_by_context(dsl, backend, ..., exclude=)``  → list[Skill]
      - ``classify_case_type(skill)``  [staticmethod]       → 'fix' | 'improvement'
      - ``render_as_markdown(skills)`` [classmethod]        → str
      - ``loader``, ``selector``  (read-only properties for callers
        that need to drop down to the raw primitives directly)
    """

    # Layer ordering for ``render_as_markdown`` — copied verbatim from
    # ``kernel_gen.KernelGen.CATEGORY_LAYER`` so the rendered markdown
    # format the hint advisor sees is byte-identical to what kernel_gen
    # produces. Future-changes warning: if kernel_gen's CATEGORY_LAYER
    # diverges from this, the two renderings will drift. Acceptable
    # because both are stable conventions, and the long-term cleanup
    # is to point kernel_gen at this constant (separate commit).
    CATEGORY_LAYER = {
        "fundamental": (0, 0),
        "reference":   (0, 1),
        "guide":       (1, 0),
        "example":     (1, 1),
        "case":        (2, 0),
    }

    def __init__(self) -> None:
        self._loader = SkillLoader()
        self._selector = OperatorSkillSelector()
        self._raw_cache: Dict[str, List] = {}  # dsl_key -> raw skill list

    # -- Read-only access to underlying primitives -----------------------

    @property
    def loader(self) -> SkillLoader:
        """Underlying ``SkillLoader``. Exposed for callers that need to
        do raw file loads bypassing the per-DSL caching layer."""
        return self._loader

    @property
    def selector(self) -> OperatorSkillSelector:
        """Underlying ``OperatorSkillSelector``. Exposed for callers
        that want to coarse_filter their own skill list with a
        custom context (without going through ``filter_by_context``)."""
        return self._selector

    # -- Loading + caching ----------------------------------------------

    def load_by_dsl(self, dsl: str) -> List:
        """Load all skills under ``SKILLS_DIR/{dsl_key}/``, cached per DSL.

        Returns ``[]`` (and caches the empty result) if the directory
        doesn't exist or load fails. Subsequent calls for the same
        ``dsl`` are O(1) dict lookups.

        Mirrors the body of ``kernel_gen.KernelGen._load_skills_by_dsl``.
        """
        dsl_key = dsl_to_dir_key(dsl)
        if dsl_key in self._raw_cache:
            return self._raw_cache[dsl_key]

        dsl_dir = SKILLS_DIR / dsl_key
        if not dsl_dir.exists():
            logger.warning(
                f"Skill directory not found for DSL '{dsl_key}': {dsl_dir}"
            )
            self._raw_cache[dsl_key] = []
            return []

        try:
            skills = self._loader.load_from_directory(dsl_dir)
            self._raw_cache[dsl_key] = skills
            logger.info(f"Loaded {len(skills)} skills from {dsl_dir}")
            return skills
        except Exception as e:
            logger.error(f"Failed to load skills for DSL '{dsl_key}': {e}")
            self._raw_cache[dsl_key] = []
            return []

    # -- Loading + filtering ---------------------------------------------

    def filter_by_context(
        self,
        dsl: str,
        backend: str,
        framework: str = "",
        hardware: str = "",
        operator_type: str = "",
        exclude_names: Optional[Set[str]] = None,
    ) -> List:
        """Load + coarse_filter for the given operator context.

        Cached at the load step (per-dsl), filtered fresh per call.
        ``exclude_names`` is applied **after** coarse filtering so the
        caller can dynamically exclude skills the operator has already
        tried (e.g., autoresearch's applied/abandoned set).

        ``operator_type`` is optional — when omitted, the operator-type
        filter (which matches against ``skill.metadata.operator_type``
        or its ``operator_patterns`` fallback) is a no-op pass-through.

        Returns the empty list on any failure; callers should treat
        an empty result as "no skills available, fail open" rather
        than as an error condition.
        """
        all_skills = self.load_by_dsl(dsl)
        if not all_skills:
            return []

        try:
            ctx = OperatorSelectionContext(
                backend=backend, dsl=dsl,
                framework=framework or "",
                hardware=_normalize_hardware(hardware or ""),
                operator_type=operator_type or None,
            )
            filtered = self._selector.coarse_filter(all_skills, ctx)
        except Exception as e:
            logger.warning(f"coarse_filter failed for dsl={dsl}: {e}")
            filtered = list(all_skills)

        if exclude_names:
            filtered = [s for s in filtered if s.name not in exclude_names]

        return filtered

    def find_examples_by_operator_type(
        self,
        dsl: str,
        backend: str,
        operator_type: str,
        *,
        framework: str = "",
        hardware: str = "",
        max_results: int = 2,
    ) -> List:
        """Find example skills matching the given operator_type.

        Used by the supervisor to provide concrete code references
        when a deviated verdict fires. Leverages the cached load +
        coarse_filter pipeline with an ``include_categories``
        whitelist.

        Returns up to ``max_results`` skills, empty list on failure.
        """
        if not operator_type:
            return []
        all_skills = self.load_by_dsl(dsl)
        if not all_skills:
            return []
        try:
            ctx = OperatorSelectionContext(
                backend=backend,
                dsl=dsl,
                framework=framework,
                hardware=_normalize_hardware(hardware or ""),
                operator_type=operator_type,
                include_categories=["example", "implementation"],
            )
            filtered = self._selector.coarse_filter(all_skills, ctx)
        except Exception as e:
            logger.warning(f"find_examples_by_operator_type failed: {e}")
            return []
        return filtered[:max_results]

    # -- Case type classification ----------------------------------------

    @staticmethod
    def classify_case_type(skill) -> str:
        """Classify a case skill as ``'fix'`` or ``'improvement'``.

        Legacy helper for autoresearch and historical ``category='case'``
        skills that carry fix/improvement hints. KernelGen now treats
        ``category='case'`` as non-fix generation examples and relies on
        explicit ``category='fix'`` for repair knowledge. Priority order:

          1. ``metadata.case_type`` (explicit, if 'fix' or 'improvement')
          2. ``metadata.source == 'error_fix'`` → 'fix'
          3. ``skill_path`` containing 'evolved-fix' → 'fix'
          4. ``skill_path`` containing 'evolved-improvement' → 'improvement'
          5. Default → 'improvement'
        """
        meta = getattr(skill, "metadata", {}) or {}
        ct = meta.get("case_type", "")
        if ct in ("fix", "improvement"):
            return ct
        if meta.get("source") == "error_fix":
            return "fix"
        skill_path = getattr(skill, "skill_path", None)
        if skill_path:
            path_str = str(skill_path)
            if "evolved-fix" in path_str:
                return "fix"
            if "evolved-improvement" in path_str:
                return "improvement"
        return "improvement"

    # -- Rendering --------------------------------------------------------

    @classmethod
    def render_as_markdown(cls, skills: List) -> str:
        """Render skills as a category-ordered markdown block.

        Mirrors ``kernel_gen.KernelGen._assemble_skill_contents`` —
        copied here for the same reason as ``classify_case_type``.

        Section order is determined by ``CATEGORY_LAYER``:

          1. fundamentals + references → "### 基础知识与规范"
          2. guides                    → "### 算子优化指南"
          3. examples                  → "### 代码示例参考"
          4. cases                     → "### 优化/修复案例"

        Sections are joined by the same ``\\n\\n---\\n\\n`` separator
        kernel_gen uses, so the format the hint advisor sees is
        byte-identical to what kernel_gen renders.
        """
        if not skills:
            return ""

        def _sort_key(s):
            cat = (getattr(s, "category", "") or "")
            layer, sublayer = cls.CATEGORY_LAYER.get(cat, (9, 9))
            return (layer, sublayer, getattr(s, "name", ""))

        sorted_skills = sorted(skills, key=_sort_key)
        by_cat = {"fundamental": [], "reference": [],
                  "guide": [], "example": [], "case": []}
        for s in sorted_skills:
            cat = (getattr(s, "category", "") or "")
            if cat in by_cat:
                by_cat[cat].append(s)

        sections = []
        basics = by_cat["fundamental"] + by_cat["reference"]
        if basics:
            sections.append(
                "### 基础知识与规范\n\n"
                + "\n\n---\n\n".join(s.content for s in basics)
            )
        if by_cat["guide"]:
            sections.append(
                "### 算子优化指南\n\n"
                + "\n\n---\n\n".join(s.content for s in by_cat["guide"])
            )
        if by_cat["example"]:
            sections.append(
                "### 代码示例参考\n\n"
                + "\n\n---\n\n".join(s.content for s in by_cat["example"])
            )
        if by_cat["case"]:
            sections.append(
                "### 优化/修复案例\n\n"
                + "\n\n---\n\n".join(s.content for s in by_cat["case"])
            )

        order_desc = [
            f"{s.name}[{getattr(s, 'category', '?')}]" for s in sorted_skills
        ]
        logger.info(f"Skill assembly order: {order_desc}")

        return "\n\n---\n\n".join(sections)
