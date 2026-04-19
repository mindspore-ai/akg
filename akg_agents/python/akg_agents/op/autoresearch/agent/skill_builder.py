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
"""SkillBuilder — a bookkeeping registry for skills seen across a run.

No terminal states. Every registered skill is always bindable; the
agent's ``acknowledge_skill(applicability="unbind")`` call releases
the binding from one item but the skill stays available for later
items. Two monotonic badges drive the binding priority:

    applied_versions   : plan versions in which this skill's binding
                         item settled KEEP. Proven to have helped.
    unbound_at_versions: plan versions in which the agent acknowledged
                         the skill as not useful for that item. Not
                         terminal — just a demotion signal.

Binding priority (used by ``SkillPool.match_by_keywords``):

    tier 0: applied_versions non-empty  → proven effective, try again first
    tier 1: neither applied nor unbound → fresh candidates
    tier 2: unbound_at_versions non-empty without applied → last resort

A skill that was unbound once and later successfully applied is tier 0
(applied takes precedence). This lets the registry self-correct:
retrying a previously-unbound skill on a different item and winning a
KEEP promotes it back to the top of the list.

What this class intentionally does NOT track:
  - which plan item is currently bound to which skill — derivable
    from FeedbackBuilder._plan_items (status == "active" AND
    backing_skill != None).
  - settled history — FeedbackBuilder owns the plan-item history.

Persistence: skill_state_to_dict / skill_state_from_dict round-trip
into session.json. Format is v5; v4 payloads (the old terminal-
"abandoned" shape) are silently migrated by mapping
``abandoned_at_version`` → ``unbound_at_versions=[version]``.

Plan.md rendering: ``render_for_plan_md()`` produces four sections
(Active / Selected / Applied / Previously unbound) derived from the
current snapshot. "Active" and "Applied" are computed on the fly
from the live plan_items passed in by FeedbackBuilder.format_plan_file.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillRecord:
    """Registry entry for one skill."""
    name: str
    category: str = ""
    description: str = ""
    metadata: dict = field(default_factory=dict)
    content_preview: str = ""
    registered_at_version: int = 0
    registered_reason: str = ""
    applied_versions: list = field(default_factory=list)
    unbound_at_versions: list = field(default_factory=list)
    last_reason: str = ""

    def tier(self) -> int:
        """Binding priority — 0 = applied, 1 = fresh, 2 = previously unbound.

        Applied always wins even if the skill was unbound earlier; a
        successful KEEP demotes "previously unbound" back to front of
        line.
        """
        if self.applied_versions:
            return 0
        if self.unbound_at_versions:
            return 2
        return 1


class SkillBuilder:
    """Lightweight registry of skills registered during a run.

    Passed to ``FeedbackBuilder(skill_builder=...)`` at AgentLoop init.
    FeedbackBuilder queries it on plan.md render and on settlement to
    badge applied skills; TurnExecutor calls ``mark_unbound`` on the
    agent's ``acknowledge_skill(applicability="unbind")`` call.
    Diagnose does NOT flip skills — it only rewinds the plan.
    """

    def __init__(self, config=None, task_dir: Optional[str] = None):
        self.config = config
        self.task_dir = task_dir
        self._skills: dict[str, SkillRecord] = {}

    # -- Queries ----------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def is_empty(self) -> bool:
        return not self._skills

    def get(self, name: str) -> Optional[SkillRecord]:
        return self._skills.get(name)

    def get_applied_names(self) -> set:
        return {r.name for r in self._skills.values() if r.applied_versions}

    def get_unbound_names(self) -> set:
        """Skills that were unbound at some point (regardless of applied state).

        Consumers that want "demoted but not applied" should intersect
        with ``get_applied_names()`` themselves.
        """
        return {r.name for r in self._skills.values()
                if r.unbound_at_versions}

    # -- Mutations --------------------------------------------------------

    def register(self, skill, reason: str, plan_version: int = 0) -> None:
        """Ensure ``skill`` has a registry entry.

        Idempotent. An already-registered entry has its
        ``registered_reason`` and ``registered_at_version`` updated so
        the latest selection context wins; this matters for plan.md's
        "Selected: skill (reason)" line.
        """
        name = getattr(skill, "name", None) or skill.get("name") if isinstance(skill, dict) else getattr(skill, "name", None)
        if not name:
            return
        existing = self._skills.get(name)
        if existing is not None:
            existing.registered_reason = reason
            existing.registered_at_version = plan_version
            return
        content = getattr(skill, "content", "") or ""
        preview = content[:400] + ("..." if len(content) > 400 else "")
        self._skills[name] = SkillRecord(
            name=name,
            category=getattr(skill, "category", "") or "",
            description=getattr(skill, "description", "") or "",
            metadata=dict(getattr(skill, "metadata", {}) or {}),
            content_preview=preview,
            registered_at_version=plan_version,
            registered_reason=reason,
        )
        logger.info(
            f"[skill-registry] registered {name!r} reason={reason!r} "
            f"v{plan_version}",
        )

    def record_applied(self, skill_name: str, plan_version: int,
                       reason: str = "") -> None:
        """Badge ``skill_name`` as applied at ``plan_version``.

        No-op if the skill is not registered. Duplicates of the same
        version are suppressed (applying twice in one plan version is
        still one applied event). KEEPing a previously-unbound skill
        promotes it back to tier 0 automatically — ``applied_versions``
        non-empty wins over ``unbound_at_versions`` in ``tier()``.
        """
        rec = self._skills.get(skill_name)
        if rec is None:
            logger.warning(
                f"[skill-registry] record_applied: unknown skill "
                f"{skill_name!r} — ignoring",
            )
            return
        if plan_version not in rec.applied_versions:
            rec.applied_versions.append(plan_version)
            rec.last_reason = reason or f"applied in v{plan_version}"
            logger.info(
                f"[skill-registry] {skill_name!r} applied in v{plan_version}",
            )

    def mark_unbound(self, skill_name: str, reason: str,
                     plan_version: int = 0) -> None:
        """Record that ``skill_name`` was released from one plan item.

        Non-terminal — the skill remains bindable for other items;
        ``tier()`` demotes it to tier 2 only when no ``applied_versions``
        exists. Duplicate same-version unbinds are suppressed.
        """
        rec = self._skills.get(skill_name)
        if rec is None:
            logger.warning(
                f"[skill-registry] mark_unbound: unknown skill "
                f"{skill_name!r} — ignoring",
            )
            return
        if plan_version in rec.unbound_at_versions:
            return
        rec.unbound_at_versions.append(plan_version)
        reason_preview = (reason or "")[:200].replace("\n", " ")
        rec.last_reason = reason_preview or f"unbound in v{plan_version}"
        logger.info(
            f"[skill-registry] {skill_name!r} unbound v{plan_version} "
            f"reason={reason_preview!r}",
        )

    # -- Serialization ----------------------------------------------------

    def skill_state_to_dict(self) -> dict:
        return {
            "version": 5,
            "skills": {name: asdict(rec) for name, rec in self._skills.items()},
        }

    def skill_state_from_dict(self, d: dict) -> None:
        """Restore from session.json.

        Accepts:
          * v5 — native format (unbound_at_versions list).
          * v4 — legacy "abandoned" shape; migrated on load by mapping
                 ``abandoned=True`` + ``abandoned_at_version=N`` to
                 ``unbound_at_versions=[N]``.

        Unknown versions are rejected (empty registry); the caller is
        expected to have validated at the SessionStore layer.
        """
        self._skills = {}
        if not d:
            return
        version = d.get("version")
        if version not in (4, 5):
            logger.warning(
                "[skill-registry] skill_state_from_dict: unsupported "
                "version %r — starting empty", version,
            )
            return
        for name, rec_dict in (d.get("skills") or {}).items():
            unbound_versions = list(rec_dict.get("unbound_at_versions") or [])
            if version == 4:
                # v4 migration: collapse (abandoned, abandoned_at_version)
                # into unbound_at_versions so downstream code only has to
                # deal with the v5 shape.
                if rec_dict.get("abandoned"):
                    av = rec_dict.get("abandoned_at_version")
                    unbound_versions = (
                        [int(av)] if isinstance(av, int) else [0]
                    )
            self._skills[name] = SkillRecord(
                name=rec_dict.get("name", name),
                category=rec_dict.get("category", ""),
                description=rec_dict.get("description", ""),
                metadata=rec_dict.get("metadata", {}) or {},
                content_preview=rec_dict.get("content_preview", ""),
                registered_at_version=rec_dict.get("registered_at_version", 0),
                registered_reason=rec_dict.get("registered_reason", ""),
                applied_versions=list(rec_dict.get("applied_versions") or []),
                unbound_at_versions=unbound_versions,
                last_reason=rec_dict.get("last_reason", ""),
            )

    # -- Rendering --------------------------------------------------------

    def format_short_status(self) -> str:
        """One-liner: ``Skills: N registered, X applied, Y previously unbound``."""
        if not self._skills:
            return ""
        total = len(self._skills)
        applied = sum(1 for r in self._skills.values() if r.applied_versions)
        unbound_only = sum(
            1 for r in self._skills.values()
            if r.unbound_at_versions and not r.applied_versions
        )
        return (
            f"Skills: {total} registered, {applied} applied, "
            f"{unbound_only} previously unbound"
        )

    def render_for_plan_md(
        self, plan_items: Optional[Iterable[dict]] = None,
    ) -> str:
        """Render the ``## Skill State`` section for plan.md.

        Four buckets (order matches the binding-priority story agent-facing):

            Active               — skills whose name appears as an
                                   active item's backing_skill.
            Applied              — at least one applied_versions entry.
            Selected             — registered, never applied, never
                                   unbound.
            Previously unbound   — unbound_at_versions non-empty AND
                                   never applied. Last-resort tier.

        Applied takes precedence over Previously unbound: a skill that
        was unbound once and later KEEP'd shows up in Applied, not in
        Previously unbound — which matches ``tier()``.
        """
        if not self._skills:
            return ""

        active_names: set[str] = set()
        for item in (plan_items or []):
            if item.get("status") == "active" and item.get("backing_skill"):
                active_names.add(item["backing_skill"])

        lines = ["## Skill State"]

        def _render_entry(rec: SkillRecord) -> list[str]:
            out = []
            cat = f"({rec.category})" if rec.category else ""
            applied_str = (
                " applied@v" + ",".join(str(v) for v in rec.applied_versions)
                if rec.applied_versions else ""
            )
            unbound_str = (
                " unbound@v"
                + ",".join(str(v) for v in rec.unbound_at_versions)
                if rec.unbound_at_versions else ""
            )
            if rec.name in active_names:
                out.append(
                    f"- [>>>] {rec.name} {cat}{applied_str}{unbound_str}".rstrip()
                )
            elif rec.applied_versions:
                out.append(
                    f"- [O] {rec.name} {cat}{applied_str}{unbound_str}".rstrip()
                )
            elif rec.unbound_at_versions:
                out.append(
                    f"- [~] {rec.name} {cat}{unbound_str} — "
                    f"{rec.last_reason}".rstrip()
                )
            else:
                reason = (
                    f" — {rec.registered_reason}" if rec.registered_reason else ""
                )
                out.append(f"- [ ] {rec.name} {cat}{reason}".rstrip())
            return out

        by_bucket: dict[str, list[SkillRecord]] = {
            "active": [], "applied": [], "selected": [], "previously_unbound": [],
        }
        for rec in self._skills.values():
            if rec.name in active_names:
                by_bucket["active"].append(rec)
            elif rec.applied_versions:
                by_bucket["applied"].append(rec)
            elif rec.unbound_at_versions:
                by_bucket["previously_unbound"].append(rec)
            else:
                by_bucket["selected"].append(rec)

        if by_bucket["active"]:
            lines.append("")
            lines.append("### Active (adopting now)")
            for rec in by_bucket["active"]:
                lines.extend(_render_entry(rec))
        if by_bucket["applied"]:
            lines.append("")
            lines.append("### Applied (pattern adopted successfully)")
            for rec in by_bucket["applied"]:
                lines.extend(_render_entry(rec))
        if by_bucket["selected"]:
            lines.append("")
            lines.append("### Selected (candidates, not yet bound)")
            for rec in by_bucket["selected"]:
                lines.extend(_render_entry(rec))
        if by_bucket["previously_unbound"]:
            lines.append("")
            lines.append(
                "### Previously unbound (last-resort tier; still "
                "selectable, ranked last)"
            )
            for rec in by_bucket["previously_unbound"]:
                lines.extend(_render_entry(rec))

        return "\n".join(lines)
