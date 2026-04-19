"""
SkillPool - single owner of the keyword-ranked candidate skill list.

Replaces the old `feedback.preselected_skills` (a naked list) plus the
scattered free helpers `iter_bindable_skills` / `build_auto_plan_items` /
`narrow_skills_once`. Holds a back-reference to the SkillBuilder so
callers querying "what's bindable right now" never have to compute
exclude sets themselves.

Two write entry points:

  - :meth:`refill` (mode="replace") — startup, called once by
    AgentLoop. Wipes the pool and runs the keyword pipeline from
    scratch.

  - :meth:`refill` (mode="append") — agent-driven, called by the
    `search_skills` tool handler. Reuses the same pipeline with a
    natural-language hint biasing keyword generation; appends only
    the dedup-fresh new candidates and registers them in SkillBuilder
    as `selected`.

Two read entry points:

  - Iteration / `len()` / `top_k(n)` — used by prompt rendering
    (`build_initial_message`, `compress.py` bootstrap) and any test
    that peeks at pool contents.  For binding decisions (update_plan
    keyword matching) use :meth:`residual_bindable` instead.

  - :meth:`match_by_keywords` — used by
    `TurnExecutor._match_keywords_to_skills`. Ranks residual-bindable
    skills against agent-supplied keywords and returns scored matches
    that have at least one keyword hit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Set

from .skill_adapter import (
    QueryKeywords,
    TRACKABLE_PATTERN_CATEGORIES,
    _canonical_category,
    _get_catalog,
    _normalize_token,
    generate_query_keywords,
    rank_skills_by_keywords,
)

if TYPE_CHECKING:
    from .skill_builder import SkillBuilder


logger = logging.getLogger(__name__)


def _first_sentence(text: str) -> str:
    text = " ".join((text or "").strip().split())
    if not text:
        return ""
    cut = len(text)
    for sep in ("\u3002", "."):
        idx = text.find(sep)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip()


def _truncate_with_ellipsis(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars == 1:
        return "\u2026"
    return text[: max_chars - 1].rstrip() + "\u2026"


def _ctx_from_config(config) -> dict:
    """Read DSL / backend / framework / arch with metadata fallback.

    Mirrors the layout AKG configs use today: top-level fields take
    priority, ``config.metadata`` provides legacy fallback. Also accepts
    ``hardware`` as a synonym for ``arch`` because the catalog filter
    expects that key.
    """
    op_name = getattr(config, "name", "") or ""
    dsl = getattr(config, "dsl", "") or ""
    framework = getattr(config, "framework", "") or ""
    backend = getattr(config, "backend", "") or ""
    arch = getattr(config, "arch", "") or ""
    metadata = getattr(config, "metadata", None) or {}
    if not dsl:
        dsl = metadata.get("dsl", "") or ""
    if not backend:
        backend = metadata.get("backend", "") or ""
    if not framework:
        framework = metadata.get("framework", "") or ""
    if not arch:
        arch = metadata.get("arch", "") or metadata.get("hardware", "") or ""
    return {
        "op_name": op_name,
        "dsl": dsl,
        "framework": framework,
        "backend": backend,
        "arch": arch,
    }


class SkillPool:
    """Owns the candidate skill list + binding-eligibility queries."""

    def __init__(self, skill_builder: "SkillBuilder"):
        self._ranked: List[Any] = []
        self._skill_builder = skill_builder

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Any]:
        return iter(self._ranked)

    def __len__(self) -> int:
        return len(self._ranked)

    def __bool__(self) -> bool:
        return bool(self._ranked)

    def is_empty(self) -> bool:
        return not self._ranked

    # ------------------------------------------------------------------
    # Direct write hooks (used by tests; refill() is the prod entry)
    # ------------------------------------------------------------------

    def replace(self, ranked: List[Any]) -> None:
        """Wipe the pool and store the given ranked list verbatim.

        Used by ``refill(mode='replace')`` and by tests that want to
        seed a deterministic pool without going through the keyword
        pipeline.
        """
        self._ranked = list(ranked)

    def append_new(self, ranked: List[Any]) -> List[Any]:
        """Append dedup-fresh entries to the end of the pool.

        Returns the list of newly-added skill objects (those whose
        ``name`` was not already present). Used by
        ``refill(mode='append')`` and by tests that want to simulate
        an agent-driven pool extension.
        """
        existing = {getattr(s, "name", "") or "" for s in self._ranked}
        added: List[Any] = []
        for s in ranked:
            name = getattr(s, "name", "") or ""
            if not name or name in existing:
                continue
            self._ranked.append(s)
            existing.add(name)
            added.append(s)
        return added

    def top_k(self, n: int) -> List[Any]:
        """First ``n`` ranked entries (no eligibility filter).

        Used by prompt rendering callers that need the raw ranked
        window. For binding decisions (update_plan), use
        :meth:`residual_bindable` instead — it skips terminal skills.
        """
        return list(self._ranked[:n]) if n > 0 else []

    def names(self) -> List[str]:
        """Names of all ranked entries (debug / log helper)."""
        return [getattr(s, "name", "") or "" for s in self._ranked]

    def get_skill_content(self, name: str) -> str:
        """Return the full content (SKILL.md body) for a skill by name.

        Returns ``""`` if not found.  Used by the loop to inject the
        full skill document into the conversation on item activation.
        """
        for s in self._ranked:
            if (getattr(s, "name", "") or "") == name:
                return getattr(s, "content", "") or ""
        return ""


    # ------------------------------------------------------------------
    # Read-side bind queries
    # ------------------------------------------------------------------

    def residual_bindable(
        self,
        *,
        top_k: int,
        trackable_categories: frozenset = TRACKABLE_PATTERN_CATEGORIES,
    ) -> List[Any]:
        """Sliding window over the pool keeping trackable-category entries.

        Post-abandon refactor: every registered skill is bindable. We
        no longer exclude "previously unbound" skills here — instead
        ``match_by_keywords`` uses ``SkillBuilder.tier()`` to rank them
        last within the keyword-match result set.
        """
        if top_k <= 0:
            return []
        seen: Set[str] = set()
        out: List[Any] = []
        for s in self._ranked:
            if len(out) >= top_k:
                break
            name = getattr(s, "name", "") or ""
            category = getattr(s, "category", "") or ""
            if not name or category not in trackable_categories:
                continue
            if name in seen:
                continue
            seen.add(name)
            out.append(s)
        return out

    def _tier_of(self, skill: Any) -> int:
        """Binding priority via SkillBuilder — 0 applied, 1 fresh, 2 unbound.

        Unknown / unregistered skills default to tier 1 (fresh) so the
        ordering is stable for skills the pool carries but the registry
        hasn't seen yet (shouldn't normally happen, but defensive).
        """
        name = getattr(skill, "name", "") or ""
        rec = self._skill_builder.get(name) if name else None
        if rec is None:
            return 1
        return rec.tier()

    def match_by_keywords(
        self,
        keywords: List[str],
        *,
        top_k: int,
        op_name: str = "",
        exclude_names: Optional[Set[str]] = None,
    ) -> List[tuple[float, Any]]:
        """Rank residual bindable skills for a keyword-requested item.

        Category prior alone is never enough: at least one keyword must
        actually hit the skill text/content for the skill to remain
        eligible. Within the eligible set, the result order is:

          1. Applied-first (tier 0, proven KEEP on this run).
          2. Fresh candidates (tier 1, registered, never applied,
             never unbound).
          3. Previously unbound (tier 2, last resort — still
             selectable, just ranked last).

          Within each tier, entries are ordered by the keyword-ranker's
          score (descending).
        """
        if top_k <= 0:
            return []

        normalized: List[str] = []
        seen_tokens: Set[str] = set()
        for raw in keywords or []:
            token = _normalize_token(raw)
            if not token or token in seen_tokens:
                continue
            seen_tokens.add(token)
            normalized.append(token)
        if not normalized:
            return []

        excluded = {
            (name or "").strip()
            for name in (exclude_names or set())
            if (name or "").strip()
        }

        candidates = [
            skill
            for skill in self.residual_bindable(top_k=max(len(self._ranked), top_k))
            if (getattr(skill, "name", "") or "") not in excluded
        ]
        if not candidates:
            return []

        ranked = rank_skills_by_keywords(
            candidates,
            QueryKeywords(must_have=normalized),
            stage="optimize",
            op_name=op_name,
        )

        def _has_keyword_hit(skill: Any) -> bool:
            searchable = _normalize_token(
                " ".join(
                    part for part in [
                        getattr(skill, "name", "") or "",
                        getattr(skill, "description", "") or "",
                        getattr(skill, "content", "") or "",
                    ]
                    if part
                )
            )
            return any(token in searchable for token in normalized)

        filtered: List[tuple[float, Any]] = [
            (score, skill) for score, skill in ranked
            if _has_keyword_hit(skill)
        ]
        # Stable sort by (tier, -score). Python's sort is stable, so
        # equal (tier, score) pairs keep the keyword-ranker's input
        # order. Negative score for descending within tier.
        filtered.sort(key=lambda p: (self._tier_of(p[1]), -p[0]))
        return filtered[:top_k]

    # ------------------------------------------------------------------
    # Write-side: refill from the keyword pipeline
    # ------------------------------------------------------------------

    async def refill(
        self,
        *,
        llm,
        config,
        hint: str = "",
        mode: str = "replace",
        plan_version: int = 0,
        include_categories: list[str] | None = None,
    ) -> List[Any]:
        """Run the keyword-ranking pipeline.

        ``mode="replace"`` (startup): wipe the pool, write the freshly
        ranked list. Returns the new ranked list.

        ``mode="append"`` (agent-driven via ``search_skills``): keep
        existing entries, append only dedup-fresh candidates to the
        end. Returns the list of newly-added skill objects.

        ``include_categories``: if set, only skills whose canonical
        category is in this list pass through. Canonical mapping
        collapses ``method → guide``, ``implementation → example``.

        Diagnostic prefix is ``[skill_pool]`` so
        ``grep [skill_pool] agent.log`` shows the full refill history.
        """
        if mode not in ("replace", "append"):
            raise ValueError(f"refill mode must be 'replace' or 'append', got {mode!r}")

        ctx = _ctx_from_config(config)
        if not ctx["dsl"]:
            logger.info(
                "[skill_pool] refill skipped: no DSL in config/metadata "
                "(no skill package to draw from)",
            )
            if mode == "replace":
                self.replace([])
            return []

        catalog = _get_catalog()
        # Post-abandon refactor: there is no terminal state anymore.
        # Previously-unbound skills stay eligible — they just get
        # demoted to tier 2 in ``match_by_keywords``. The only exclude
        # set we still build is the in-pool dedup set for append mode.
        exclude_names: Set[str] = set()
        if mode == "append":
            exclude_names = {
                getattr(s, "name", "") or "" for s in self._ranked
            }

        skills = catalog.filter_by_context(
            dsl=ctx["dsl"],
            backend=ctx["backend"],
            framework=ctx["framework"],
            hardware=ctx["arch"],
            exclude_names=exclude_names,
        )
        before_case_filter = len(skills)
        skills = [
            skill
            for skill in skills
            if (getattr(skill, "category", "") or "") != "case"
            or catalog.classify_case_type(skill) == "improvement"
        ]
        after_case_filter = len(skills)
        # Category tier filter (e.g. initial load = guides only)
        if include_categories:
            canonical_set = {_canonical_category(c) for c in include_categories}
            skills = [
                s for s in skills
                if _canonical_category(
                    getattr(s, "category", "") or ""
                ) in canonical_set
            ]
        logger.info(
            f"[skill_pool] refill mode={mode} ctx="
            f"dsl={ctx['dsl']!r} backend={ctx['backend']!r} "
            f"framework={ctx['framework']!r} arch={ctx['arch']!r}; "
            f"catalog={before_case_filter} -> case={after_case_filter} "
            f"-> tier={len(skills)}"
            f"{f' (include={include_categories})' if include_categories else ''}"
            f" (excluded: {sorted(exclude_names)})",
        )
        if not skills:
            logger.info(
                f"[skill_pool] refill mode={mode}: catalog returned 0 "
                f"matches (no new candidates)",
            )
            if mode == "replace":
                self.replace([])
            return []

        keywords = await generate_query_keywords(
            op_name=ctx["op_name"],
            task_desc=getattr(config, "description", "") or "",
            dsl=ctx["dsl"],
            backend=ctx["backend"],
            framework=ctx["framework"],
            arch=ctx["arch"],
            hint=hint,
            stage="optimize",
            llm=llm,
            timeout=getattr(config.agent, "skill_narrow_timeout", 30.0),
        )
        ranked = [
            skill
            for _, skill in rank_skills_by_keywords(
                skills,
                keywords,
                stage="optimize",
                op_name=ctx["op_name"],
            )
        ]

        if mode == "replace":
            self.replace(ranked)
            added: List[Any] = list(ranked)
        else:  # append
            added = self.append_new(ranked)

        # Register new trackable entries in SkillBuilder. In both
        # replace and append mode, every newly-added skill whose
        # category is trackable is registered in the registry.
        for skill in added:
            category = getattr(skill, "category", "") or ""
            if category not in TRACKABLE_PATTERN_CATEGORIES:
                continue
            self._skill_builder.register(
                skill,
                reason=(
                    "agent search hint" if hint else "runtime pre-selected"
                ),
                plan_version=plan_version,
            )

        logger.info(
            f"[skill_pool] refill mode={mode}: pool now has "
            f"{len(self._ranked)} entries; "
            f"new this call={len(added)} "
            f"({[getattr(s, 'name', '') for s in added[:10]]}"
            f"{'...' if len(added) > 10 else ''})",
        )
        return added
