"""
FeedbackBuilder — Constructs system feedback messages for the agent.

Manages:
  - Structured plan items with enforced execution order
  - Auto-settlement of items after eval
  - Settled history across plan versions
  - Eval/failure/quick_check feedback message construction

Couples loosely with ``SkillBuilder`` (see ``skill_builder.py``) via an
optional ``self.skill_builder`` reference attached by ``AgentLoop``.
Plan items declare a ``backing_skill`` name, and ``settle_active``
badges the skill as applied via ``SkillBuilder.record_applied`` when the
item settles OK. All such calls are guarded so autoresearch still runs
when no SkillBuilder is attached.
"""

import logging
import os
import re
from collections import Counter
from typing import TYPE_CHECKING, Optional

from . import feedback_validation as _fv

if TYPE_CHECKING:
    from .skill_builder import SkillBuilder
    from .skill_pool import SkillPool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed records for plan items / settled history
#
# These are ``dict`` subclasses so every existing reader that does
# ``item["id"]``, ``item.get("backing_skill")``, ``**item`` unpacking,
# ``isinstance(item, dict)`` or ``json.dumps(item)`` keeps working
# unchanged. The only behaviour they add over plain dict is a required-
# field check on construction, which turns silent typos at write sites
# ("back_skill" vs "backing_skill") into loud ValueErrors during tests.
# A legacy call site can opt out of the schema check by building a raw
# dict — but all in-tree builders now go through these wrappers.
# ---------------------------------------------------------------------------


class PlanItem(dict):
    """A single item in the agent's plan. Dict-compatible for legacy
    call sites; validated at construction time."""
    __slots__ = ()

    REQUIRED = ("id", "text", "status", "sketch",
                "backing_skill", "keywords", "rationale")

    def __init__(self, **kwargs):
        missing = [k for k in self.REQUIRED if k not in kwargs]
        if missing:
            raise ValueError(
                f"PlanItem missing required fields: {missing}")
        super().__init__(**kwargs)

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        return f"PlanItem({dict.__repr__(self)})"


class SettledRecord(dict):
    """A single entry in ``_settled_history``. Dict-compatible for legacy
    call sites; validated at construction time."""
    __slots__ = ()

    REQUIRED = ("item_id", "plan_text", "sketch", "backing_skill",
                "keywords", "rationale", "text", "ok", "reason",
                "metrics", "version", "signal_eligible")

    def __init__(self, **kwargs):
        missing = [k for k in self.REQUIRED if k not in kwargs]
        if missing:
            raise ValueError(
                f"SettledRecord missing required fields: {missing}")
        super().__init__(**kwargs)

    def __repr__(self) -> str:  # pragma: no cover — debug aid
        return f"SettledRecord({dict.__repr__(self)})"


class FeedbackBuilder:
    """Builds feedback messages injected into the agent conversation."""

    def __init__(self, config, task_dir: str = "",
                 skill_builder: Optional["SkillBuilder"] = None,
                 skill_pool: Optional["SkillPool"] = None):
        self.config = config
        self._task_dir = task_dir
        self._plan: Optional[str] = None
        self._plan_items: list[dict] = []       # [{id, text, status, sketch, backing_skill, keywords}]
        self._active_item_id: Optional[str] = None
        self._plan_version: int = 0
        self._settled_history: list[dict] = []  # [{text, ok, reason, metrics, version, sketch, backing_skill}]
        self._must_replan: bool = False         # Set by require_replan, blocks edits until replacement submitted
        self._last_diagnosis: Optional[str] = None  # Set by require_replan; persists across replace for session resume

        # Skill registry (``SkillBuilder``) and candidate pool
        # (``SkillPool``). Both may be None when autoresearch is used
        # without skill tracking (tests, standalone). Passed at
        # construction time rather than attached post-hoc so callers
        # cannot observe a half-initialized FeedbackBuilder.
        self.skill_builder: Optional["SkillBuilder"] = skill_builder
        self.skill_pool: Optional["SkillPool"] = skill_pool

    def _validate_backing_skill(self, name: str | None) -> str | None:
        """Return *name* only if it is a registered skill in the ledger.

        Post-abandon refactor: every registered skill is bindable —
        there is no terminal state anymore. A skill that was unbound
        by the agent on a previous item stays available; the only
        effect is a priority-tier demotion (see ``SkillRecord.tier``).
        """
        if not name:
            return None
        if self.skill_builder is None:
            return None
        return name if name in self.skill_builder else None

    def _sanitize_text(self, raw, max_chars: int) -> str:
        """Thin wrapper — see :func:`feedback_validation.sanitize_text`."""
        return _fv.sanitize_text(raw, max_chars)

    def _validate_rationale(self, raw) -> tuple[Optional[str], str]:
        """Thin wrapper — see :func:`feedback_validation.validate_rationale`.

        Reads ``plan_item_rationale_{min,max}_chars`` off
        ``self.config.agent`` with the same defaults the old in-line
        implementation used.
        """
        a = getattr(self.config, "agent", None)
        max_chars = getattr(a, "plan_item_rationale_max_chars", 400) or 400
        min_chars = getattr(a, "plan_item_rationale_min_chars", 30) or 30
        return _fv.validate_rationale(
            raw, min_chars=min_chars, max_chars=max_chars,
        )

    def _build_plan_item(
        self,
        *,
        item_id: str,
        text: str,
        rationale: str,
        keywords: Optional[list[str]] = None,
        backing_skill: Optional[str] = None,
        sketch: str = "",
        status: str = "pending",
    ) -> dict:
        """Single source of truth for plan_item dict shape.

        Caller must pre-validate ``rationale`` via ``_validate_rationale``;
        this factory only constructs the dict.  ``backing_skill`` flows
        through ``_validate_backing_skill`` so terminal-state skills are
        silently stripped.
        """
        return PlanItem(
            id=item_id,
            text=text,
            status=status,
            sketch=sketch,
            backing_skill=self._validate_backing_skill(backing_skill),
            keywords=list(keywords or []),
            rationale=rationale,
        )

    @staticmethod
    def _is_eval_signal(ok: bool, reason: str) -> bool:
        """Thin wrapper — see :func:`feedback_validation.is_eval_signal`."""
        return _fv.is_eval_signal(ok, reason)

    def _build_history_entry(
        self,
        *,
        item: dict,
        ok: bool,
        reason: str,
        metrics: Optional[dict] = None,
        edit_desc: str = "",
    ) -> dict:
        """Single source of truth for settled_history entry shape.

        Sets ``signal_eligible`` so consumers (history pattern
        detection, future stats) can filter out control events
        without re-implementing the classifier.
        """
        return SettledRecord(
            item_id=item.get("id"),
            plan_text=item.get("text", ""),
            sketch=item.get("sketch", ""),
            backing_skill=item.get("backing_skill"),
            keywords=list(item.get("keywords") or []),
            rationale=item.get("rationale") or "",
            text=edit_desc or item.get("text", ""),
            ok=ok,
            reason=reason,
            metrics=metrics or {},
            version=self._plan_version,
            signal_eligible=self._is_eval_signal(ok, reason),
        )

    def _sanitize_keywords(self, raw_keywords) -> list[str]:
        """Normalize optional plan-item keywords for storage/rendering."""
        if not isinstance(raw_keywords, list):
            return []
        cap = getattr(
            getattr(self.config, "agent", None),
            "skill_keyword_max_per_item",
            5,
        )
        try:
            cap = int(cap)
        except (TypeError, ValueError):
            cap = 5
        if cap <= 0:
            return []

        out: list[str] = []
        seen: set[str] = set()
        for token in raw_keywords:
            if not isinstance(token, str):
                continue
            cleaned = token.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            out.append(cleaned)
            if len(out) >= cap:
                break
        return out

    def _persist_plan(self):
        """Write plan.md to task_dir on every state change."""
        if not self._task_dir:
            return
        content = self.format_plan_file()
        if not content:
            return
        try:
            path = os.path.join(self._task_dir, "plan.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception:
            pass

    # -- Phase (derived, never stored) ------------------------------------

    @property
    def phase(self) -> str:
        if not self._plan_items:
            return "no_plan"
        if self._active_item_id:
            return "active"
        return "replanning"

    @property
    def must_replan(self) -> bool:
        return self._must_replan

    def get_keep_item_keys(self) -> set:
        """Return (version, item_id) tuples for items that produced KEEP."""
        return {
            (e.get("version", 0), e["item_id"])
            for e in self._settled_history
            if e.get("ok") and e.get("item_id")
        }

    @property
    def plan_version(self) -> int:
        """Read-only access to the current plan version.

        Bumped by ``submit_plan`` each time a new plan replaces the
        old one. ``SkillPool.refill`` reads it to tag newly-registered
        skills, and ``SkillBuilder.mark_unbound`` / ``record_applied``
        append it to the per-skill ``unbound_at_versions`` /
        ``applied_versions`` lists.
        """
        return self._plan_version

    @property
    def last_diagnosis(self) -> Optional[str]:
        """Last diagnosis text from the diagnose subagent.

        Paired with ``must_replan``: both are written by
        ``require_replan(diagnosis=...)`` and both are read together
        when injecting the diagnose message on resume / compact bootstrap.
        """
        return self._last_diagnosis

    @last_diagnosis.setter
    def last_diagnosis(self, value: Optional[str]) -> None:
        """Setter exists so the resume path can restore the text from
        session.json without going through ``require_replan`` (which
        would also re-mutate plan items)."""
        self._last_diagnosis = value

    # -- Plan state --------------------------------------------------------

    @property
    def plan(self) -> Optional[str]:
        return self._plan

    @plan.setter
    def plan(self, value: Optional[str]):
        self._plan = value

    def submit_plan(
        self,
        *,
        items: list[dict],
        sketches: Optional[list[str]] = None,
    ) -> tuple[bool, str]:
        """Parse plan items, validate rationale, activate the first one; version++.

        Each item dict carries ``text``, ``rationale`` (required),
        and optionally ``keywords`` and ``backing_skill``.
        ``backing_skill`` is validated via ``_validate_backing_skill``.

        Args:
            items: Structured plan items.
            sketches: Optional list of code sketches, one per item.

        Returns (success, message).
        """
        if not items:
            return False, "Plan must contain at least one item."

        parsed_texts: list[str] = []
        item_backing_skills: list[Optional[str]] = []
        item_keywords: list[list[str]] = []
        item_rationales: list[str] = []
        for idx, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            text = (it.get("text") or "").strip()
            if not text:
                continue
            rationale, err = self._validate_rationale(it.get("rationale"))
            if rationale is None:
                return False, (
                    f"Plan rejected: item {idx + 1} ({text[:60]!r}): {err}"
                )
            parsed_texts.append(text)
            item_backing_skills.append(
                (it.get("backing_skill") or "").strip() or None
            )
            item_keywords.append(
                self._sanitize_keywords(it.get("keywords"))
            )
            item_rationales.append(rationale)
        if not parsed_texts:
            return False, (
                "Plan must contain at least one non-empty item."
            )
        min_items = getattr(
            getattr(self.config, "agent", None), "min_items_per_plan", 1,
        )
        if min_items and len(parsed_texts) < min_items:
            return False, (
                f"Plan rejected: only {len(parsed_texts)} item(s) submitted, "
                f"min_items_per_plan is {min_items}. Fresh plans must cover "
                f"several DISTINCT directions so settled history can surface "
                f"patterns. Bundle tightly-related parameter sweeps into ONE "
                f"item's rationale (they count as one direction). If you "
                f"genuinely have only one hypothesis, you are under-exploring — "
                f"call `search_skills(hint=...)` or propose a structural "
                f"alternative before retrying."
            )
        # Diversity gate: reject plans dominated by parameter tuning
        if len(parsed_texts) >= 3:
            div_ok, div_err = _fv.check_plan_diversity(item_keywords)
            if not div_ok:
                return False, f"Plan rejected: {div_err}"
        backing_skills = item_backing_skills
        raw_text = "\n".join(f"- [ ] {t}" for t in parsed_texts)

        # ------------------------------------------------------------
        # Shared tail — behavior below is independent of entry point.
        # ------------------------------------------------------------
        # Record any in-flight (active/pending) items from the outgoing
        # plan as "superseded by replan" BEFORE we rebuild _plan_items.
        # In normal flow the agent only replans from the `replanning` phase
        # where all items are already done_ok/done_fail, so this is a no-op.
        # The paths that actually hit this are:
        #   1. Main agent submitting a fresh update_plan while items
        #      from the previous plan are still active/pending.
        #   2. require_replan() — already retroactively tags the last
        #      failure and rewinds any promoted-but-unrun pending; but
        #      older pending items may still exist and this loop will
        #      re-tag them as "superseded by replan".
        for _item in self._plan_items:
            if _item["status"] in ("active", "pending"):
                self._settled_history.append(
                    self._build_history_entry(
                        item=_item,
                        ok=False,
                        reason="superseded by replan",
                    )
                )

        self._plan_version += 1
        self._plan_items = [
            self._build_plan_item(
                item_id=f"p{i+1}",
                text=text,
                rationale=item_rationales[i],
                keywords=item_keywords[i] if i < len(item_keywords) else None,
                backing_skill=(
                    backing_skills[i]
                    if backing_skills and i < len(backing_skills)
                    else None
                ),
                sketch=(sketches[i] if sketches and i < len(sketches) else ""),
            )
            for i, text in enumerate(parsed_texts)
        ]
        self._plan_items[0]["status"] = "active"
        self._active_item_id = "p1"
        self._plan = raw_text
        self._must_replan = False

        self._persist_plan()
        return True, (
            f"Plan v{self._plan_version} accepted "
            f"({len(parsed_texts)} items). Active: p1"
        )

    def get_active_item(self) -> Optional[dict]:
        """Return the currently active plan item dict, or None."""
        if not self._active_item_id:
            return None
        for item in self._plan_items:
            if item["id"] == self._active_item_id:
                return item
        return None

    def record_skill_acknowledgement(self, item_id: str, ack: dict) -> bool:
        """Record the agent's acknowledgement of the injected backing_skill.

        Stores the structured ack on the item (rendered in plan.md,
        persisted in session.json). When ``ack['applicability'] ==
        'unbind'`` the binding is released for THIS item (backing_skill
        cleared, item stays active as free exploration) and the skill
        is marked ``unbound_at_versions+=plan_version`` in the registry
        — NOT terminal. The skill remains bindable for future items;
        ``SkillBuilder.tier`` just demotes it to tier 2 until a
        successful KEEP promotes it back.

        Returns True if the item was unbound (applicability='unbind'),
        False otherwise.
        """
        unbound = False
        for item in self._plan_items:
            if item["id"] != item_id:
                continue
            item["skill_ack"] = dict(ack)
            if ack.get("applicability") == "unbind":
                bs = item.get("backing_skill")
                if bs and self.skill_builder is not None:
                    reason = (
                        "agent unbound: "
                        + (ack.get("kernel_application") or "")[:200]
                    )
                    self.skill_builder.mark_unbound(
                        skill_name=bs,
                        reason=reason,
                        plan_version=self._plan_version,
                    )
                item["backing_skill"] = None
                unbound = True
            break
        self._persist_plan()
        return unbound

    # -- Settlement helpers (shared by settle / abandon / replan) -----------

    def _close_active_item(
        self,
        *,
        status: str,
        ok: bool,
        reason: str,
        metrics: dict | None = None,
        edit_desc: str = "",
    ) -> tuple[dict | None, str | None]:
        """Mark the active item with *status*, append to history.

        Returns ``(item_dict, backing_skill_name)`` so the caller can
        decide which SkillBuilder notification to send.  Returns
        ``(None, None)`` when there is no active item.

        Does NOT advance, does NOT notify SkillBuilder, does NOT
        persist — those are the caller's responsibility.
        """
        if not self._active_item_id:
            return None, None
        for item in self._plan_items:
            if item["id"] == self._active_item_id:
                item["status"] = status
                backing_skill_name = item.get("backing_skill")
                self._settled_history.append(
                    self._build_history_entry(
                        item=item,
                        ok=ok,
                        reason=reason,
                        metrics=metrics,
                        edit_desc=edit_desc,
                    )
                )
                return item, backing_skill_name
        return None, None

    def validate_edit(self, plan_item_id: str) -> tuple[bool, str]:
        """Check that an edit is allowed given current phase and item ID."""
        if self._must_replan:
            return False, (
                "BLOCKED: Direction change required after diagnostic report. "
                "Submit replacement item(s) via update_plan(items=[...])."
            )
        if self.phase != "active":
            return False, (
                f"BLOCKED: No active plan item (phase={self.phase}). "
                f"Submit a plan first with update_plan(...)."
            )
        if plan_item_id != self._active_item_id:
            return False, (
                f"BLOCKED: plan_item_id='{plan_item_id}' does not match "
                f"active item '{self._active_item_id}'."
            )
        return True, ""

    def settle_active(self, ok: bool, reason: str, metrics: dict,
                      edit_desc: str = ""):
        """Settle the active item after eval. Advances to next pending.

        Args:
            edit_desc: Actual edit description (what was done), stored in
                       history instead of the plan item text.
        """
        # Only store a short status label, not the full error.
        # Detailed errors are already fed back to the agent via messages.
        short_reason = "keep" if ok else self._classify_reason(reason)
        item, bs = self._close_active_item(
            status="done_ok" if ok else "done_fail",
            ok=ok,
            reason=short_reason,
            metrics=metrics,
            edit_desc=edit_desc,
        )
        if item is None:
            return
        if ok and bs and self.skill_builder is not None:
            self.skill_builder.record_applied(
                skill_name=bs,
                plan_version=self._plan_version,
                reason=short_reason,
            )
        self._advance()
        self._persist_plan()

    @staticmethod
    def _classify_reason(reason: str) -> str:
        """Thin wrapper — see :func:`feedback_validation.classify_reason`."""
        return _fv.classify_reason(reason)

    def require_replan(self, diagnosis: Optional[str] = None, *,
                       failing_item_already_settled: bool = False):
        """Abandon the failing plan item and wait for a single
        replacement via ``replace_active_item``.

        Three call shapes, disambiguated by
        ``failing_item_already_settled``:

          **already_settled=True** (post-eval path —
          ``eval_fail`` / ``eval_discard``). ``settle_active`` has
          already closed the failing item and advanced the active
          pointer. Two sub-cases:
            - ``_active_item_id`` is ``None`` (the failed item was
              last in the queue) — just retroactively tag the most-
              recent failure entry.
            - ``_active_item_id`` points to an *un-run* pending item
              that ``_advance`` promoted. Rewind it to ``pending``
              (otherwise the never-executed item gets bogusly marked
              ``done_fail``) and retroactively tag the last history.

          **already_settled=False** (pre-eval path —
          ``edit_fail`` / ``quick_check_fail``). ``settle_active``
          was not called; the active item is still the failing one
          and no history entry exists for this failure. Close the
          active item as ``done_fail`` with reason
          ``abandoned (diagnose)`` so the history reflects the
          direction change (matching the "last failed item has been
          tagged" language the DiagnoseHandler message uses).

        Remaining pending items survive untouched in all cases — the
        agent's ``update_plan`` replacement is inserted before them
        via :meth:`replace_active_item`.

        Args:
            diagnosis: Optional diagnosis text from the diagnose
                subagent. Stored as ``self.last_diagnosis``.
            failing_item_already_settled: Whether the failing plan
                item was closed by ``settle_active`` before this call.
                DiagnoseHandler derives this from ``turn_result.outcome``
                (``eval_*`` → True; ``edit_fail`` / ``quick_check_fail``
                → False).
        """
        self._must_replan = True
        if diagnosis is not None:
            self._last_diagnosis = diagnosis

        if failing_item_already_settled:
            # Post-eval: rewind any promoted-but-unrun pending item,
            # then retag the last failure entry.
            if self._active_item_id:
                for it in self._plan_items:
                    if it["id"] == self._active_item_id \
                            and it["status"] == "active":
                        it["status"] = "pending"
                        break
                self._active_item_id = None
            if self._settled_history:
                last = self._settled_history[-1]
                if not last.get("ok", False):
                    last["reason"] = "abandoned (diagnose)"
                    last["signal_eligible"] = _fv.is_eval_signal(
                        False, "abandoned (diagnose)",
                    )
        else:
            # Pre-eval: the active item never reached settle_active.
            # Close it now so a real history entry records the
            # direction change.
            self._close_active_item(
                status="done_fail",
                ok=False,
                reason="abandoned (diagnose)",
            )
            self._active_item_id = None

        self._persist_plan()

    def replace_active_item(self, new_items: list[dict]) -> tuple[bool, str]:
        """Insert replacement item(s) at the position of the last
        abandoned item, activate the first one, and clear
        ``_must_replan``.

        Called by ``_handle_update_plan`` when ``must_replan`` is
        True. The replacement items are inserted right after the last
        ``done_fail`` item and before the first ``pending`` item, so
        the existing queue is preserved.

        Returns ``(ok, message)`` like ``submit_plan``.
        """
        if not new_items:
            return False, "Replacement must contain at least one item."

        # Find insertion point: right after the last done_fail item.
        insert_idx = 0
        for i, item in enumerate(self._plan_items):
            if item["status"] in ("done_ok", "done_fail"):
                insert_idx = i + 1

        # Assign IDs continuing from the current max.
        max_id = 0
        for item in self._plan_items:
            try:
                max_id = max(max_id, int(item["id"][1:]))
            except (ValueError, IndexError):
                pass

        built: list[dict] = []
        for j, it in enumerate(new_items):
            text = (it.get("text") or "").strip()
            if not text:
                continue
            rationale, err = self._validate_rationale(it.get("rationale"))
            if rationale is None:
                return False, (
                    f"Replacement rejected: item {j + 1} ({text[:60]!r}): {err}"
                )
            built.append(
                self._build_plan_item(
                    item_id=f"p{max_id + j + 1}",
                    text=text,
                    rationale=rationale,
                    keywords=self._sanitize_keywords(it.get("keywords")),
                    backing_skill=(it.get("backing_skill") or "").strip() or None,
                )
            )

        if not built:
            return False, "Replacement must contain at least one non-empty item."

        # Diversity gate (stricter after diagnose: max 1 param-tuning item)
        if len(built) >= 2:
            all_kws = [b.get("keywords") or [] for b in built]
            div_ok, div_err = _fv.check_plan_diversity(all_kws, max_param_items=1)
            if not div_ok:
                return False, (
                    f"Replacement rejected (post-diagnose): {div_err} "
                    f"After a direction change, at least propose structurally "
                    f"different approaches (algorithmic, fusion, memory layout)."
                )

        # Insert and activate the first replacement.
        for j, b in enumerate(built):
            self._plan_items.insert(insert_idx + j, b)
        built[0]["status"] = "active"
        self._active_item_id = built[0]["id"]
        self._must_replan = False

        self._persist_plan()
        return True, (
            f"Replacement accepted: {len(built)} item(s) inserted. "
            f"Active: {self._active_item_id}"
        )

    def _advance(self):
        """Activate next pending item, or clear active (-> replanning)."""
        for item in self._plan_items:
            if item["status"] == "pending":
                item["status"] = "active"
                self._active_item_id = item["id"]
                return
        self._active_item_id = None

    # -- Display ------------------------------------------------------------

    @staticmethod
    def _render_item_line(item: dict) -> str:
        """Thin wrapper — see :func:`feedback_validation.render_item_line`."""
        return _fv.render_item_line(item)

    def _plan_header_lines(self, heading: str = "## Plan Status") -> list[str]:
        """Render the plan header with item counts."""
        backed_count = sum(
            1 for it in self._plan_items if it.get("backing_skill")
        )
        unbound_count = len(self._plan_items) - backed_count
        return [
            heading,
            f"_v{self._plan_version} · {len(self._plan_items)} items "
            f"· {backed_count} skill-backed · {unbound_count} unbound_",
            "",
        ]

    def format_status(self) -> str:
        """Format current plan items + phase hint for injection into messages."""
        if not self._plan_items:
            return "[Plan] No plan. Call update_plan(...) to submit one."

        lines = self._plan_header_lines("## Plan Status")
        active_item = None
        for item in self._plan_items:
            lines.append(self._render_item_line(item))
            if item["status"] == "active":
                active_item = item

        phase = self.phase
        if self._must_replan:
            lines.append(
                "\n⚠ DIRECTION CHANGE REQUIRED. Edits are BLOCKED. "
                "Submit replacement item(s) via update_plan(items=[...])."
            )
        elif phase == "active":
            lines.append(
                f"\nActive item: {self._active_item_id}. "
                f"Use plan_item_id='{self._active_item_id}' in your edits."
            )
        elif phase == "replanning":
            lines.append(
                "\nAll items settled. Submit a NEW plan via "
                "update_plan(items=[...]) with the directions you want "
                "to explore next. Or call finish if done."
            )

        # If the active item has a hint-provided code sketch, show it.
        # The sketch is a compilable skeleton the agent should transcribe
        # directly — it exists precisely to avoid typo/syntax errors that
        # burned previous attempts.
        if active_item and active_item.get("sketch"):
            lines.append(
                f"\n### Code sketch for {active_item['id']} "
                f"(transcribe variable names EXACTLY)\n"
                f"```\n{active_item['sketch']}\n```"
            )

        return "\n".join(lines)

    # Tokens in edit_desc that we DON'T want to surface as "repeated
    # across edits" — they are filler or action verbs that carry no
    # directional signal. If the agent writes three fails that all
    # say "tune BLOCK_N to X", the meaningful repeated signal is
    # BLOCK_N, not "tune" — but the naive token extractor would
    # surface both and the follow-up nudge "avoid directions matching
    # the repeated optimization keywords above" could be misread as
    # "stop tuning anything", which is actively harmful.
    _PREV_SUMMARY_TOKEN_STOPWORDS = frozenset({
        # Articles / prepositions / connectives / common filler
        "the", "a", "an", "with", "and", "for", "to", "of", "in", "on",
        "from", "by", "at", "is", "as", "or", "not", "be", "this",
        "that", "it", "its", "into", "also", "then", "again", "still",
        "now", "different", "same", "each", "every", "multiple", "new",
        "old", "both", "more", "less", "than",
        # Generic action verbs (present / past / gerund)
        "add", "added", "adding", "use", "used", "using",
        "try", "tried", "trying", "test", "tested", "testing",
        "remove", "removed", "removing", "change", "changed",
        "changing", "update", "updated", "updating", "fix", "fixed",
        "fixing", "set", "setting", "make", "made", "making",
        "keep", "kept", "keeping", "run", "running", "pass",
        "passes", "passing",
        "tune", "tuned", "tuning", "sweep", "swept", "sweeping",
        "adjust", "adjusted", "adjusting", "tweak", "tweaked",
        "explore", "explored", "exploring", "experiment",
        "apply", "applied", "applying", "introduce", "introduced",
        "enable", "enabled", "disable", "disabled",
        "replace", "replaced", "replacing", "swap", "swapped",
        "swapping", "split", "splitting", "merge", "merged",
        "merging", "combine", "combined", "combining",
        "implement", "implemented", "implementing",
        "rewrite", "rewrote", "refactor", "refactored",
        "increase", "increased", "increasing",
        "decrease", "decreased", "decreasing",
        "reduce", "reduced", "reducing", "raise", "raised",
        "lower", "lowered", "lowering", "shrink", "shrunk",
        "expand", "expanded", "expanding",
        "boost", "boosted", "improve", "improved", "improving",
        "optimize", "optimized", "optimizing",
        "optimise", "optimised", "optimising",
        "enhance", "enhanced",
        # Comparatives / subjective qualifiers
        "smaller", "larger", "bigger", "simpler", "faster",
        "slower", "better", "worse", "higher", "lower",
        "stronger", "weaker",
    })

    def format_prev_plan_summary(self, prev_version: int) -> str:
        """Return a compact markdown block summarizing plan v{prev_version}.

        Called by ``_handle_update_plan`` right after a fresh
        ``submit_plan`` bumps the version — gives the agent a distilled
        view of the plan that just ended, so the next plan doesn't
        silently repeat the same directions.

        Pure statistics over ``_settled_history`` filtered to
        ``entry["version"] == prev_version``. Zero LLM cost.

        Empty string when there is nothing to report (version has no
        settled entries, or ``prev_version < 1``).
        """
        if prev_version < 1 or not self._settled_history:
            return ""
        entries = [e for e in self._settled_history
                   if e.get("version") == prev_version]
        if not entries:
            return ""

        # -- counts by settlement shape ------------------------------------
        keeps = [e for e in entries if e.get("ok")]
        fails = [e for e in entries if not e.get("ok")]
        # Control events (superseded / diagnose-abandoned / screened)
        # should not be framed as "failures the agent chose" — split
        # them out via ``signal_eligible``.
        real_fails = [e for e in fails if e.get("signal_eligible", True)]
        control = [e for e in fails if not e.get("signal_eligible", True)]
        # DISCARD vs FAIL classification: DISCARD entries' reason is
        # "no improvement" (see settle_active → _classify_reason).
        discards = [e for e in real_fails if e.get("reason") == "no improvement"]
        hard_fails = [e for e in real_fails
                      if e.get("reason") != "no improvement"]

        # -- best metric trajectory ---------------------------------------
        primary = getattr(self.config, "primary_metric", "latency_us")
        lower_is_better = getattr(self.config, "lower_is_better", True)
        kept_metrics = []
        for e in keeps:
            m = (e.get("metrics") or {}).get(primary)
            if isinstance(m, (int, float)):
                kept_metrics.append(float(m))
        if kept_metrics:
            best_kept = min(kept_metrics) if lower_is_better else max(kept_metrics)
            metric_line = (
                f"- Best {primary} recorded this version: {best_kept:g} "
                f"(from {len(kept_metrics)} KEEP{'s' if len(kept_metrics) != 1 else ''})"
            )
        else:
            metric_line = f"- No KEEP this version — best {primary} unchanged."

        # -- failure reason counter --------------------------------------
        reason_counter = Counter(
            (e.get("reason") or "").strip() for e in real_fails
        )
        reason_str = (
            ", ".join(
                f"{r}×{n}" if n > 1 else r
                for r, n in reason_counter.most_common()
                if r
            )
            or "n/a"
        )

        # -- repeated tokens in edit_desc --------------------------------
        # SCOPE: only ``real_fails`` — the same set that feeds
        # ``reason_counter`` above. Including KEEPs would flag
        # validated directions as "avoid"; including control events
        # (superseded / abandoned (diagnose)) would promote
        # never-actually-run items into repeated-failure signal.
        # Both would produce misleading "avoid this direction"
        # guidance in the follow-up nudge.
        token_counter: Counter[str] = Counter()
        word_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
        for e in real_fails:
            desc = (e.get("text") or "").lower()
            for tok in word_re.findall(desc):
                if tok in self._PREV_SUMMARY_TOKEN_STOPWORDS:
                    continue
                if len(tok) < 3:
                    continue
                token_counter[tok] += 1
        repeated = [(t, n) for t, n in token_counter.most_common(6) if n >= 2]
        repeated_str = (
            ", ".join(f"{t}×{n}" for t, n in repeated) or "none"
        )

        # -- backing skills exercised ------------------------------------
        bs_counter = Counter(
            (e.get("backing_skill") or "").strip() for e in entries
        )
        bs_counter.pop("", None)
        bs_str = (
            ", ".join(f"{s}×{n}" for s, n in bs_counter.most_common())
            or "none"
        )

        # -- assemble ----------------------------------------------------
        outcome_parts = [f"{len(keeps)} KEEP"]
        if discards:
            outcome_parts.append(f"{len(discards)} DISCARD")
        if hard_fails:
            outcome_parts.append(f"{len(hard_fails)} FAIL")
        if control:
            outcome_parts.append(f"{len(control)} control-event")
        outcome_line = ", ".join(outcome_parts)

        lines = [
            f"## Previous plan (v{prev_version}) summary",
            f"- {len(entries)} settled item(s): {outcome_line}",
            metric_line,
            f"- Failure reasons: {reason_str}",
            f"- Repeated optimization keywords: {repeated_str}",
            f"- Backing skills exercised: {bs_str}",
            "",
            "Avoid proposing items whose direction matches the repeated "
            "optimization keywords above unless you have a concrete "
            "reason the previous attempts were insufficient.",
        ]
        return "\n".join(lines)

    def format_history(self, max_versions: int = 0) -> str:
        """Format settled_history grouped by plan version.

        Args:
            max_versions: If > 0, only show the most recent N versions in detail.
                          Older versions get a one-line summary.
        """
        if not self._settled_history:
            return ""

        # Group by version
        versions: dict[int, list[dict]] = {}
        for entry in self._settled_history:
            versions.setdefault(entry["version"], []).append(entry)

        ver_nums = sorted(versions.keys())
        if max_versions > 0 and len(ver_nums) > max_versions:
            detail_vers = set(ver_nums[-max_versions:])
            summary_vers = ver_nums[:-max_versions]
        else:
            detail_vers = set(ver_nums)
            summary_vers = []

        lines = ["## Optimization History (do NOT repeat failed directions)"]

        for ver in summary_vers:
            items = versions[ver]
            ok_count = sum(1 for e in items if e["ok"])
            fail_count = len(items) - ok_count
            lines.append(f"### Plan v{ver} (summary: {ok_count} kept, {fail_count} failed)")

        for ver in sorted(detail_vers):
            lines.append(f"### Plan v{ver}")
            for entry in versions[ver]:
                tag = "[O]" if entry["ok"] else "[X]"
                reason = entry["reason"]
                metrics = entry.get("metrics", {})
                metric_str = ""
                if metrics:
                    primary = self.config.primary_metric
                    if primary in metrics:
                        metric_str = f", {primary}={metrics[primary]}"
                # Item id + backing skill: makes three similarly-worded
                # edit_desc lines distinguishable at a glance and links
                # each history row back to a plan item / skill.
                item_id = entry.get("item_id") or "?"
                bs = entry.get("backing_skill") or ""
                bs_suffix = f" (bs: {bs})" if bs else ""
                lines.append(
                    f"- {tag} [{item_id}] {entry['text']}  "
                    f"-- {reason}{metric_str}{bs_suffix}"
                )
        return "\n".join(lines)

    # -- Plan file output --------------------------------------------------

    def format_plan_file(self) -> str:
        """Format full plan state for plan.md persistence (no truncation).

        If a non-empty SkillBuilder is attached, its state is appended at
        the end as a ``## Skill State`` section so that plan.md acts as the
        single source of truth for both plan items and skill lifecycle.
        Compact cycles read plan.md into [STATE_ATTACHMENT] and get skill
        state for free.
        """
        lines = []
        if self._plan_items:
            lines.extend(self._plan_header_lines(f"# Plan v{self._plan_version}"))
            for item in self._plan_items:
                lines.append(self._render_item_line(item))
                if item.get("sketch"):
                    lines.append(f"\n  <details><summary>code sketch</summary>\n")
                    lines.append(f"  ```\n{item['sketch']}\n  ```\n  </details>\n")

        history = self.format_history(max_versions=0)
        if history:
            lines.append("")
            lines.append(history)

        # Append skill state (conditional: only if skill_builder exists
        # AND has at least one skill registered). Empty SkillBuilder → no
        # Skill State section appears, so autoresearch runs without hint
        # triggering produce plan.md identical to Phase 0.
        if self.skill_builder is not None and not self.skill_builder.is_empty():
            skill_md = self.skill_builder.render_for_plan_md(self._plan_items)
            if skill_md:
                lines.append("")
                lines.append(skill_md)

        return "\n".join(lines) if lines else ""

    # -- Serialization ------------------------------------------------------

    def plan_state_to_dict(self) -> dict:
        return {
            "plan": self._plan,
            "plan_items": self._plan_items,
            "active_item_id": self._active_item_id,
            "plan_version": self._plan_version,
            "settled_history": self._settled_history,
            "must_replan": self._must_replan,
        }

    def plan_state_from_dict(self, d: dict):
        self._plan = d.get("plan")
        self._plan_items = d.get("plan_items", [])
        self._active_item_id = d.get("active_item_id")
        self._plan_version = d.get("plan_version", 0)
        self._settled_history = d.get("settled_history", [])
        self._must_replan = d.get("must_replan", False)

        # One-shot migration: backfill ``signal_eligible`` on history
        # entries that predate the field. Without this, old sessions
        # would carry control events (superseded / diagnose-replan)
        # into format_history_signal as if they were real failures,
        # re-introducing the noise this field was added to suppress.
        for entry in self._settled_history:
            if "signal_eligible" not in entry:
                entry["signal_eligible"] = self._is_eval_signal(
                    entry.get("ok", False), entry.get("reason", ""),
                )

    # -- Derived signals over settled_history -------------------------------

    def format_history_signal(self, lookback: int = 5) -> str:
        """Return at most one pattern hint line from recent settled history.

        Only entries flagged ``signal_eligible`` (real KEEP/FAIL/DISCARD
        eval outcomes) are considered — control events like
        ``superseded by replan`` / ``abandoned (diagnose)`` are filtered
        out upstream by ``_build_history_entry`` so they cannot be
        misclassified as failure trends.

        Empty string when there is no clear signal — silence is the
        default. Single line, single signal: avoid stacking hints to
        keep eval feedback noise low.

        Priority order:
          1. repeated specific failure reason (>= 3 times)
          2. consecutive DISCARD bucket (>= 3)
          3. high FAIL bucket (>= 4)
        """
        # Filter first, then take the last `lookback` so the window
        # spans real eval outcomes regardless of how many control
        # events were interleaved.
        eligible = [
            e for e in self._settled_history
            if e.get("signal_eligible", True)
        ]
        recent = eligible[-lookback:]
        if len(recent) < 3:
            return ""

        from collections import Counter

        fail_count = sum(1 for e in recent if not e.get("ok"))
        discard_count = sum(
            1 for e in recent if e.get("reason") == "no improvement"
        )
        reason_counts = Counter(
            e.get("reason", "") for e in recent if not e.get("ok")
        )
        top_reason, top_n = next(
            iter(reason_counts.most_common(1)), (None, 0),
        )

        # Generic fail/keep buckets are too vague to be actionable.
        _vague = {None, "", "fail", "keep", "no improvement"}

        if top_n >= 3 and top_reason not in _vague:
            return (
                f"[Pattern] {top_n}/{len(recent)} recent attempts failed "
                f"with {top_reason!r} — fix the root cause, not retries."
            )
        if discard_count >= 3:
            return (
                f"[Pattern] {discard_count}/{len(recent)} recent attempts "
                f"settled DISCARD (correct but no gain). Try a different "
                f"magnitude or approach."
            )
        if fail_count >= 4:
            return (
                f"[Pattern] {fail_count}/{len(recent)} recent attempts "
                f"FAILed. Step back: read the kernel and pick a different "
                f"direction."
            )
        return ""

    def format_goal_anchor(self, best_metric_str: str = "") -> str:
        """One-line goal restatement for compact bootstrap.

        Format: ``Goal: minimize <metric>, current best <str>.``
        Used by ``compress._build_bootstrap`` so the agent regains the
        big picture after every context compression.
        """
        direction = "minimize" if self.config.lower_is_better else "maximize"
        metric = self.config.primary_metric
        if best_metric_str:
            return f"Goal: {direction} {metric}, current best {best_metric_str}."
        return f"Goal: {direction} {metric}."

    # -- Eval feedback ------------------------------------------------------

    def build_eval_feedback(
        self,
        eval_record: dict,
        eval_calls_made: int,
        max_rounds: int,
        best_result,
    ) -> str:
        """Build eval feedback message. Pure display — no side effects."""
        eval_status = eval_record.get("status", "FAIL")
        eval_metrics = eval_record.get("metrics", {})
        primary_val = eval_metrics.get(self.config.primary_metric)

        lines = [
            f"[System] Evaluation complete — **{eval_status}**",
            f"Eval round {eval_calls_made}/{max_rounds}",
        ]

        if eval_status == "KEEP":
            lines.append(
                f"Change accepted! {self.config.primary_metric}={primary_val}")
        elif eval_status == "FAIL":
            fail_reason = eval_record.get("fail_reason", "unknown")
            lines.append(f"Failed: {fail_reason}")
        else:  # DISCARD — correct but no improvement
            best_val = (best_result.metrics.get(self.config.primary_metric)
                        if best_result else None)
            if primary_val is not None and best_val is not None:
                lines.append(
                    f"No improvement: {primary_val} vs best {best_val}")

        if eval_status != "KEEP":
            lines.append(
                "Code rolled back to last KEEP snapshot (already the best state).")

        raw_tail = eval_record.get("raw_output_tail", "")
        if raw_tail and eval_status != "KEEP":
            tail_limit = self.config.agent.eval_feedback_tail
            lines.append(f"Eval output (tail):\n```\n{raw_tail[-tail_limit:]}\n```")

        if best_result:
            bv = best_result.metrics.get(self.config.primary_metric)
            lines.append(f"Current best: {self.config.primary_metric}={bv}")

        remaining = max_rounds - eval_calls_made
        if remaining <= self.config.agent.finish_hint_threshold:
            lines.append("⚠ Budget nearly exhausted — consider calling finish.")

        signal = self.format_history_signal()
        if signal:
            lines.append(signal)

        # Reflection prompt on non-KEEP outcomes. Short, concrete
        # questions focus the next plan item on STRUCTURAL change —
        # matching the "avoid parameter-only tuning" guidance the
        # protocol already carries. Purely textual; no side effects.
        if eval_status in ("FAIL", "DISCARD"):
            lines.append(self._render_reflection_prompt(eval_status,
                                                        eval_record))

        lines.append(f"\n{self.format_status()}")

        return "\n".join(lines)

    @staticmethod
    def _render_reflection_prompt(eval_status: str,
                                  eval_record: dict) -> str:
        """Short reflection ribbon appended after FAIL / DISCARD.

        Asks three concrete questions: what was the hypothesis, why
        did metrics go the wrong way, and what STRUCTURAL (not
        parameter) change comes next. Kept brief so it doesn't
        balloon the feedback message."""
        reason = eval_record.get("fail_reason") or ""
        if eval_status == "FAIL" and reason:
            hypothesis_hint = f"You last claimed: *{reason[:180]}*. "
        else:
            hypothesis_hint = ""
        return (
            "\n## Reflection (answer implicitly before the next item)\n"
            f"{hypothesis_hint}"
            "1. What was the SPECIFIC bottleneck the last edit tried "
            "to relieve?\n"
            "2. Why did metrics move the wrong way — was the "
            "hypothesis wrong, or the implementation?\n"
            "3. What STRUCTURAL change (algorithm / access pattern / "
            "memory hierarchy / kernel fusion-split) should come next? "
            "Parameter-only tuning is a weak follow-up."
        )

    # -- Nudge messages ------------------------------------------------------

    def build_phase_nudge(self) -> str | None:
        """Return a phase-appropriate nudge message, or None if no nudge needed."""
        phase = self.phase
        if self._must_replan:
            return (
                "[System] ⚠ DIRECTION CHANGE REQUIRED. "
                "Submit replacement item(s) via update_plan(items=[...])."
            )
        if phase == "no_plan":
            return "[System] Submit a plan first: call update_plan(...)."
        if phase == "active":
            aid = self._active_item_id
            return (
                f"[System] Active item: {aid}. "
                f"Make edits with plan_item_id='{aid}'."
            )
        if phase == "replanning":
            return (
                "[System] All items settled. Submit a NEW multi-item "
                "plan via update_plan(items=[...]), or finish."
            )
        return None

    # -- Failure feedback --------------------------------------------------

    def build_failure_feedback(self, failed_edits: list, total_edits: int) -> str:
        """Build feedback message for edit failures (atomic rollback)."""
        feedback = (
            f"ATOMIC ROLLBACK: {len(failed_edits)} of {total_edits} edits failed, "
            f"all changes reverted. Errors: "
            + "; ".join(e["result"].message for e in failed_edits)
        )
        feedback += f"\n\n{self.format_status()}"
        return f"[System] {feedback}"

    def build_quick_check_feedback(self, qc_message: str) -> str:
        """Build feedback for quick_check failure."""
        qc_feedback = (
            f"[System] Quick check FAILED (no eval budget spent). "
            f"All edits rolled back.\n{qc_message}\n"
            f"Fix the error and try again."
        )
        qc_feedback += f"\n\n{self.format_status()}"
        return qc_feedback
