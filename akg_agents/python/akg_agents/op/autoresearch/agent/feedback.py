"""
FeedbackBuilder — Constructs system feedback messages for the agent.

Manages:
  - Structured plan items with enforced execution order
  - Auto-settlement of items after eval
  - Settled history across plan versions
  - Eval/failure/quick_check feedback message construction
"""

from typing import Optional


class FeedbackBuilder:
    """Builds feedback messages injected into the agent conversation."""

    def __init__(self, config):
        self.config = config
        self._plan: Optional[str] = None
        self._plan_items: list[dict] = []       # [{id, text, status}]
        self._active_item_id: Optional[str] = None
        self._plan_version: int = 0
        self._settled_history: list[dict] = []  # [{text, ok, reason, metrics, version}]
        self._must_replan: bool = False         # Set by auto-diagnose, blocks edits

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

    # -- Plan state --------------------------------------------------------

    @property
    def plan(self) -> Optional[str]:
        return self._plan

    @plan.setter
    def plan(self, value: Optional[str]):
        self._plan = value

    def submit_plan(self, raw_text: str) -> tuple[bool, str]:
        """Parse '- [ ]' lines into items, activate first, version++.

        Returns (success, message).
        """
        items = []
        for line in raw_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("- [ ]"):
                text = stripped[5:].strip()
                if text:
                    items.append(text)
        if not items:
            return False, "Plan must contain at least one '- [ ]' item."

        self._plan_version += 1
        self._plan_items = [
            {"id": f"p{i+1}", "text": text, "status": "pending"}
            for i, text in enumerate(items)
        ]
        self._plan_items[0]["status"] = "active"
        self._active_item_id = "p1"
        self._plan = raw_text
        self._must_replan = False
        return True, f"Plan v{self._plan_version} accepted ({len(items)} items). Active: p1"

    def validate_edit(self, plan_item_id: str) -> tuple[bool, str]:
        """Check that an edit is allowed given current phase and item ID."""
        if self._must_replan:
            return False, (
                "BLOCKED: Direction change required after diagnostic report. "
                "Call update_plan(plan=...) with a NEW plan before editing."
            )
        if self.phase != "active":
            return False, (
                f"BLOCKED: No active plan item (phase={self.phase}). "
                f"Submit a plan first with update_plan(plan=...)."
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
        if not self._active_item_id:
            return
        for item in self._plan_items:
            if item["id"] == self._active_item_id:
                item["status"] = "done_ok" if ok else "done_fail"
                self._settled_history.append({
                    "text": edit_desc or item["text"],
                    "ok": ok,
                    "reason": self._summarize_reason(reason) if not ok else reason,
                    "metrics": metrics,
                    "version": self._plan_version,
                })
                break
        self._advance()

    @staticmethod
    def _summarize_reason(reason: str) -> str:
        """Extract a short label from a raw error string."""
        if not reason or len(reason) < 120:
            return reason
        import re
        if re.search(r"timed out", reason):
            return "timeout"
        if re.search(r"验证失败|correctness|mismatch", reason):
            return "correctness error"
        return "runtime/compile error"

    def require_replan(self):
        """Force a replan — blocks edits until update_plan is called.

        Active item is marked done_fail and recorded in history.
        Pending items are dropped (never attempted).
        """
        self._must_replan = True
        for item in self._plan_items:
            if item["status"] == "active":
                item["status"] = "done_fail"
                self._settled_history.append({
                    "text": item["text"],
                    "ok": False,
                    "reason": "abandoned (forced replan)",
                    "metrics": {},
                    "version": self._plan_version,
                })
        self._plan_items = [i for i in self._plan_items if i["status"] != "pending"]
        self._active_item_id = None

    def _advance(self):
        """Activate next pending item, or clear active (-> replanning)."""
        for item in self._plan_items:
            if item["status"] == "pending":
                item["status"] = "active"
                self._active_item_id = item["id"]
                return
        self._active_item_id = None

    # -- Display ------------------------------------------------------------

    def format_status(self) -> str:
        """Format current plan items + phase hint for injection into messages."""
        if not self._plan_items:
            return "[Plan] No plan. Call update_plan(plan=...) to submit one."

        lines = ["## Plan Status"]
        for item in self._plan_items:
            if item["status"] == "active":
                marker = f">>> [{item['id']}]"
            elif item["status"] == "done_ok":
                marker = f"[O] [{item['id']}]"
            elif item["status"] == "done_fail":
                marker = f"[X] [{item['id']}]"
            else:  # pending
                marker = f"[ ] [{item['id']}]"
            lines.append(f"- {marker} {item['text']}")

        phase = self.phase
        if self._must_replan:
            lines.append(
                "\n⚠ DIRECTION CHANGE REQUIRED. Edits are BLOCKED. "
                "Call update_plan(plan=...) with a new plan."
            )
        elif phase == "active":
            lines.append(
                f"\nActive item: {self._active_item_id}. "
                f"Use plan_item_id='{self._active_item_id}' in your edits."
            )
        elif phase == "replanning":
            lines.append(
                "\nAll items settled. Call update_plan(plan=...) for a new plan, or finish."
            )

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
                lines.append(f"- {tag} {entry['text']}  -- {reason}{metric_str}")
        return "\n".join(lines)

    # -- Plan file output --------------------------------------------------

    def format_plan_file(self) -> str:
        """Format full plan state for plan.md persistence (no truncation)."""
        lines = []
        if self._plan_items:
            lines.append(f"# Plan v{self._plan_version}\n")
            for item in self._plan_items:
                if item["status"] == "active":
                    marker = f">>> [{item['id']}]"
                elif item["status"] == "done_ok":
                    marker = f"[O] [{item['id']}]"
                elif item["status"] == "done_fail":
                    marker = f"[X] [{item['id']}]"
                elif item["status"] == "skipped":
                    marker = f"[-] [{item['id']}]"
                else:
                    marker = f"[ ] [{item['id']}]"
                lines.append(f"- {marker} {item['text']}")

        history = self.format_history(max_versions=0)
        if history:
            lines.append("")
            lines.append(history)

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
            lines.append(f"Eval output (tail):\n```\n{raw_tail[-1000:]}\n```")

        if best_result:
            bv = best_result.metrics.get(self.config.primary_metric)
            lines.append(f"Current best: {self.config.primary_metric}={bv}")

        remaining = max_rounds - eval_calls_made
        if remaining <= self.config.agent.finish_hint_threshold:
            lines.append("⚠ Budget nearly exhausted — consider calling finish.")

        lines.append(f"\n{self.format_status()}")

        return "\n".join(lines)

    # -- Nudge messages ------------------------------------------------------

    def build_phase_nudge(self) -> str | None:
        """Return a phase-appropriate nudge message, or None if no nudge needed."""
        phase = self.phase
        if self._must_replan:
            return (
                "[System] ⚠ DIRECTION CHANGE REQUIRED. "
                "Call update_plan(plan=...) with a new plan."
            )
        if phase == "no_plan":
            return "[System] Submit a plan first: call update_plan(plan=...)."
        if phase == "active":
            aid = self._active_item_id
            return (
                f"[System] Active item: {aid}. "
                f"Make edits with plan_item_id='{aid}'."
            )
        if phase == "replanning":
            return (
                "[System] All items settled. "
                "Call update_plan(plan=...) for a new plan, or finish."
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
