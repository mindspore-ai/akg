"""
SubagentSession — Reusable LLM tool-use loop for autoresearch subagents.

Provides the shared `_run_loop` used by every autoresearch subagent:
LLM call → tool dispatch (read_file + one exit tool) → iterate until
exit or max_iterations. Can be one-shot or persistent.

Also hosts the two entry points that drive the loop's post-eval
safety nets:

  - ``run_diagnostic_subagent`` — free function, one-shot, returns a
    structured ``Root cause / Fix / Avoid`` report. Driven by
    ``DiagnoseHandler.apply``.

  - ``DiagnoseHandler`` — owns the diagnose decide → apply seam.
    AgentLoop constructs one instance at init and delegates the
    should_fire check + the apply side (run the subagent, store the
    report on ``feedback.last_diagnosis``, inject the mandatory
    direction change, call ``save_session``) to this handler.
    ``decide`` is trivial (a threshold check), so it is exposed as
    ``should_fire(turn_result) -> bool`` instead of returning a
    ``PostEvalDecision``; the loop's ``_decide_post_eval_action``
    still wraps the result in ``PostEvalDecision(kind="diagnose")``.
"""

import logging
from typing import Callable, Optional

from .tools import execute_read_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas shared by all subagents (read_file + an exit tool).
# ---------------------------------------------------------------------------

SUBAGENT_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file within the project.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "finish_diagnosis",
        "description": (
            "Call this when you have gathered enough information to produce "
            "your diagnosis. Pass your full analysis as the 'report' argument. "
            "This immediately ends the diagnostic session."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "report": {
                    "type": "string",
                    "description": (
                        "Your complete diagnosis in the format: "
                        "**Root cause**: ... / **Fix**: ... / **Avoid**: ..."
                    ),
                },
            },
            "required": ["report"],
        },
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_code_section(current_code: dict, truncate: int = 8_000) -> str:
    """Format code dict into markdown for subagent prompts."""
    parts = []
    for fname, content in current_code.items():
        truncated = content[:truncate] if len(content) > truncate else content
        parts.append(f"\n### {fname}\n```\n{truncated}\n```\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# SubagentSession — shared LLM tool-use loop for all subagents.
# Diagnostic, hint, and review subagents all use the same loop.
# ---------------------------------------------------------------------------

class SubagentSession:
    """Reusable LLM tool-use session for subagents.

    Provides a shared _run_loop that handles: LLM call → tool dispatch
    (read_file + one exit tool) → iterate until exit or max_iterations.

    Can be used as one-shot (diagnose) or persistent (hint + review).
    """

    def __init__(self, llm, system_prompt: str, task_dir: str,
                 max_iterations: int = 15, result_truncate: int = 10_000,
                 read_whitelist: list[str] | None = None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.task_dir = task_dir
        self.max_iterations = max_iterations
        self.result_truncate = result_truncate
        self.read_whitelist = read_whitelist  # e.g. ["docs/"] — None = allow all
        self.sub_msgs: list = []

    async def run(self, prompt: str, tools: list, exit_tool: str,
                  label: str) -> dict:
        """Start a fresh session with a prompt. Returns exit tool arguments."""
        self.sub_msgs = [{"role": "user", "content": prompt}]
        return await self._resume(tools, exit_tool, label)

    async def resume(self, followup: str, tools: list, exit_tool: str,
                     label: str) -> dict:
        """Continue an existing session with a follow-up message."""
        self.sub_msgs.append({"role": "user", "content": followup})
        return await self._resume(tools, exit_tool, label)

    async def _resume(self, tools: list, exit_tool: str, label: str) -> dict:
        """Shared LLM tool-use loop. Returns the exit tool's arguments,
        or {"_text": ...} if the LLM stopped without calling the exit tool."""
        for iteration in range(self.max_iterations):
            logger.info(f"  [{label}] iteration {iteration + 1}/{self.max_iterations}")

            response = await self.llm.call(
                self.system_prompt, self.sub_msgs, tools=tools)
            self.llm.append_assistant(self.sub_msgs, response)

            text = self.llm.get_response_text(response)
            if text:
                logger.info(f"  [{label}] {text[:300]}")

            if self.llm.get_stop_reason(response) != "tool_use":
                return {"_text": text or ""}

            tool_calls = self.llm.extract_tool_calls(response)

            tool_results = []
            for tc in tool_calls:
                if tc["tool_name"] == exit_tool:
                    logger.info(f"  [{label}] {exit_tool} called (iter {iteration + 1})")
                    return tc["arguments"]
                if tc["tool_name"] == "read_file":
                    path = tc["arguments"].get("path", "")
                    logger.info(f"  [{label}] read_file: {path}")
                    # Enforce whitelist
                    if (self.read_whitelist is not None
                            and not any(path.startswith(p)
                                        for p in self.read_whitelist)):
                        content = f"BLOCKED: read_file only allowed for: {', '.join(self.read_whitelist)}"
                    else:
                        result = execute_read_file(path, self.task_dir)
                        content = result.message[:self.result_truncate]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["tool_use_id"],
                        "content": content,
                    })

            if tool_results:
                self.sub_msgs.append({"role": "user", "content": tool_results})

        return {"_text": f"({label} reached max iterations)"}


# ---------------------------------------------------------------------------
# Diagnostic subagent — one-shot, uses SubagentSession
# ---------------------------------------------------------------------------

_DIAGNOSE_SYSTEM = "You are a diagnostic agent. Analyze errors and suggest fixes."


async def run_diagnostic_subagent(
    llm,
    error_context: str,
    current_code: dict,
    task_description: str,
    task_dir: str,
    knowledge_prompt: str = None,
    max_iterations: int = 15,
    code_truncate: int = 8_000,
    result_truncate: int = 10_000,
    failure_history: str = "",
    current_plan: str = "",
) -> str:
    """Spawn a one-shot subagent to diagnose a persistent failure."""
    code_section = _format_code_section(current_code, code_truncate)

    # Build optional history/plan sections
    history_section = ""
    if failure_history:
        history_section = f"## Recent Failures (do NOT repeat these directions)\n{failure_history}\n\n"
    plan_section = ""
    if current_plan:
        plan_section = f"## Current Plan (what was attempted)\n{current_plan}\n\n"

    prompt = (
        f"You are a diagnostic agent. The optimization agent is stuck with repeated failures.\n\n"
        f"## Task\n{task_description}\n\n"
        f"{history_section}"
        f"{plan_section}"
        f"## Current Code\n{code_section}\n"
        f"## Problem\n{error_context}\n\n"
        f"## Instructions\n"
        f"1. First, analyze: **what pattern caused these repeated failures?**\n"
        f"2. Use read_file to examine relevant files (eval script, reference, config) **only if needed**.\n"
        f"3. Identify the ROOT CAUSE of the repeated failures.\n"
        f"4. When you have enough information, call **finish_diagnosis** with your analysis.\n"
        f"   Format your report as:\n"
        f"   **Root cause**: (1-2 sentences)\n"
        f"   **Fix**: (>= 3 specific, STRUCTURALLY different approaches — algorithmic changes, "
        f"fusion, memory layout, data flow restructuring. NOT parameter tuning.)\n"
        f"   **Avoid**: (what patterns from the failed attempts must NOT be repeated)\n"
        f"\n## Error Pattern Checklist\n"
        f"Match failures to these patterns FIRST, then propose fixes that address the matched pattern:\n"
        f"- **A. Signature mismatch**: kernel def has param X but call site omits it → `missing argument` / `KeyError`\n"
        f"- **B. constexpr/runtime confusion**: runtime var where `tl.constexpr` needed → `CompilationError`\n"
        f"- **C. Timeout**: grid too large, loop unbounded, OOM → kernel never returns\n"
        f"- **D. Correctness**: wrong axis, broadcasting, dtype, off-by-one → output mismatch\n"
        f"- **E. Repeated same approach**: every attempt tries identical optimization → planning failure\n"
        f"- **F. API mismatch**: using unsupported Triton op for this backend → `ImportError` / `AttributeError`\n"
        f"- **G. Shape/stride**: hardcoded dims vs actual input, non-contiguous memory → index errors\n\n"
        f"\nCRITICAL CONSTRAINTS:\n"
        f"- Proposed fixes must be STRUCTURALLY different from each other and from failed attempts.\n"
        f"- At most 1 out of 3+ fixes may be parameter tuning; the rest MUST be algorithmic/structural.\n"
        f"- If the same error message appears across multiple failures, the fix must address "
        f"the error directly (e.g., missing argument → add the argument), not work around it.\n"
        f"\nIMPORTANT: Do NOT read files unnecessarily. If the problem and current code "
        f"already give you enough context, call finish_diagnosis immediately. "
        f"Be specific and actionable. The optimization agent will use your diagnosis directly."
    )
    system = _DIAGNOSE_SYSTEM
    if knowledge_prompt:
        system = knowledge_prompt + "\n\n" + _DIAGNOSE_SYSTEM

    session = SubagentSession(llm, system, task_dir,
                              max_iterations=max_iterations,
                              result_truncate=result_truncate)
    result = await session.run(prompt, SUBAGENT_TOOLS, "finish_diagnosis", "diagnose")
    if "_text" in result:
        return result["_text"] or "(no diagnosis)"
    return result.get("report", "(empty diagnosis)")


# ---------------------------------------------------------------------------
# DiagnoseHandler — owns the diagnose decide → apply seam
# ---------------------------------------------------------------------------
#
# AgentLoop constructs one instance at init and delegates the entire
# diagnose safety net to this handler. The post-eval dispatcher asks
# ``should_fire(turn_result)`` (a cheap threshold check) and, if yes,
# awaits ``apply(turn_result)`` which drives the LLM subagent, mutates
# feedback / counters / buffer, and calls save_session.


class DiagnoseHandler:
    """Owns the diagnose subagent's decide → apply seam.

    Fires when the consecutive-failure counter crosses a multiple of
    ``config.agent.diagnose_suggest_threshold``. Runs the diagnostic
    subagent on a fresh ``SubagentSession``, stores the report via
    ``feedback.require_replan`` (which also sets the must_replan
    flag so further edits are blocked until the agent submits a new
    plan), injects the mandatory-direction-change message into the
    buffer, resets the consecutive-failure counter, and calls the
    ``save_session`` callback.

    The handler captures references to the loop's state bags once at
    construction time. ``knowledge_prompt`` is captured at init too
    — it is static for the lifetime of a run (built once in
    ``build_system_prompt``) and never changes.

    Heartbeat hook: the optional ``heartbeat_cb`` is called with a
    short ``extra`` string when the subagent starts, so monitoring
    tooling can observe the transition out of the main loop.
    """

    def __init__(
        self,
        *,
        llm,
        config,
        task_dir: str,
        runner,
        counters,
        feedback,
        buffer,
        knowledge_prompt: str,
        verbose: bool = True,
        save_session_cb: Optional[Callable[[], None]] = None,
        heartbeat_cb: Optional[Callable[[str], None]] = None,
    ):
        self._llm = llm
        self._config = config
        self._task_dir = task_dir
        self._runner = runner
        self._counters = counters
        self._feedback = feedback
        self._buffer = buffer
        self._knowledge_prompt = knowledge_prompt
        self._verbose = verbose
        self._save_session = save_session_cb or (lambda: None)
        self._heartbeat = heartbeat_cb or (lambda extra: None)

    # -- Decide ------------------------------------------------------------

    def should_fire(self, turn_result) -> bool:
        """Cheap threshold check. Fires every Nth consecutive failure.

        This method is the single source of truth for diagnose trigger
        gating. ``AgentLoop._decide_post_eval_action`` delegates to it
        directly instead of re-encoding the threshold rule inline.

        Returns True only when:
          - ``diagnose_suggest_threshold > 0``
          - ``counters.consecutive_failures > 0``
          - ``counters.consecutive_failures`` is an exact multiple of
            the threshold
        """
        diag_t = self._config.agent.diagnose_suggest_threshold
        if diag_t <= 0:
            return False
        cf = self._counters.consecutive_failures
        return cf > 0 and cf % diag_t == 0

    def rebind_counters(self, counters) -> None:
        """Swap the RunCounters reference held by the handler.

        Called by AgentLoop after ``--resume`` rebinds
        ``self._counters`` wholesale via ``RunCounters.from_dict``.
        """
        self._counters = counters

    # -- Apply -------------------------------------------------------------

    async def apply(self, turn_result) -> None:
        """Run the diagnose subagent and force a replan.

        Never raises — exceptions are caught and logged. A failed
        subagent degrades to "no direction change this turn"; the
        main loop continues unchanged.
        """
        try:
            diagnosis = await self._force_diagnose(turn_result)
        except Exception as e:
            if self._verbose:
                logger.warning(f"[DiagnoseHandler] failed: {e}")
            return
        if not diagnosis:
            return

        # Two shapes of "failing item":
        #   - eval_* outcomes: settle_active already closed it + _advance
        #     may have promoted a successor. require_replan retagging
        #     path rewinds the successor and annotates history. Skill
        #     reads for this item were already elided by
        #     ``TurnExecutor.execute`` right after settle_active.
        #   - edit_fail / quick_check_fail: settle_active NOT called;
        #     the failing item is still the active one with no history
        #     entry. require_replan pre-eval path closes it now — but
        #     the turn handler never called unload_item_reads for this
        #     item, so we must do it here. Otherwise the item's
        #     auto-injected SKILL.md and read_file('skills/...')
        #     tool_results leak into the replacement prompt.
        # ``failing_item_already_settled`` selects between the two.
        already_settled = bool(
            getattr(turn_result, "outcome", "").startswith("eval_")
        )

        # Capture the pre-eval item id BEFORE require_replan moves it
        # into history; the post-eval path leaves it None since
        # TurnExecutor already handled the elide.
        pre_eval_item_id: str = ""
        if not already_settled:
            _active = self._feedback.get_active_item() or {}
            pre_eval_item_id = _active.get("id") or ""

        self._feedback.require_replan(
            diagnosis=diagnosis,
            failing_item_already_settled=already_settled,
        )

        if pre_eval_item_id:
            self._buffer.unload_item_reads(pre_eval_item_id)

        self._buffer.append({
            "role": "user",
            "content": (
                f"[System] ⚠ DIRECTION CHANGE — "
                f"{self._counters.consecutive_failures} consecutive failures.\n"
                f"The last failed item has been tagged as "
                f"abandoned (diagnose). Submit replacement item(s) via "
                f"`update_plan(items=[...])` based on the diagnostic "
                f"report below. Remaining queued items will execute after."
                f"\n\n{diagnosis}"
            ),
        })
        self._counters.reset_after_diagnose()
        self._save_session()

    async def _force_diagnose(self, turn_result) -> str:
        """Run the diagnostic subagent with the current error context
        and return the formatted report. Failures inside
        ``run_diagnostic_subagent`` collapse to a string error
        message rather than raising — the caller in ``apply`` still
        injects this as the diagnosis so the run history records
        that the direction change was attempted.
        """
        error_ctx = turn_result.feedback or "Multiple consecutive failures."
        if self._verbose:
            logger.info(
                f"[DiagnoseHandler] Auto-diagnose triggered "
                f"({self._counters.consecutive_failures} consecutive failures) …",
            )
        self._heartbeat("running diagnose subagent")

        # Build failure history + current plan for the subagent
        failure_history = ""
        current_plan = ""
        try:
            history = self._feedback.format_history(max_versions=2)
            if history:
                failure_history = history
            plan_status = self._feedback.format_status()
            if plan_status:
                current_plan = plan_status
        except Exception as e:
            logger.debug(f"[DiagnoseHandler] failed to build history/plan context: {e}")

        try:
            sa = self._config.agent
            diagnosis = await run_diagnostic_subagent(
                llm=self._llm,
                error_context=error_ctx,
                current_code=self._runner.get_editable_contents(),
                task_description=self._config.description,
                task_dir=self._task_dir,
                knowledge_prompt=self._knowledge_prompt,
                max_iterations=sa.subagent_max_iterations,
                code_truncate=sa.subagent_code_truncate,
                result_truncate=sa.subagent_result_truncate,
                failure_history=failure_history,
                current_plan=current_plan,
            )
        except Exception as e:
            diagnosis = f"Diagnostic subagent failed: {e}"
        if self._verbose:
            logger.info(f"[DiagnoseHandler] Diagnosis:\n{diagnosis}")
        return f"[Diagnostic Subagent Report]\n{diagnosis}"
