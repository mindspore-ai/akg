"""
TurnExecutor — Per-turn tool dispatch, rollback, and eval pipeline.

Owns the single-turn state machine:
  tool dispatch → edit tracking → rollback on failure
  → quick_check → eval → feedback injection

Extracted from AgentLoop to isolate the turn-level logic from
the outer loop lifecycle (resume, context management, finalization).
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from .tools import (
    ToolResult, build_tool_handlers,
    execute_quick_check, execute_run_eval,
)
from .feedback import FeedbackBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Turn result data structure
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """Full outcome of a single turn — all state changes bundled here."""
    outcome: str                         # "finish", "edit_fail", "read_only",
                                         # "quick_check_fail", "eval_*"
    # Counters (may be updated by the turn)
    eval_calls_made: int = 0
    consecutive_failures: int = 0
    consecutive_no_edit_turns: int = 0

    # Messages list (possibly compacted by the compact tool)
    messages: list = field(default_factory=list)

    # Diagnostics
    tool_calls: list = field(default_factory=list)
    results_log: list = field(default_factory=list)
    feedback: str = ""
    eval_record: Optional[dict] = None
    has_finish: bool = False


# ---------------------------------------------------------------------------
# TurnExecutor
# ---------------------------------------------------------------------------

class TurnExecutor:
    """Executes a single agent turn: dispatch tools, rollback, eval.

    Stateless across turns — all mutable state is passed in or returned
    via TurnResult.
    """

    def __init__(
        self,
        task_dir: str,
        config,
        runner,
        feedback: FeedbackBuilder,
        session,  # SessionStore — for snapshots, logging, archival
        llm,      # ConversationAdapter — for compact/diagnose
        *,
        model: str,
        provider: str,
        device_id: int = 0,
        verbose: bool = True,
    ):
        self.task_dir = task_dir
        self.config = config
        self.runner = runner
        self.feedback = feedback
        self.session = session
        self.llm = llm
        self.model = model
        self.provider = provider
        self.device_id = device_id
        self.verbose = verbose

        self._tool_handlers = build_tool_handlers(task_dir, config)

    # -- Public API ----------------------------------------------------------

    async def execute(
        self,
        tool_calls: list,
        messages: list,
        eval_calls_made: int,
        max_rounds: int,
        consecutive_failures: int,
        consecutive_no_edit_turns: int,
        total_api_calls: int,
        baseline_commit: str | None = None,
    ) -> TurnResult:
        """Execute one turn's tool calls and the post-edit pipeline."""
        snapshots = self.session.snapshot_editable_files()

        results, edits_made, results_log, has_finish, messages = \
            await self._dispatch_tools(tool_calls, messages,
                                       eval_calls_made=eval_calls_made,
                                       max_rounds=max_rounds)

        # Append tool results to conversation
        messages.append({"role": "user", "content": results})

        def _result(outcome, eval_calls, failures, no_edit_turns, **kw):
            """All counters are required — no defaults that can go stale."""
            return TurnResult(
                outcome=outcome,
                eval_calls_made=eval_calls,
                consecutive_failures=failures,
                consecutive_no_edit_turns=no_edit_turns,
                messages=messages,
                tool_calls=tool_calls,
                results_log=results_log,
                **kw,
            )

        # --- finish ---
        if has_finish:
            successful = [e for e in edits_made if e["result"].ok]
            if successful:
                self.session.restore_snapshots(snapshots)
            self.session.log_turn(total_api_calls, tool_calls, results_log,
                                  "finish", eval_calls_made)
            return _result("finish", eval_calls_made, consecutive_failures,
                           consecutive_no_edit_turns, has_finish=True)

        # --- edit failures (atomic rollback) ---
        failed_edits = [e for e in edits_made if not e["result"].ok]
        successful_edits = [e for e in edits_made if e["result"].ok]

        if failed_edits:
            if successful_edits:
                self.session.restore_snapshots(snapshots)
            consecutive_failures += 1
            feedback = self.feedback.build_failure_feedback(
                failed_edits, len(edits_made))
            messages.append({"role": "user", "content": feedback})
            self.session.log_turn(total_api_calls, tool_calls, results_log,
                                  "edit_fail", eval_calls_made)
            return _result("edit_fail", eval_calls_made, consecutive_failures,
                           consecutive_no_edit_turns, feedback=feedback)

        # --- no edits this turn ---
        if not successful_edits:
            consecutive_no_edit_turns += 1
            self._nudge_if_no_edits(messages, tool_calls, consecutive_no_edit_turns)
            self.session.log_turn(total_api_calls, tool_calls, results_log,
                                  "read_only", eval_calls_made)
            return _result("read_only", eval_calls_made, consecutive_failures,
                           consecutive_no_edit_turns)

        # --- successful edits → quick_check → eval ---
        consecutive_no_edit_turns = 0
        edit_desc = " + ".join(e["description"] for e in successful_edits if e["description"])
        if not edit_desc:
            edit_desc = f"edit {successful_edits[0]['path']}"

        # Quick check
        if self.verbose:
            print(f"[Turn] quick_check after {len(successful_edits)} edit(s) …")
        qc_result = execute_quick_check(self.task_dir, self.config, device_id=self.device_id)

        if not qc_result.ok:
            if self.verbose:
                print(f"[Turn] quick_check FAILED — rolling back")
                print(f"  {qc_result.message[:200]}")
            self.session.restore_snapshots(snapshots)
            consecutive_failures += 1
            qc_feedback = self.feedback.build_quick_check_feedback(
                qc_result.message)
            messages.append({"role": "user", "content": qc_feedback})
            self.session.log_turn(total_api_calls, tool_calls, results_log,
                                  "quick_check_fail", eval_calls_made,
                                  {"error": qc_result.message[:1000]})
            return _result("quick_check_fail", eval_calls_made,
                           consecutive_failures, consecutive_no_edit_turns,
                           feedback=qc_feedback)

        # Full eval
        if self.verbose:
            print(f"[Turn] quick_check OK — running eval …")
        eval_json = await execute_run_eval(edit_desc, self.runner,
                                           raw_output_tail=self.config.agent.raw_output_tail)
        eval_calls_made += 1

        if self.verbose:
            print(f"[Turn] eval: {eval_json[:200]}")

        try:
            eval_record = json.loads(eval_json)
        except json.JSONDecodeError:
            eval_record = {"status": "FAIL", "fail_reason": eval_json,
                           "metrics": {}}

        # -- Auto-settle the active plan item (3-way: KEEP / FAIL / DISCARD) --
        eval_status = eval_record.get("status", "FAIL")
        eval_metrics = eval_record.get("metrics", {})
        if eval_status == "KEEP":
            self.feedback.settle_active(
                True, "keep", eval_metrics, edit_desc=edit_desc)
            consecutive_failures = 0
        elif eval_status == "FAIL":
            reason = eval_record.get("fail_reason", "unknown")
            self.feedback.settle_active(
                False, reason, eval_metrics, edit_desc=edit_desc)
            consecutive_failures += 1
        else:  # DISCARD — correct but no improvement
            self.feedback.settle_active(
                False, "no improvement", eval_metrics, edit_desc=edit_desc)
            consecutive_failures = 0

        feedback_text = self.feedback.build_eval_feedback(
            eval_record=eval_record,
            eval_calls_made=eval_calls_made,
            max_rounds=max_rounds,
            best_result=self.runner.best_result,
        )
        # Append performance ranking so the agent can see top strategies
        ranking = self.runner.logger.get_performance_ranking()
        if ranking:
            feedback_text += f"\n\n{ranking}"

        # Append cumulative diff after KEEP (diff changed, worth showing)
        if eval_status == "KEEP" and baseline_commit:
            from ..framework.runner import git_diff
            diff = git_diff(self.task_dir, baseline_commit,
                            paths=list(self.config.editable_files))
            if diff:
                limit = self.config.agent.cumulative_diff_truncate
                if len(diff) > limit:
                    diff = diff[:limit] + f"\n... [truncated at {limit} chars]"
                feedback_text += (
                    f"\n\n## Cumulative Changes (all kept edits vs baseline)"
                    f"\n```diff\n{diff}\n```"
                )

        messages.append({"role": "user", "content": feedback_text})

        self.session.log_turn(total_api_calls, tool_calls, results_log,
                              f"eval_{eval_status.lower()}", eval_calls_made,
                              {"metrics": eval_metrics, "description": edit_desc})

        return _result(f"eval_{eval_status.lower()}", eval_calls_made,
                       consecutive_failures, consecutive_no_edit_turns,
                       feedback=feedback_text, eval_record=eval_record)

    # -- Tool dispatch ------------------------------------------------------

    async def _dispatch_tools(self, tool_calls: list, messages: list,
                              *, eval_calls_made: int = 0,
                              max_rounds: int = 0,
                              ) -> tuple[list, list, list, bool, list]:
        """Dispatch tool calls. Returns (results, edits, log, has_finish, messages).

        messages is returned because compact may replace it with a new list.
        """
        results = []
        edits_made = []
        has_finish = False
        results_log = []

        for tc in tool_calls:
            tool_name = tc["tool_name"]
            args = tc["arguments"]
            tool_id = tc["tool_use_id"]

            if self.verbose:
                args_preview = str(args)[:120]
                print(f"  > {tool_name}: {args_preview}")

            # -- Phase-based permission checks --
            if tool_name in ("patch_file", "write_file"):
                ok, err = self.feedback.validate_edit(args.get("plan_item_id", ""))
                if not ok:
                    msg = f"{err}\n\n{self.feedback.format_status()}"
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": msg,
                    })
                    results_log.append(f"BLOCKED {tool_name}: {err[:200]}")
                    continue

            if tool_name == "update_plan":
                if self.feedback.phase == "active":
                    active_id = self.feedback._active_item_id
                    msg = (
                        f"BLOCKED: Cannot rewrite plan while item {active_id} is active. "
                        f"You MUST make a code edit now using patch_file or write_file "
                        f"with plan_item_id='{active_id}'."
                        f"\n\n{self.feedback.format_status()}"
                    )
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": msg,
                    })
                    results_log.append("BLOCKED update_plan: active phase")
                    continue

            if tool_name == "finish":
                # Block finish before reaching half of max_rounds
                min_rounds = max_rounds // 2
                if eval_calls_made < min_rounds:
                    remaining = min_rounds - eval_calls_made
                    msg = (
                        f"BLOCKED: must complete at least {min_rounds} eval rounds "
                        f"(half of max_rounds={max_rounds}) before finishing. "
                        f"Currently at {eval_calls_made}, need {remaining} more. "
                        f"Keep optimizing."
                    )
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": msg,
                    })
                    results_log.append(f"BLOCKED finish: {eval_calls_made}/{min_rounds} min rounds")
                    continue
                if self.feedback.phase != "replanning":
                    msg = (
                        f"BLOCKED: finish requires all plan items settled "
                        f"(current phase={self.feedback.phase})."
                        f"\n\n{self.feedback.format_status()}"
                    )
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": msg,
                    })
                    results_log.append("BLOCKED finish: not replanning")
                    continue
                if edits_made:
                    msg = (
                        "BLOCKED: finish cannot be combined with edits in the same turn."
                        f"\n\n{self.feedback.format_status()}"
                    )
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": msg,
                    })
                    results_log.append("BLOCKED finish: edits in same turn")
                    break

            # State-mutation tools handled directly
            if tool_name == "finish":
                has_finish = True
                summary = args.get("summary", "")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": f"Optimization finished. {summary}",
                })
                results_log.append(f"finish: {summary[:200]}")
                break

            if tool_name == "update_plan":
                reply = self._handle_update_plan(args)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": reply,
                })
                results_log.append(f"update_plan: {reply[:200]}")
                continue

            if tool_name == "compact":
                reply, messages = await self._handle_compact(messages)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": reply,
                })
                results_log.append("compact")
                continue

            # Dispatch to handler
            handler = self._tool_handlers.get(tool_name)
            if handler is None:
                output = ToolResult(ok=False, message=f"ERROR: Unknown tool '{tool_name}'")
            else:
                try:
                    output = handler(**args)
                except Exception as e:
                    output = ToolResult(ok=False, message=f"ERROR: {tool_name} raised {type(e).__name__}: {e}")

            results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": output.message,
            })
            results_log.append(output.message[:200])

            if tool_name in ("patch_file", "write_file"):
                edits_made.append({
                    "path": args.get("path", "?"),
                    "description": args.get("description", ""),
                    "result": output,
                    "diff": ({"old_str": args.get("old_str", ""),
                              "new_str": args.get("new_str", "")}
                             if tool_name == "patch_file" else "full rewrite"),
                })

        return results, edits_made, results_log, has_finish, messages

    # -- Special tool handlers ----------------------------------------------

    def _handle_update_plan(self, args: dict) -> str:
        plan_text = args.get("plan", "")
        if len(plan_text) > self.config.agent.plan_max_chars:
            plan_text = plan_text[:self.config.agent.plan_max_chars]
        ok, msg = self.feedback.submit_plan(plan_text)
        if not ok:
            return f"Plan rejected: {msg}"
        return msg

    async def _handle_compact(self, messages: list) -> tuple[str, list]:
        """Compress context via auto_compact. Returns (reply, new_messages)."""
        from .compress import auto_compact
        a = self.config.agent
        if len(messages) >= a.compact_min_messages:
            if self.verbose:
                print("  [compact] Compressing …", flush=True)
            new_messages = await auto_compact(
                messages, self.llm.client, self.model,
                self.provider, self.task_dir,
                text_limit=a.auto_compact_text_limit,
                summary_max_tokens=a.compact_summary_max_tokens,
                session_dir=a.session_dir,
            )
            return "Context compressed.", new_messages
        return "Nothing to compress.", messages

    # -- Nudge helpers ------------------------------------------------------

    def _nudge_if_no_edits(self, messages: list, tool_calls: list,
                           consecutive_no_edit_turns: int):
        """Inject nudge messages when the agent isn't making edits."""
        nudge = self.feedback.build_phase_nudge()
        if nudge:
            messages.append({"role": "user", "content": nudge})

        if consecutive_no_edit_turns >= self.config.agent.max_no_edit_turns:
            messages.append({
                "role": "user",
                "content": (
                    f"[System] WARNING: {consecutive_no_edit_turns} consecutive turns "
                    f"without edits. You MUST make a code change (patch_file or write_file) "
                    f"in your next turn."
                ),
            })
