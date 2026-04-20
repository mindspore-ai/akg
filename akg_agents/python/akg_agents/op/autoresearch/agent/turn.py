"""
TurnExecutor - Per-turn tool dispatch, rollback, and eval pipeline.

Owns the single-turn state machine:
  tool dispatch -> edit tracking -> rollback on failure
  -> quick_check -> eval -> feedback injection

Extracted from AgentLoop to isolate the turn-level logic from
the outer loop lifecycle (resume, context management, finalization).
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .tools import (
    ToolResult, build_tool_handlers,
    execute_quick_check, execute_run_eval,
)
from .feedback import FeedbackBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FileTouchTracker — per-file freshness from the agent's POV
# ---------------------------------------------------------------------------


def _format_tool_result_content(tool_name: str, args: dict,
                                output) -> str:
    """Render a tool_result payload for the conversation buffer.

    Edit results get wrapped in a compact XML envelope so the model
    can parse status / path / mode / retry count without regexing
    free-form English. Non-edit tools keep the old plain-string shape
    so nothing else in the pipeline has to change.

    XML is deliberately tiny — no external namespaces, just enough
    structure to make ``<status>OK</status>`` vs ``<status>ERROR</status>``
    trivial to skim in a long conversation. Parses round-trip with
    Python's stdlib ``xml.etree`` if a downstream consumer ever needs
    to.
    """
    msg = output.message if hasattr(output, "message") else str(output)
    if tool_name != "edit":
        return msg
    status = "OK" if getattr(output, "ok", False) else "ERROR"
    path = args.get("path", "?")
    edit_specs = args.get("edits")
    if edit_specs:
        mode = "batch({})".format(
            ",".join((e.get("mode") or "exact") for e in edit_specs)
        )
    else:
        mode = args.get("mode") or "exact"
    # Parse retries=N tag emitted by execute_edit so the dispatcher's
    # internal recovery count surfaces into the visible envelope, then
    # remove just that tag from the body — the old "slice everything
    # before the tag" approach swallowed the whole message when the
    # tag happened to be at the front (retry-exhausted failures).
    retries = 0
    _retry_re = re.compile(r"\s*\[retries=(\d+)(?:,[^\]]*)?\]\s*")
    m = _retry_re.search(msg)
    if m:
        try:
            retries = int(m.group(1))
        except ValueError:
            pass
        summary = (msg[:m.start()] + " " + msg[m.end():]).strip()
    else:
        summary = msg
    header = (
        f'<tool_result kind="edit" status="{status}" mode="{mode}" '
        f'path="{path}" retries="{retries}">'
    )
    return f"{header}\n{summary}\n</tool_result>"


class FileTouchTracker:
    """Tracks per-file freshness from the agent's POV.

    A file is 'stale' when the agent's last ``edit`` (any batch that
    included a string-based sub-edit) failed with "old_str not found"
    — meaning the agent's mental model of the file content is wrong.
    The next edit that references old_str on that file is blocked
    until a successful ``read_file`` (or any successful write) refreshes
    the agent's view.

    Single owner of the per-file freshness rules: any new policy
    (external modification detection, time-based invalidation, etc.)
    plugs into this class without touching the dispatch loop.
    """

    def __init__(self) -> None:
        self._stale: set[str] = set()

    def mark_read(self, abs_path: str) -> None:
        """A successful read refreshes the agent's view."""
        self._stale.discard(abs_path)

    def mark_write(self, abs_path: str) -> None:
        """A successful edit refreshes the agent's view of the file."""
        self._stale.discard(abs_path)

    def mark_patch_stale(self, abs_path: str) -> None:
        """Record a failed edit whose old_str could not be located."""
        self._stale.add(abs_path)

    def is_stale(self, abs_path: str) -> bool:
        return abs_path in self._stale


# ---------------------------------------------------------------------------
# Turn result data structure
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    """Full outcome of a single turn — all state changes bundled here.

    Note: counter fields and the message list are NOT carried here.
    TurnExecutor mutates ``buffer`` via ConversationBuffer methods and
    mutates ``counters`` via RunCounters' record_* methods, so the
    loop's owned objects already reflect every change by the time
    ``execute()`` returns. TurnResult only carries per-turn diagnostics
    and the outcome dispatch tag.
    """
    outcome: str                         # "finish", "edit_fail", "read_only",
    # "quick_check_fail", "eval_*"
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
        llm,      # AkgLLMAdapter — for compact/diagnose
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
        self.knowledge_prompt = None  # set by AgentLoop after system prompt is built
        # last_diagnosis text now lives on FeedbackBuilder paired with
        # must_replan — read via self.feedback.last_diagnosis. Removing
        # the local mirror eliminates the per-iteration sync that used
        # to copy AgentLoop._last_diagnosis here.
        self.compacted_this_turn = False  # set by _handle_compact when messages replaced

        # Per-file freshness tracker (lifetime = whole run); see
        # FileTouchTracker docstring for the staleness rules.
        self._files = FileTouchTracker()

        self._tool_handlers = build_tool_handlers(task_dir, config)

    # -- Public API ----------------------------------------------------------

    async def execute(
        self,
        tool_calls: list,
        buffer,
        counters,
        max_rounds: int,
        baseline_commit: str | None = None,
    ) -> TurnResult:
        """Execute one turn's tool calls and the post-edit pipeline.

        ``buffer`` is the AgentLoop's ConversationBuffer; ``counters``
        is its RunCounters. Both are mutated in place — ``buffer`` via
        ``buffer.append`` / ``buffer.auto_compact`` and ``counters``
        via ``counters.record_*`` — so the caller never has to copy
        state back from TurnResult.
        """
        self.compacted_this_turn = False
        snapshots = self.runner.file_state.snapshot()

        results, edits_made, results_log, has_finish = \
            await self._dispatch_tools(tool_calls, buffer,
                                       counters=counters,
                                       eval_calls_made=counters.eval_calls_made,
                                       max_rounds=max_rounds)

        # Append tool results to conversation
        buffer.append({"role": "user", "content": results})

        def _result(outcome, **kw):
            return TurnResult(
                outcome=outcome,
                tool_calls=tool_calls,
                results_log=results_log,
                **kw,
            )

        # --- finish ---
        if has_finish:
            successful = [e for e in edits_made if e["result"].ok]
            if successful:
                self.runner.file_state.restore(snapshots)
            self.session.log_turn(counters.total_api_calls, tool_calls,
                                  results_log, "finish", counters.eval_calls_made)
            return _result("finish", has_finish=True)

        # --- edit failures (atomic rollback) ---
        failed_edits = [e for e in edits_made if not e["result"].ok]
        successful_edits = [e for e in edits_made if e["result"].ok]

        if failed_edits:
            if successful_edits:
                self.runner.file_state.restore(snapshots)
            counters.record_edit_attempt(ok=False)
            feedback = self.feedback.build_failure_feedback(
                failed_edits, len(edits_made))
            buffer.append({"role": "user", "content": feedback})
            self.session.log_turn(counters.total_api_calls, tool_calls,
                                  results_log, "edit_fail",
                                  counters.eval_calls_made)
            return _result("edit_fail", feedback=feedback)

        # --- no edits this turn ---
        if not successful_edits:
            counters.record_turn_with_no_edits()
            self._nudge_if_no_edits(buffer, tool_calls,
                                    counters.consecutive_no_edit_turns)
            self.session.log_turn(counters.total_api_calls, tool_calls,
                                  results_log, "read_only",
                                  counters.eval_calls_made)
            return _result("read_only")

        # --- successful edits — quick_check — eval ---
        counters.record_turn_with_edits()
        edit_desc = " + ".join(e["description"] for e in successful_edits if e["description"])
        if not edit_desc:
            edit_desc = f"edit {successful_edits[0]['path']}"

        # Quick check
        if self.verbose:
            logger.info(f"[Turn] quick_check after {len(successful_edits)} edit(s) ...")
        qc_result = execute_quick_check(self.task_dir, self.config, device_id=self.device_id)

        if not qc_result.ok:
            if self.verbose:
                logger.warning(f"[Turn] quick_check FAILED - rolling back")
                logger.info(f"  {qc_result.message[:200]}")
            self.runner.file_state.restore(snapshots)
            counters.record_edit_attempt(ok=False)
            qc_feedback = self.feedback.build_quick_check_feedback(
                qc_result.message)
            buffer.append({"role": "user", "content": qc_feedback})
            self.session.log_turn(counters.total_api_calls, tool_calls,
                                  results_log, "quick_check_fail",
                                  counters.eval_calls_made,
                                  {"error": qc_result.message[:1000]})
            return _result("quick_check_fail", feedback=qc_feedback)

        # Edits proceed straight to eval.

        # Full eval
        if self.verbose:
            logger.info(f"[Turn] quick_check OK - running eval ...")
        eval_json = await execute_run_eval(edit_desc, self.runner,
                                           raw_output_tail=self.config.agent.raw_output_tail)

        if self.verbose:
            logger.info(f"[Turn] eval: {eval_json[:200]}")

        try:
            eval_record = json.loads(eval_json)
        except json.JSONDecodeError:
            eval_record = {"status": "FAIL", "fail_reason": eval_json,
                           "metrics": {}}

        # -- Auto-settle the active plan item (3-way: KEEP / FAIL / DISCARD) --
        # All counter side-effects (eval_calls_made +=, consecutive_failures
        # rules, no_improvement rules) are encapsulated in
        # ``counters.record_eval`` so the per-outcome rules live in one place
        # — see RunCounters.record_eval docstring.
        eval_status = eval_record.get("status", "FAIL")
        eval_metrics = eval_record.get("metrics", {})
        # Capture the settling item's id before settle_active advances
        # the plan; we use it to evict any skill reads the agent
        # accumulated for this item.
        settled_item = self.feedback.get_active_item()
        settled_id = (settled_item or {}).get("id") or ""
        if eval_status == "KEEP":
            self.feedback.settle_active(
                True, "keep", eval_metrics, edit_desc=edit_desc)
            counters.record_eval("eval_keep")
        elif eval_status == "FAIL":
            reason = eval_record.get("fail_reason", "unknown")
            self.feedback.settle_active(
                False, reason, eval_metrics, edit_desc=edit_desc)
            counters.record_eval("eval_fail")
        else:  # DISCARD — correct but no improvement
            self.feedback.settle_active(
                False, "no improvement", eval_metrics, edit_desc=edit_desc)
            counters.record_eval("eval_discard")
        if settled_id:
            buffer.unload_item_reads(settled_id)

        feedback_text = self.feedback.build_eval_feedback(
            eval_record=eval_record,
            eval_calls_made=counters.eval_calls_made,
            max_rounds=max_rounds,
            best_result=self.runner.best_result,
        )
        # Append cumulative diff after KEEP (diff changed, worth showing)
        if eval_status == "KEEP" and baseline_commit:
            diff = self.runner.git.diff(
                baseline_commit,
                paths=list(self.config.editable_files),
            )
            if diff:
                limit = self.config.agent.cumulative_diff_truncate
                if len(diff) > limit:
                    diff = diff[:limit] + f"\n... [truncated at {limit} chars]"
                feedback_text += (
                    f"\n\n## Cumulative Changes (all kept edits vs baseline)"
                    f"\n```diff\n{diff}\n```"
                )

        buffer.append({"role": "user", "content": feedback_text})

        self.session.log_turn(counters.total_api_calls, tool_calls,
                              results_log, f"eval_{eval_status.lower()}",
                              counters.eval_calls_made,
                              {"metrics": eval_metrics, "description": edit_desc})

        return _result(
            f"eval_{eval_status.lower()}",
            feedback=feedback_text,
            eval_record=eval_record,
        )

    # -- Tool dispatch ------------------------------------------------------

    async def _dispatch_tools(self, tool_calls: list, buffer,
                              *, counters,
                              eval_calls_made: int = 0,
                              max_rounds: int = 0,
                              ) -> tuple[list, list, list, bool]:
        """Dispatch tool calls. Returns (results, edits, log, has_finish).

        ``buffer`` is the AgentLoop's ConversationBuffer; the compact
        tool's path mutates it via ``buffer.auto_compact``, so the
        buffer doesn't need to be returned.
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
                logger.info(f"  > {tool_name}: {args_preview}")

            # -- Phase-based permission checks --
            # The unified ``edit`` tool is the ONLY file-mutation surface
            # exposed to the LLM. The older patch_file / write_file tools
            # were removed in the multi-edit refactor — any such call
            # arrives from a stale cache and should be rejected.
            if tool_name == "edit":
                ok, err = self.feedback.validate_edit(args.get("plan_item_id", ""))
                if not ok:
                    msg = err
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": msg,
                    })
                    results_log.append(f"BLOCKED {tool_name}: {err[:200]}")
                    continue

                # Skill acknowledgement gate: if the active item has a
                # backing_skill, the agent must have called
                # acknowledge_skill for it (either applying or declining)
                # before any edit is allowed.
                _active = self.feedback.get_active_item()
                if _active and _active.get("backing_skill") \
                        and not _active.get("skill_ack"):
                    _bs = _active.get("backing_skill")
                    _aid = _active.get("id")
                    msg = (
                        f"BLOCKED {tool_name}: item {_aid!r} has "
                        f"backing_skill={_bs!r}. Call acknowledge_skill "
                        f"first (read the injected SKILL.md, then submit "
                        f"valuable_aspects + kernel_application + "
                        f"applicability={{'apply'|'unbind'}}). Edits are "
                        f"unblocked once the ack lands."
                    )
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": msg,
                    })
                    results_log.append(
                        f"BLOCKED {tool_name}: missing acknowledge_skill "
                        f"for {_aid}"
                    )
                    continue

            # -- Stale-file guard for string-based edit modes --
            # If a previous ``edit`` failed with old_str-not-found on
            # this path, force a read_file before retrying. When the
            # batch contains any exact/block edit (either via ``edits``
            # list or top-level shorthand), the guard applies. A pure
            # rewrite/unified batch bypasses the guard: it doesn't rely
            # on old_str, so a stale mental model doesn't corrupt it.
            if tool_name == "edit":
                edit_specs = args.get("edits") or [{
                    "mode": args.get("mode") or "exact",
                }]
                has_string_mode = any(
                    (e.get("mode") or "exact") in ("exact", "block")
                    for e in edit_specs
                )
                if has_string_mode:
                    abs_path = self._abs_path(args.get("path", ""))
                    if abs_path and self._files.is_stale(abs_path):
                        msg = (
                            f"BLOCKED: {args.get('path', '?')} is stale "
                            f"(previous edit failed with 'old_str not "
                            f"found'). Call read_file on this path first "
                            f"to refresh your view, then retry the edit."
                        )
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": msg,
                        })
                        results_log.append(
                            f"BLOCKED edit: stale {args.get('path', '?')}"
                        )
                        continue

            if tool_name == "update_plan":
                if self.feedback.phase == "active":
                    active_id = self.feedback._active_item_id
                    msg = (
                        f"BLOCKED: Cannot rewrite plan while item {active_id} is active. "
                        f"You MUST make a code edit now using `edit` "
                        f"with plan_item_id='{active_id}'."
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
                    msg = (
                        f"BLOCKED: eval budget not yet exhausted "
                        f"({eval_calls_made}/{max_rounds} rounds used). "
                        f"Submit a fresh `update_plan` with new "
                        f"optimization directions you haven't tried yet."
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
                reply = await self._handle_update_plan(args, counters=counters)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": reply,
                })
                results_log.append(f"update_plan: {reply[:200]}")
                continue

            if tool_name == "search_skills":
                try:
                    reply = await self._handle_search_skills(args)
                except Exception as e:
                    reply = f"search_skills failed: {e}"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": reply,
                })
                results_log.append(f"search_skills: {reply[:200]}")
                continue

            if tool_name == "acknowledge_skill":
                reply = self._handle_acknowledge_skill(args)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": reply,
                })
                results_log.append(f"acknowledge_skill: {reply[:200]}")
                continue

            if tool_name == "compact":
                try:
                    reply = await self._handle_compact(buffer)
                except Exception as e:
                    reply = f"Compact failed: {e}"
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
                "content": _format_tool_result_content(
                    tool_name, args, output),
            })
            results_log.append(output.message[:200])

            # -- File touch tracking (post-handler) --
            # edit: failure with "old_str not found" → mark stale so a
            # read_file is required before the next string-based edit.
            # Any successful edit (regardless of mode) refreshes the
            # agent's view of the file.
            abs_path = self._abs_path(args.get("path", ""))
            if abs_path:
                if tool_name == "edit":
                    if output.ok:
                        self._files.mark_write(abs_path)
                    elif "old_str not found" in (output.message or ""):
                        self._files.mark_patch_stale(abs_path)
                elif tool_name == "read_file" and output.ok:
                    self._files.mark_read(abs_path)
                    # Tag reads of task_dir/skills/... so buffer can
                    # elide their tool_result bodies when the owning
                    # plan item settles.
                    path_arg = (args.get("path") or "").lstrip("./").lstrip(
                        ".\\",
                    )
                    if path_arg.startswith("skills/") or path_arg.startswith(
                        "skills\\",
                    ):
                        active = self.feedback.get_active_item()
                        active_id = (active or {}).get("id") or ""
                        if active_id:
                            buffer.track_item_skill_read(tool_id, active_id)

            if tool_name == "edit":
                # Record a compact diff descriptor for downstream
                # rollback / logging. For multi-edit batches we list
                # per-edit modes; for single-edit we record old/new
                # for string-based modes and a tag for rewrite/unified.
                edit_specs = args.get("edits")
                if edit_specs:
                    diff_info = {"batch": [
                        (e.get("mode") or "exact") for e in edit_specs
                    ]}
                else:
                    em = args.get("mode") or "exact"
                    if em in ("exact", "block"):
                        diff_info = {"old_str": args.get("old_str", ""),
                                     "new_str": args.get("new_str", ""),
                                     "mode": em}
                    elif em == "rewrite":
                        diff_info = "full rewrite"
                    else:  # unified
                        diff_info = {"unified": args.get("diff", "")[:400]}
                edits_made.append({
                    "path": args.get("path", "?"),
                    "description": args.get("description", ""),
                    "result": output,
                    "diff": diff_info,
                })

        return results, edits_made, results_log, has_finish

    def _abs_path(self, path: str) -> str:
        """Resolve a tool-supplied path to a normalized absolute path.

        Returns ``""`` for empty input. Mirrors the resolution rule used
        by ``_validate_editable_path`` so the FileTouchTracker key matches
        the path the file-op handlers operate on.
        """
        if not path:
            return ""
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(self.task_dir, path))

    # -- Special tool handlers ----------------------------------------------

    async def _handle_update_plan(self, args: dict, *, counters) -> str:
        """Match keywords + submit a plan from agent-supplied items or markdown.

        Two modes:

        **Normal** (``must_replan`` is False): full plan submission.
        Items with ``keywords`` are matched against the SkillPool
        via ``_match_keywords_to_skills``; unmatched items stay
        unbound (free exploration). ``submit_plan(items=...)``
        replaces the entire plan.

        **Replace** (``must_replan`` is True, set by diagnose): the
        agent submits one or more replacement items for the abandoned
        active item. ``replace_active_item`` inserts them at the
        abandoned item's position and preserves all pending items in
        the queue.

        In both modes, agent-supplied ``backing_skill`` is stripped
        at the trust boundary (binding is system-only via keyword
        matching).
        """
        raw_items = args.get("items")
        plan_text = args.get("plan", "")

        if raw_items is None:
            if plan_text:
                return (
                    "Plan rejected: legacy markdown `plan` no longer accepted "
                    "(rationale is required per item). Use `items=[...]`."
                )
            return "Plan rejected: must provide `items`."

        if not isinstance(raw_items, list):
            return "Plan rejected: `items` must be a list of objects."
        agent_items = self._parse_structured_items(raw_items)

        # Pre-validate rationale BEFORE keyword matching so a rejected
        # plan has no side effect on the SkillPool. The feedback
        # validator is the single source of truth (repeated below), but
        # running it here first keeps `_match_keywords_to_skills` — and
        # its auto-widen `pool.refill(mode="append")` — out of the
        # reject path.
        if not agent_items:
            return "Plan rejected: items must contain at least one non-empty text."
        for it in agent_items:
            _, err = self.feedback._validate_rationale(it.get("rationale"))
            if err:
                return f"Plan rejected: {err}"

        plan_items = await self._match_keywords_to_skills(agent_items)

        # Replace mode: single-item replacement after diagnose.
        if self.feedback.must_replan:
            if not plan_items:
                return (
                    "Plan rejected: must provide at least one replacement "
                    "item after a direction change."
                )
            ok, msg = self.feedback.replace_active_item(plan_items)
            if not ok:
                return f"Plan rejected: {msg}"
            return msg

        if not plan_items:
            return "Plan rejected: items must contain at least one non-empty text."

        ok, msg = self.feedback.submit_plan(items=plan_items)
        if not ok:
            return f"Plan rejected: {msg}"
        bound_count = sum(1 for item in plan_items if item.get("backing_skill"))
        if bound_count:
            msg = f"{msg} ({bound_count} keyword-matched skill binding(s))"

        # Append a statistical summary of the plan version that just
        # ended so the agent has a distilled view in the ack message
        # and doesn't silently repeat failed directions. ``submit_plan``
        # already bumped plan_version, so ``plan_version - 1`` is the
        # version whose items were just moved into settled_history.
        # The ``replace_active_item`` path above is deliberately NOT
        # summarized — it stays on the same plan_version, no transition.
        prev_summary = self.feedback.format_prev_plan_summary(
            self.feedback.plan_version - 1,
        )
        if prev_summary:
            msg = f"{msg}\n\n{prev_summary}"
        return msg

    def _sanitize_keywords(self, raw_keywords) -> list[str]:
        """Normalize agent-submitted plan-item keywords.

        Delegates to ``FeedbackBuilder._sanitize_keywords`` so the
        validation logic stays single-sourced.
        """
        return self.feedback._sanitize_keywords(raw_keywords)

    def _parse_structured_items(self, raw_items: list) -> list[dict]:
        """Strip agent-submitted items to trusted text/keyword/rationale fields.

        Drops any extra keys (including ``backing_skill``) at the trust
        boundary so binding stays a system-only privilege regardless
        of what the LLM emits or what stale tool schema a resumed
        session might carry. ``rationale`` is preserved untouched here;
        ``FeedbackBuilder._validate_rationale`` is the single source of
        truth for its content rules.
        """
        out: list[dict] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            built = {"text": text}
            keywords = self._sanitize_keywords(item.get("keywords"))
            if keywords:
                built["keywords"] = keywords
            rationale = item.get("rationale")
            if isinstance(rationale, str) and rationale.strip():
                built["rationale"] = rationale
            out.append(built)
        return out

    def _match_backing_skill(self, keywords: list[str],
                             pool, bound_names: set[str]) -> str:
        """Single-item keyword → backing_skill match. Empty string on miss."""
        try:
            ranked = pool.match_by_keywords(
                keywords,
                top_k=max(len(pool), 1),
                op_name=getattr(self.config, "name", "") or "",
                exclude_names=bound_names,
            )
        except Exception as exc:
            logger.warning(
                "[update_plan] keyword skill match failed for %s: %r",
                keywords, exc,
            )
            return ""
        if not ranked:
            return ""
        return getattr(ranked[0][1], "name", "") or ""

    async def _match_keywords_to_skills(self, items: list[dict]) -> list[dict]:
        """Bind keyword-requested items to residual skills when possible.

        Preserves the agent's `rationale` field untouched through to
        FeedbackBuilder, which is the single validator.

        Two-pass matching with on-demand pool widening: any
        keyword-requested item whose keywords don't hit the current
        pool triggers a one-shot append-refill over example/case/
        method/implementation, then those misses are retried. The
        trigger is "a specific keyword request couldn't be served",
        not "pool has zero bindable" — with guides still selected but
        semantically unrelated to the agent's request, the pool is
        non-empty yet every match comes up empty.
        """
        if not items:
            return []

        pool = getattr(self.feedback, "skill_pool", None)

        bound_names: set[str] = set()
        out: list[dict] = []
        # Pass 1: build items; remember which keyword-requested items
        # missed so we can retry after widening the pool.
        misses: list[tuple[int, list[str]]] = []
        for idx, item in enumerate(items):
            built = {"text": item.get("text", "")}
            rationale = item.get("rationale")
            if isinstance(rationale, str) and rationale.strip():
                built["rationale"] = rationale
            keywords = self._sanitize_keywords(item.get("keywords"))
            if keywords:
                built["keywords"] = keywords
            if keywords and pool is not None:
                skill_name = self._match_backing_skill(keywords, pool, bound_names)
                if skill_name:
                    built["backing_skill"] = skill_name
                    bound_names.add(skill_name)
                else:
                    misses.append((idx, keywords))
            out.append(built)

        # Pass 2: widen the pool and retry misses only when needed.
        if misses and pool is not None:
            try:
                added = await pool.refill(
                    llm=self.llm,
                    config=self.config,
                    mode="append",
                    plan_version=self.feedback.plan_version,
                    include_categories=[
                        "example", "case", "method", "implementation",
                    ],
                )
            except Exception as exc:
                logger.warning(
                    "[update_plan] auto-widen refill failed: %r", exc,
                )
                added = []
            if added:
                logger.info(
                    "[update_plan] auto-widen appended %d skills; "
                    "retrying %d keyword miss(es)",
                    len(added), len(misses),
                )
                for idx, keywords in misses:
                    skill_name = self._match_backing_skill(
                        keywords, pool, bound_names,
                    )
                    if skill_name:
                        out[idx]["backing_skill"] = skill_name
                        bound_names.add(skill_name)

        return out

    _ACK_FIELD_MIN = 100
    _ACK_FIELD_MAX = 500

    def _handle_acknowledge_skill(self, args: dict) -> str:
        """Record the agent's structured acknowledgement of the injected
        backing_skill for the currently-active plan item.

        Schema-strict: rejects malformed input rather than silently
        accepting a half-filled ack (otherwise the gate loses meaning).
        On ``applicability == 'unbind'`` the binding is released for
        THIS item (item.backing_skill cleared, item stays active for
        free exploration) and the skill is marked unbound-at-this-version
        in the registry. The skill itself stays available — a future
        item can bind it again, and a successful KEEP promotes it back
        to the top of the priority tier (see ``SkillBuilder.tier``).
        """
        item_id = (args.get("plan_item_id") or "").strip()
        valuable = (args.get("valuable_aspects") or "").strip()
        application = (args.get("kernel_application") or "").strip()
        applicability = (args.get("applicability") or "").strip().lower()

        active = self.feedback.get_active_item()
        if not active:
            return (
                "acknowledge_skill rejected: no active plan item. "
                "Submit `update_plan` first."
            )
        aid = active.get("id") or ""
        if item_id != aid:
            return (
                f"acknowledge_skill rejected: plan_item_id={item_id!r} is "
                f"not the active item ({aid!r})."
            )
        bs = active.get("backing_skill")
        if not bs:
            return (
                f"acknowledge_skill rejected: item {aid!r} has no "
                f"backing_skill; acknowledgement is only required for "
                f"skill-bound items."
            )
        if applicability not in ("apply", "unbind"):
            return (
                "acknowledge_skill rejected: applicability must be "
                "one of 'apply' / 'unbind'."
            )
        for field_name, field_val in (
            ("valuable_aspects", valuable),
            ("kernel_application", application),
        ):
            if not field_val:
                return (
                    f"acknowledge_skill rejected: {field_name} is required."
                )
            if len(field_val) < self._ACK_FIELD_MIN:
                return (
                    f"acknowledge_skill rejected: {field_name} is too "
                    f"short ({len(field_val)} < {self._ACK_FIELD_MIN} "
                    f"chars). Be concrete — quote a pattern from the "
                    f"skill or name the code site you will change."
                )
            if len(field_val) > self._ACK_FIELD_MAX:
                return (
                    f"acknowledge_skill rejected: {field_name} is too "
                    f"long ({len(field_val)} > {self._ACK_FIELD_MAX} "
                    f"chars). Trim to the essentials."
                )

        ack = {
            "skill": bs,
            "valuable_aspects": valuable,
            "kernel_application": application,
            "applicability": applicability,
        }
        unbound = self.feedback.record_skill_acknowledgement(aid, ack)
        if unbound:
            return (
                f"Acknowledged. '{bs}' released from {aid} (applicability="
                f"'unbind'). The skill STAYS available — a future item "
                f"can still bind it. Item {aid} continues as free "
                f"exploration. Proceed with `edit`; "
                f"update_plan is blocked until {aid} settles."
            )
        return (
            f"Acknowledged. Skill '{bs}' bound to {aid} with "
            f"applicability='apply'. Proceed with `edit`."
        )

    async def _handle_search_skills(self, args: dict) -> str:
        """Refill the SkillPool with hint-driven keyword search.

        Thin wrapper over ``feedback.skill_pool.refill(mode='append')``.
        Adds zero items to the currently-active plan; new candidates
        only become bindable on the agent's next ``update_plan`` call.
        """
        hint = (args.get("hint") or "").strip()
        if not hint:
            return (
                "search_skills rejected: hint is required and must be "
                "non-empty."
            )
        pool = getattr(self.feedback, "skill_pool", None)
        if pool is None:
            return (
                "search_skills unavailable: no skill pool attached to "
                "this run (DSL has no skill package)."
            )
        try:
            added = await pool.refill(
                llm=self.llm,
                config=self.config,
                hint=hint,
                mode="append",
                plan_version=self.feedback.plan_version,
            )
        except Exception as exc:
            return f"search_skills failed: {exc!r}"

        if not added:
            return (
                f"search_skills: 0 new candidates for hint={hint!r} "
                f"(all matches were duplicates or non-trackable). "
                f"Note: previously-unbound skills are NOT excluded — "
                f"they stay bindable at a lower priority tier. If the "
                f"skill you want is already in the pool but didn't "
                f"bind, refine the plan item's keywords rather than "
                f"calling search_skills again."
            )
        names = [getattr(s, "name", "") or "" for s in added]
        return (
            f"search_skills: added {len(added)} candidates to the pool: "
            f"{names}. They become available for keyword binding on "
            f"your next update_plan call."
        )

    async def _handle_compact(self, buffer) -> str:
        """Compress context via ``buffer.auto_compact``. Returns a reply string.

        The buffer is mutated in place; the no-op signal (input list
        identity) is folded into the buffer's ``auto_compact`` return
        value (False = no-op, True = actually compacted).
        """
        from .tools import TOOLS
        a = self.config.agent
        if len(buffer) >= a.compact_min_messages:
            if self.verbose:
                logger.info("  [compact] Compressing ...")
            compacted = await buffer.auto_compact(
                self.llm, self.task_dir,
                config=self.config, tools=TOOLS,
                feedback=self.feedback,
                last_diagnosis=self.feedback.last_diagnosis,
            )
            if compacted:
                self.compacted_this_turn = True
                return "Context compressed."
        return "Nothing to compress (not enough rounds)."

    # -- Nudge helpers ------------------------------------------------------

    def _nudge_if_no_edits(self, buffer, tool_calls: list,
                           consecutive_no_edit_turns: int):
        """Inject nudge messages when the agent isn't making edits."""
        nudge = self.feedback.build_phase_nudge()
        if nudge:
            buffer.append({"role": "user", "content": nudge})

        if consecutive_no_edit_turns >= self.config.agent.max_no_edit_turns:
            buffer.append({
                "role": "user",
                "content": (
                    f"[System] WARNING: {consecutive_no_edit_turns} consecutive turns "
                    f"without edits. You MUST make a code change via `edit` "
                    f"in your next turn."
                ),
            })
