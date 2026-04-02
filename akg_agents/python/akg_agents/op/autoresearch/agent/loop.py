"""
AgentLoop — Single-task autonomous optimization loop.

Orchestrator that delegates to:
  - ConversationAdapter (agent/llm_client.py) — LLM communication
  - TurnExecutor (agent/turn.py) — per-turn tool dispatch, rollback, eval
  - SessionStore (agent/session.py) — persistence, heartbeat, snapshots
  - FeedbackBuilder (agent/feedback.py) — plan tracking, system messages
  - FileLogger (agent/file_logger.py) — stdout tee to agent.log

Core loop:
  while budget remains:
      response = LLM(messages, tools)
      turn_result = TurnExecutor.execute(tool_calls, ...)
      update state from turn_result
"""

import json
import logging
import os
import time
from typing import Optional

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

from ..framework.runner import (
    ExperimentRunner, load_task_config,
    git_commit, git_current_commit, git_rollback_files,
)
from .compress import estimate_tokens, microcompact, auto_compact
from .tools import execute_run_eval, run_diagnostic_subagent
from .llm_client import ConversationAdapter
from .session import SessionStore
from .feedback import FeedbackBuilder
from .file_logger import FileLogger
from .turn import TurnExecutor

logger = logging.getLogger(__name__)


def _load_template(name: str) -> str:
    path = os.path.join(PROMPTS_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------

class AgentLoop:
    """
    Single-task autonomous optimization loop.

    Usage:
        loop = AgentLoop(task_dir="tasks/demo/vectoradd", device_id=0)
        result = asyncio.run(loop.run())
    """

    def __init__(
        self,
        task_dir: str,
        model: str = "claude-sonnet-4-6",
        device_id: int = 0,
        max_rounds: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbose: bool = True,
        skip_branch_switch: bool = False,
        resume: bool = False,
        llm_adapter=None,
        eval_fn=None,
    ):
        self.task_dir = os.path.abspath(task_dir)
        self.device_id = device_id
        self.verbose = verbose
        self.resume = resume

        # Load task config
        self.config = load_task_config(self.task_dir)
        self.max_rounds = max_rounds if max_rounds is not None else self.config.max_rounds

        # LLM: injected adapter takes priority, otherwise create standalone
        if llm_adapter is not None:
            self._llm = llm_adapter
            self.model = llm_adapter.model
            self.provider = llm_adapter.provider
        else:
            # Auto-detect provider
            if provider is None:
                if any(kw in model.lower() for kw in ("gpt-", "codex", "o1", "o3", "o4")):
                    provider = "openai"
                else:
                    provider = "anthropic"
            self.model = model
            self.provider = provider

            agent_cfg = self.config.agent
            self._llm = ConversationAdapter(
                model=model, provider=self.provider,
                call_timeout=agent_cfg.call_timeout,
                api_key=api_key, base_url=base_url,
                reasoning_effort=reasoning_effort,
                thinking_budget=agent_cfg.thinking_budget,
                llm_max_tokens=agent_cfg.llm_max_tokens,
                retry_initial_backoff=agent_cfg.retry_initial_backoff,
                retry_max_backoff_rate_limit=agent_cfg.retry_max_backoff_rate_limit,
                retry_max_backoff_other=agent_cfg.retry_max_backoff_other,
                verbose=verbose,
            )

        # Create components
        self._session = SessionStore(self.task_dir, self.config, verbose=verbose)
        self._feedback = FeedbackBuilder(self.config)
        self._file_logger = FileLogger(self.task_dir)

        # Framework runner (eval_fn injected here → run_eval_robust)
        self.runner = ExperimentRunner(
            self.task_dir, skip_branch_switch=skip_branch_switch,
            device_id=self.device_id, eval_fn=eval_fn,
        )
        self.runner.extra_commit_files = ["agent.log"]

        # Turn executor (uses self.model/self.provider set above)
        self._turn = TurnExecutor(
            task_dir=self.task_dir,
            config=self.config,
            runner=self.runner,
            feedback=self._feedback,
            session=self._session,
            llm=self._llm,
            model=self.model,
            provider=self.provider,
            device_id=device_id,
            verbose=verbose,
        )

        # Agent state
        self.eval_calls_made = 0
        self._baseline_commit: Optional[str] = None
        self._consecutive_failures = 0
        self._consecutive_no_edit_turns = 0
        self._consecutive_no_tool_turns = 0
        self._last_diagnosis: Optional[str] = None

        # Messages list
        self._messages: list = []

        # Build static system prompt
        self._system_prompt = self._build_system_prompt()

    # -- System prompt ----------------------------------------------------

    def _build_system_prompt(self) -> str:
        cfg = self.config

        metadata_section = ""
        metadata = dict(cfg.metadata) if cfg.metadata else {}
        # 自动检测运行时设备，覆盖任务配置中的静态 device 字段
        try:
            from ..framework.device import get_device_info
            dev_info = get_device_info()
            metadata["device"] = f"{dev_info['type'].upper()} ({dev_info['name']})"
        except Exception:
            pass
        if metadata:
            lines = ["## Task Metadata"]
            for k, v in metadata.items():
                lines.append(f"- {k}: {v}")
            metadata_section = "\n".join(lines) + "\n"

        context_entries = []
        if cfg.program_file:
            context_entries.append(("Agent Instructions", cfg.program_file))
        if cfg.ref_file:
            context_entries.append(("Reference Implementation", cfg.ref_file))
        for f in (cfg.context_files or []):
            context_entries.append(("Context", f))

        context_blocks = []
        per_file_limit = cfg.agent.system_context_file_truncate
        total_limit = cfg.agent.system_context_total_truncate
        total_chars = 0
        for label, fpath in context_entries:
            abs_path = os.path.join(self.task_dir, fpath)
            if not os.path.exists(abs_path):
                continue
            try:
                with open(abs_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
                if len(content) > per_file_limit:
                    content = content[:per_file_limit] + f"\n... [truncated at {per_file_limit} chars]"
                block = f"## {label}: {fpath}\n```\n{content}\n```"
                separator_cost = 2 if context_blocks else 0
                block_cost = separator_cost + len(block)
                if total_chars + block_cost > total_limit:
                    break
                context_blocks.append(block)
                total_chars += block_cost
            except Exception:
                pass

        context_files_section = "\n\n".join(context_blocks) if context_blocks else ""

        constraints_section = ""
        if cfg.constraints:
            lines = [
                "## Hard Constraints",
                "Results violating ANY constraint below are automatically DISCARDED.",
                "",
            ]
            for metric_name, (op_str, threshold) in cfg.constraints.items():
                lines.append(f"- {metric_name} {op_str} {threshold}")
            constraints_section = "\n".join(lines)

        base = _load_template("system_context.md").format_map({
            "task_name": cfg.name,
            "task_description": cfg.description,
            "metadata_section": metadata_section,
            "editable_files_list": "\n".join(f"- {f}" for f in cfg.editable_files),
            "primary_metric": cfg.primary_metric,
            "metric_direction": "lower is better" if cfg.lower_is_better else "higher is better",
            "constraints_section": constraints_section,
            "context_files_section": context_files_section,
        })

        # Knowledge section: task context + DSL docs + hardware info.
        # Shared between main agent and diagnose subagent.
        self._knowledge_prompt = base

        tool_protocol = _load_template("system_tool_protocol.md").format_map({
            "editable_files": str(list(cfg.editable_files)),
            "primary_metric": cfg.primary_metric,
            "metric_direction": "lower is better" if cfg.lower_is_better else "higher is better",
        })

        return base + "\n" + tool_protocol

    # -- Initial user message ----------------------------------------------

    def _build_initial_message(self) -> str:
        lines = [
            f"# Optimization Task: {self.config.name}",
            f"Primary metric: {self.config.primary_metric} "
            f"({'lower is better' if self.config.lower_is_better else 'higher is better'})",
            f"Eval budget: {self.max_rounds} rounds",
            "",
        ]

        lines.append("## Current Editable Files")
        editable_contents = self.runner.get_editable_contents()
        for fname, content in editable_contents.items():
            if len(content) > self.config.agent.editable_file_truncate:
                content = content[:self.config.agent.editable_file_truncate] + \
                    f"\n... [truncated at {self.config.agent.editable_file_truncate} chars]"
            lines.append(f"### {fname}\n```\n{content}\n```")

        if self.runner.best_result:
            best = self.runner.best_result
            bv = best.metrics.get(self.config.primary_metric)
            lines.append(f"\n## Baseline: {self.config.primary_metric}={bv}")

        lines.append(
            "\n## Instructions"
            "\n1. Call update_plan(plan=...) with '- [ ]' items to submit your plan. "
            "Items are assigned IDs (p1, p2, ...) and the first is activated."
            "\n2. Make edits with patch_file/write_file using plan_item_id matching the active item."
            "\n3. After your edits, the system runs quick_check → eval and auto-settles the item."
            "\n4. When all items are settled, submit a new plan or call finish."
            "\n\n★ Algorithmic changes FIRST, parameter tuning SECOND."
            "\n\nStart optimizing now. Act, don't explain."
        )

        return "\n".join(lines)

    # -- Baseline ----------------------------------------------------------

    async def _ensure_baseline(self):
        if self.runner.best_result is not None:
            if self._baseline_commit is None:
                self._baseline_commit = git_current_commit(self.task_dir)
            if self.verbose:
                bv = self.runner.best_result.metrics.get(self.config.primary_metric, "?")
                print(f"[AgentLoop] Baseline: {self.config.primary_metric}={bv}", flush=True)
            return

        if self.verbose:
            print("[AgentLoop] Running baseline eval …", flush=True)

        pre_commit = git_current_commit(self.task_dir)
        eval_json = await execute_run_eval("baseline — unmodified code", self.runner,
                                            raw_output_tail=self.config.agent.raw_output_tail)
        self.eval_calls_made += 1

        try:
            record = json.loads(eval_json)
            self._baseline_commit = record.get("commit") or pre_commit
            if self.verbose:
                metric = record.get("metrics", {}).get(self.config.primary_metric, "?")
                print(f"[AgentLoop] Baseline: {self.config.primary_metric}={metric}", flush=True)
        except json.JSONDecodeError:
            self._baseline_commit = pre_commit

    # -- Finalize ----------------------------------------------------------

    def _generate_report(self):
        if self.verbose:
            print("[AgentLoop] Generating report …", flush=True)
        try:
            self.runner.generate_report()
        except Exception as e:
            print(f"[AgentLoop] Report generation failed: {e}")

    def _finalize_clean(self, total_api_calls: int):
        self._generate_report()

        # Save plan.md to task root before session cleanup
        plan_content = self._feedback.format_plan_file()
        if plan_content:
            plan_path = os.path.join(self.task_dir, "plan.md")
            try:
                with open(plan_path, "w", encoding="utf-8") as f:
                    f.write(plan_content)
            except Exception as e:
                logger.warning(f"Failed to write plan.md: {e}")

        self._session.cleanup()
        best = self.runner.best_result
        best_metric = ""
        if best:
            bv = best.metrics.get(self.config.primary_metric)
            if bv is not None:
                best_metric = f" | best {self.config.primary_metric}={bv}"
        msg = (f"final: {self.config.name} — {self.eval_calls_made} eval rounds, "
               f"{total_api_calls} api calls (clean){best_metric}")
        cr = git_commit(self.task_dir, msg, task_name=self.config.name)
        if cr.committed and self.verbose:
            print(f"[AgentLoop] Final commit: {cr.hash}")

        # Switch back to original branch (exp branch preserved for inspection)
        from ..framework.runner import git_cleanup_branch
        if self.runner.original_branch and self.runner.branch_name:
            if self.verbose:
                print(f"[AgentLoop] Switching back to '{self.runner.original_branch}'")
            git_cleanup_branch(
                self.task_dir,
                self.runner.branch_name,
                self.runner.original_branch,
                session_dir=self.config.agent.session_dir,
                heartbeat_file=self.config.agent.heartbeat_file,
            )

    def _finalize_interrupted(self):
        self._generate_report()
        if self.verbose:
            print("[AgentLoop] Non-clean exit — skipping commit for --resume")

    # -- Forced diagnose ----------------------------------------------------

    async def _force_diagnose(self, turn_result) -> str:
        """Auto-trigger diagnostic subagent after consecutive failures."""
        # Build error context from recent feedback
        error_ctx = turn_result.feedback or "Multiple consecutive failures."
        if self.verbose:
            print(f"[AgentLoop] Auto-diagnose triggered "
                  f"({self._consecutive_failures} consecutive failures) …",
                  flush=True)
        try:
            sa = self.config.agent
            diagnosis = await run_diagnostic_subagent(
                llm=self._llm,
                error_context=error_ctx,
                current_code=self.runner.get_editable_contents(),
                task_description=self.config.description,
                task_dir=self.task_dir,
                knowledge_prompt=self._knowledge_prompt,
                max_iterations=sa.subagent_max_iterations,
                code_truncate=sa.subagent_code_truncate,
                result_truncate=sa.subagent_result_truncate,
            )
        except Exception as e:
            diagnosis = f"Diagnostic subagent failed: {e}"
        if self.verbose:
            print(f"[AgentLoop] Diagnosis:\n{diagnosis}", flush=True)
        return f"[Diagnostic Subagent Report]\n{diagnosis}"

    # -- Helpers ------------------------------------------------------------

    def _best_metric_str(self) -> str:
        best = self.runner.best_result
        if best:
            bv = best.metrics.get(self.config.primary_metric)
            if bv is not None:
                return f"{self.config.primary_metric}={bv}"
        return ""

    def _save_session(self, total_api_calls: int):
        self._session.save({
            "model": self.model,
            "eval_calls_made": self.eval_calls_made,
            "total_api_calls": total_api_calls,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_no_edit_turns": self._consecutive_no_edit_turns,
            "baseline_commit": self._baseline_commit,
            "plan": self._feedback.format_plan_file() or self._feedback.plan,
            "plan_state": self._feedback.plan_state_to_dict(),
            "last_diagnosis": self._last_diagnosis,
        })

    def _update_heartbeat(self, total_api_calls: int, extra: str = ""):
        self._session.update_heartbeat(
            total_api_calls=total_api_calls,
            eval_calls_made=self.eval_calls_made,
            max_rounds=self.max_rounds,
            model=self.model,
            best_str=self._best_metric_str(),
            extra=extra,
        )

    # -- Main loop --------------------------------------------------------

    async def run(self) -> dict:
        """Main agent loop. Guarantees cleanup of file logger and heartbeat."""
        self._file_logger.open()
        try:
            return await self._run_body()
        finally:
            self._session.remove_heartbeat()
            self._file_logger.close()

    async def _run_body(self) -> dict:
        """Core run logic — resource cleanup is guaranteed by run()."""
        self._session.check_lock()

        total_api_calls = 0
        clean_exit = False

        if self.verbose:
            print(f"\n[AgentLoop] Starting: {self.config.name}")
            print(f"[AgentLoop] task_dir: {self.task_dir}")
            print(f"[AgentLoop] max_rounds={self.max_rounds}, "
                  f"device_id={self.device_id}, model={self.model}")
            print(f"[AgentLoop] editable_files={self.config.editable_files}")

        await self._llm.check_connection(verbose=self.verbose)

        # Resume
        session_restored = False
        if self.resume:
            state = self._session.load()
            if state:
                session_restored = True
                total_api_calls = state["total_api_calls"]
                self.eval_calls_made = state["eval_calls_made"]
                self._consecutive_failures = state["consecutive_failures"]
                self._consecutive_no_edit_turns = state["consecutive_no_edit_turns"]
                self._baseline_commit = state.get("baseline_commit")
                self._last_diagnosis = state.get("last_diagnosis")
                if "plan_state" in state:
                    self._feedback.plan_state_from_dict(state["plan_state"])
                elif "plan" in state and state["plan"]:
                    # Legacy fallback: old session only has plan text
                    self._feedback.plan = state["plan"]

        await self._ensure_baseline()

        # Build initial user message
        self._messages = [{"role": "user", "content": self._build_initial_message()}]

        if session_restored:
            resume_info = (
                f"\n\n## Resumed Session"
                f"\nEval rounds used: {self.eval_calls_made}/{self.max_rounds}"
                f"\nAPI calls so far: {total_api_calls}"
            )
            resume_info += f"\n\n{self._feedback.format_status()}"
            if self._feedback.plan:
                resume_info += f"\n\n## Original Plan Text\n{self._feedback.plan}"
            best = self.runner.best_result
            if best:
                bv = best.metrics.get(self.config.primary_metric)
                resume_info += f"\nCurrent best: {self.config.primary_metric}={bv}"
            self._messages[0]["content"] += resume_info
            if self._feedback.must_replan and self._last_diagnosis:
                self._messages.append({
                    "role": "user",
                    "content": (
                        f"[System] ⚠ MANDATORY DIRECTION CHANGE (from prior session). "
                        f"Edits are BLOCKED until you call update_plan(plan=...) "
                        f"with a new plan based on the diagnostic report below."
                        f"\n\n{self._last_diagnosis}"
                    ),
                })

        self._save_session(total_api_calls)
        self._update_heartbeat(total_api_calls, extra="baseline complete")

        max_turns = self.max_rounds * self.config.agent.max_turns_multiplier

        # Transcript files, updated each turn:
        #   transcript_full.jsonl   — append-only, all messages ever (survives compact)
        #   transcript_latest.jsonl — overwrite, current self._messages state
        from .compress import _save_transcript as _save_tx
        _prev_msg_count = 0

        def _save_transcript():
            nonlocal _prev_msg_count
            sd = self.config.agent.session_dir
            try:
                # Append only new messages to full history
                new_msgs = self._messages[_prev_msg_count:]
                if new_msgs:
                    _save_tx(new_msgs, self.task_dir, session_dir=sd,
                             filename="transcript_full.jsonl", mode="a")
                _prev_msg_count = len(self._messages)
                # Overwrite latest state (compact summary + new messages)
                _save_tx(self._messages, self.task_dir, session_dir=sd,
                         filename="transcript_latest.jsonl", mode="w")
            except Exception:
                pass

        # -- The core loop --
        while self.eval_calls_made < self.max_rounds and total_api_calls < max_turns:
            if self.verbose:
                print(f"\n[AgentLoop] Turn {total_api_calls + 1} "
                      f"(eval: {self.eval_calls_made}/{self.max_rounds})")
            self._update_heartbeat(total_api_calls, f"turn {total_api_calls + 1}")

            # Context management
            a = self.config.agent
            microcompact(self._messages,
                         min_chars=a.microcompact_min_chars,
                         keep_recent=a.microcompact_keep_recent)
            if a.context_limit:
                token_threshold = int(a.context_limit * a.compression_threshold)
                if estimate_tokens(self._messages, a.chars_per_token) > token_threshold:
                    if self.verbose:
                        print("[AgentLoop] Auto-compacting context …", flush=True)
                    self._messages = await auto_compact(
                        self._messages, self._llm.client, self.model,
                        self.provider, self.task_dir,
                        text_limit=a.auto_compact_text_limit,
                        summary_max_tokens=a.compact_summary_max_tokens,
                        session_dir=a.session_dir,
                    )
                    _prev_msg_count = 0  # messages replaced, reset tracker

            # Call LLM
            try:
                response = await self._llm.call(self._system_prompt, self._messages)
            except Exception as e:
                if self.verbose:
                    print(f"[AgentLoop] LLM error: {e}", flush=True)
                self._save_session(total_api_calls)
                break  # non-clean exit → preserve session for --resume

            total_api_calls += 1
            self._llm.append_assistant(self._messages, response)

            # If model didn't call tools → nudge or exit
            no_tools = (self._llm.get_stop_reason(response) != "tool_use")
            if not no_tools:
                tool_calls = self._llm.extract_tool_calls(response)
                if not tool_calls:
                    no_tools = True

            if no_tools:
                if self.verbose:
                    text = self._llm.get_response_text(response)
                    if text:
                        print(f"  [LLM] {text[:200]}")

                self._consecutive_no_tool_turns += 1
                if self._feedback.phase == "active":
                    self._consecutive_no_edit_turns += 1

                nudge = self._feedback.build_phase_nudge()
                if nudge is None:
                    clean_exit = True
                    break
                # In replanning with no budget or too many idle turns, exit
                if (self._feedback.phase == "replanning"
                        and (self.max_rounds - self.eval_calls_made <= 0
                             or self._consecutive_no_tool_turns > 2)):
                    clean_exit = True
                    break

                self._messages.append({"role": "user", "content": nudge})
                if self.verbose:
                    print(f"  [nudge] {nudge}")
                continue

            self._consecutive_no_tool_turns = 0

            # Delegate turn execution to TurnExecutor
            try:
                turn_result = await self._turn.execute(
                    tool_calls=tool_calls,
                    messages=self._messages,
                    eval_calls_made=self.eval_calls_made,
                    max_rounds=self.max_rounds,
                    consecutive_failures=self._consecutive_failures,
                    consecutive_no_edit_turns=self._consecutive_no_edit_turns,
                    total_api_calls=total_api_calls,
                    baseline_commit=self._baseline_commit,
                )
            except Exception as e:
                import traceback as _tb
                if self.verbose:
                    print(f"[AgentLoop] Turn crashed: {e}\n{_tb.format_exc()}",
                          flush=True)
                # Rollback editable files to prevent dirty state from
                # leaking into the next round.
                git_rollback_files(self.task_dir, self.config.editable_files)
                self._consecutive_failures += 1
                self._messages.append({
                    "role": "user",
                    "content": f"[System] Turn crashed: {e}. Files rolled back. Continuing.",
                })
                self._save_session(total_api_calls)
                _save_transcript()
                continue

            # Unpack updated state from TurnResult
            self.eval_calls_made = turn_result.eval_calls_made
            self._consecutive_failures = turn_result.consecutive_failures
            self._consecutive_no_edit_turns = turn_result.consecutive_no_edit_turns
            self._messages = turn_result.messages

            self._save_session(total_api_calls)
            _save_transcript()

            if turn_result.has_finish:
                clean_exit = True
                break

            # Force diagnose when threshold reached (before abort check)
            if (self._consecutive_failures > 0
                    and self._consecutive_failures % self.config.agent.diagnose_suggest_threshold == 0):
                try:
                    diagnosis = await self._force_diagnose(turn_result)
                    if diagnosis:
                        self._feedback.require_replan()
                        self._last_diagnosis = diagnosis
                        self._messages.append({
                            "role": "user",
                            "content": (
                                f"[System] ⚠ MANDATORY DIRECTION CHANGE — "
                                f"{self._consecutive_failures} consecutive failures.\n"
                                f"Edits are BLOCKED until you call update_plan(plan=...) "
                                f"with a new plan based on the diagnostic report below."
                                f"\n\n{diagnosis}"
                            ),
                        })
                        self._consecutive_failures = 0
                        self._save_session(total_api_calls)
                except Exception as e:
                    import traceback as _tb
                    if self.verbose:
                        print(f"[AgentLoop] Diagnose failed: {e}\n{_tb.format_exc()}",
                              flush=True)

            if self._consecutive_failures >= self.config.agent.max_consecutive_failures:
                if self.verbose:
                    print(f"[AgentLoop] ABORT: {self._consecutive_failures} consecutive failures")
                clean_exit = True
                break

        # -- Done --
        if self.eval_calls_made >= self.max_rounds or total_api_calls >= max_turns:
            clean_exit = True

        if self.verbose:
            print(f"\n[AgentLoop] Done. eval={self.eval_calls_made}, api_calls={total_api_calls}")
            best = self.runner.best_result
            ref = self.runner.ref_latency
            if best and ref and ref > 0:
                best_lat = best.metrics.get("latency_us")
                if isinstance(best_lat, (int, float)) and best_lat > 0:
                    print(f"[AgentLoop] Best: {best_lat:.2f} us | "
                          f"Ref: {ref:.2f} us | "
                          f"Speedup: {ref / best_lat:.2f}x", flush=True)
            print(f"\n{self.runner.status()}")

        if clean_exit:
            self._finalize_clean(total_api_calls)
        else:
            self._finalize_interrupted()

        best = self.runner.best_result
        best_metrics = dict(best.metrics) if best else {}
        ref = self.runner.ref_latency
        if ref is not None:
            best_metrics["ref_latency_us"] = round(ref, 2)
            best_lat = best_metrics.get("latency_us")
            if isinstance(best_lat, (int, float)) and best_lat > 0:
                best_metrics["speedup_vs_ref"] = round(ref / best_lat, 3)
        return {
            "task": self.config.name,
            "task_dir": self.task_dir,
            "model": self.model,
            "total_api_calls": total_api_calls,
            "eval_rounds": self.eval_calls_made,
            "best_metrics": best_metrics or None,
            "final_status": self.runner.status(),
        }
