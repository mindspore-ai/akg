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

import asyncio
import json
import logging
import os
import time
from typing import Optional

from ..framework.runner import ExperimentRunner, load_task_config
from .compress import (
    estimate_full_request_tokens,
    _is_prompt_too_long,
    _plan_analysis_path, _read_safe,
)
from .conversation import ConversationBuffer
from .counters import RunCounters
from .tools import execute_run_eval, TOOLS
from .subagents import DiagnoseHandler
from .llm_client import ConversationAdapter
from .session import SessionStore
from .feedback import FeedbackBuilder
from .file_logger import FileLogger
from .prompt_builder import (
    build_initial_message,
    build_system_prompt,
)
from .skill_pool import SkillPool
from dataclasses import dataclass as _dataclass


@_dataclass
class PostEvalDecision:
    """Post-eval action: diagnose or none."""
    kind: str  # 'diagnose' | 'none'
from .turn import TurnExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Post-eval decision: diagnose or none.
#
# ``diagnose`` fires when consecutive_failures crosses a threshold multiple.
# ``_decide_post_eval_action`` returns a PostEvalDecision;
# ``_execute_post_eval_action`` dispatches it.
# ---------------------------------------------------------------------------


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
                max_retries=agent_cfg.llm_max_retries,
                connection_check_timeout=agent_cfg.llm_connection_check_timeout,
                verbose=verbose,
            )

        # Framework runner first (owns GitRepo + FileStateManager) so the
        # session and other components can borrow its sub-components
        # instead of constructing their own.
        self.runner = ExperimentRunner(
            self.task_dir, skip_branch_switch=skip_branch_switch,
            device_id=self.device_id, eval_fn=eval_fn,
        )
        self.runner.extra_commit_files = ["agent.log"]

        # Create components — SessionStore borrows runner.git for its
        # head/dirty checks, no module-level git imports needed.
        self._session = SessionStore(
            self.task_dir, self.config, git=self.runner.git, verbose=verbose,
        )
        # SkillBuilder owns the per-skill registry; SkillPool owns the
        # keyword-ranked candidate list and the keyword-matching
        # queries for update_plan calls. Both are passed into
        # FeedbackBuilder at construction so callers never see a
        # half-initialized feedback object.
        from .skill_builder import SkillBuilder
        self._skill_builder = SkillBuilder(self.config, task_dir=self.task_dir)
        self._feedback = FeedbackBuilder(
            self.config,
            task_dir=self.task_dir,
            skill_builder=self._skill_builder,
            skill_pool=SkillPool(self._skill_builder),
        )
        self._file_logger = FileLogger(self.task_dir)

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

        # Agent state. The keyword-ranked candidate skill list lives on
        # ``feedback.skill_pool`` (a SkillPool instance attached above);
        # every downstream consumer — initial message, compress
        # bootstrap, TurnExecutor keyword matching — reads it through
        # that handle.
        self._baseline_commit: Optional[str] = None
        self._counters = RunCounters()  # single owner of all per-run counters
        # ``last_diagnosis`` text now lives on FeedbackBuilder paired with
        # must_replan — read via self._feedback.last_diagnosis.

        # Conversation buffer (single owner of the message list)
        self._buffer = ConversationBuffer()

        # Build the static system prompt. Returns both the full prompt
        # (knowledge + tool protocol) for the main agent and the
        # knowledge-only base that the diagnose subagent reuses.
        self._system_prompt, self._knowledge_prompt = build_system_prompt(
            self.config, self.task_dir,
        )
        # Share the knowledge base with TurnExecutor for subagents and compact.
        self._turn.knowledge_prompt = self._knowledge_prompt

        # Diagnose handler — post-eval subagent. Fires when
        # consecutive_failures crosses a multiple of
        # diagnose_suggest_threshold. knowledge_prompt is captured by
        # value here (static for the run lifetime).
        self._diagnose_handler = DiagnoseHandler(
            llm=self._llm,
            config=self.config,
            task_dir=self.task_dir,
            runner=self.runner,
            counters=self._counters,
            feedback=self._feedback,
            buffer=self._buffer,
            knowledge_prompt=self._knowledge_prompt,
            verbose=self.verbose,
            save_session_cb=self._save_session,
            heartbeat_cb=lambda extra: self._update_heartbeat(extra=extra),
        )

    # -- Baseline ----------------------------------------------------------

    async def _ensure_baseline(self):
        if self.runner.best_result is not None:
            if self._baseline_commit is None:
                self._baseline_commit = self.runner.git.current_commit()
            if self.verbose:
                bv = self.runner.best_result.metrics.get(self.config.primary_metric, "?")
                logger.info(f"[AgentLoop] Baseline: {self.config.primary_metric}={bv}")
            return

        if self.verbose:
            logger.info("[AgentLoop] Running baseline eval …")

        pre_commit = self.runner.git.current_commit()
        eval_json = await execute_run_eval("baseline — unmodified code", self.runner,
                                            raw_output_tail=self.config.agent.raw_output_tail)
        # Baseline isn't a "real" eval result for the gating rules — just
        # bump the round counter directly without calling record_eval (which
        # would update consecutive_failures / consecutive_no_improvement
        # based on outcome and we don't want either side-effect here).
        self._counters.eval_calls_made += 1

        try:
            record = json.loads(eval_json)
            self._baseline_commit = record.get("commit") or pre_commit
            if self.verbose:
                metric = record.get("metrics", {}).get(self.config.primary_metric, "?")
                logger.info(f"[AgentLoop] Baseline: {self.config.primary_metric}={metric}")
        except json.JSONDecodeError:
            self._baseline_commit = pre_commit

    # -- Finalize ----------------------------------------------------------

    def _report_and_commit(self, tag: str):
        """Generate report + commit — always bundled."""
        if self.verbose:
            logger.info("[AgentLoop] Generating report …")
        try:
            self.runner.generate_report()
        except Exception as e:
            logger.warning(f"[AgentLoop] Report generation failed: {e}")

        best = self.runner.best_result
        best_metric = ""
        if best:
            bv = best.metrics.get(self.config.primary_metric)
            if bv is not None:
                best_metric = f" | best {self.config.primary_metric}={bv}"
        msg = (f"final: {self.config.name} — "
               f"{self._counters.eval_calls_made} eval rounds ({tag}){best_metric}")
        cr = self.runner.git.commit(msg, task_name=self.config.name)
        if self.verbose and cr.committed:
            logger.info(f"[AgentLoop] Final commit ({tag}): {cr.hash}")

    def _finalize_clean(self):
        self._report_and_commit("clean")
        self._session.cleanup()

        # Switch back to original branch (exp branch preserved for inspection)
        if self.runner.original_branch and self.runner.branch_name:
            if self.verbose:
                logger.info(f"[AgentLoop] Switching back to '{self.runner.original_branch}'")
            self.runner.git.cleanup_branch(
                self.runner.branch_name,
                self.runner.original_branch,
                session_dir=self.config.agent.session_dir,
                heartbeat_file=self.config.agent.heartbeat_file,
            )

    def _finalize_interrupted(self):
        self._report_and_commit("interrupted")

    # -- Helpers ------------------------------------------------------------

    def _best_metric_str(self) -> str:
        best = self.runner.best_result
        if best:
            bv = best.metrics.get(self.config.primary_metric)
            if bv is not None:
                return f"{self.config.primary_metric}={bv}"
        return ""

    def _force_rebuild(self) -> None:
        """Emergency context rebuild — no LLM call.

        Replaces ``self._buffer`` contents in place via
        ``ConversationBuffer.force_rebuild``.
        """
        self._buffer.force_rebuild(
            self.task_dir, self.config,
            feedback=self._feedback,
            last_diagnosis=self._feedback.last_diagnosis,
            best_metric_str=self._best_metric_str(),
        )

    def _save_session(self):
        self._session.save({
            "model": self.model,
            "counters": self._counters.to_dict(),
            "baseline_commit": self._baseline_commit,
            "plan": self._feedback.format_plan_file() or self._feedback.plan,
            "plan_state": self._feedback.plan_state_to_dict(),
            "skill_state": self._skill_builder.skill_state_to_dict(),
            "last_diagnosis": self._feedback.last_diagnosis,
        })

    def _rehydrate_pool_from_plan(self):
        """Append any skill referenced by plan_items but missing from pool.

        Called once on resume. Session only persists plan_state, not the
        pool itself, and the resume refill is category-limited. Any
        ``backing_skill`` name from the prior session's
        ``search_skills()`` append may land outside the filter.
        Pull those by name from the catalog so content lookups work
        during binding decisions.
        """
        pool = getattr(self._feedback, "skill_pool", None)
        if pool is None:
            return
        plan_items = getattr(self._feedback, "_plan_items", []) or []
        wanted: set[str] = set()
        for item in plan_items:
            bs = item.get("backing_skill")
            if bs:
                wanted.add(bs)
        if not wanted:
            return
        existing = {getattr(s, "name", "") or "" for s in pool}
        missing = wanted - existing
        if not missing:
            return
        try:
            from .skill_adapter import _get_catalog
            catalog = _get_catalog()
            dsl = getattr(self.config, "dsl", "") or ""
            all_skills = catalog.load_by_dsl(dsl) if dsl else []
            to_add = [s for s in all_skills
                      if (getattr(s, "name", "") or "") in missing]
            added = pool.append_new(to_add)
            if added:
                logger.info(
                    f"[AgentLoop] rehydrated {len(added)} skill(s) from "
                    f"plan_items: {[getattr(s, 'name', '') for s in added]}",
                )
        except Exception as exc:
            logger.warning(
                f"[AgentLoop] pool rehydration FAILED: {exc!r}",
            )

    # -- Post-eval action decision (priority-ordered) ----------------------
    #
    # See PostEvalDecision docstring at the top of this file for the
    # rationale. The split between _decide_post_eval_action and
    # _execute_post_eval_action keeps the priority logic readable and
    # makes both halves independently testable.

    async def _decide_post_eval_action(self, turn_result) -> PostEvalDecision:
        """Pick exactly ONE post-eval action (diagnose or none)."""
        if self._diagnose_handler.should_fire(turn_result):
            return PostEvalDecision(kind="diagnose")
        return PostEvalDecision(kind="none")

    async def _execute_post_eval_action(
        self, decision: PostEvalDecision, turn_result,
    ) -> None:
        """Apply a single PostEvalDecision."""
        if decision.kind == "diagnose":
            await self._diagnose_handler.apply(turn_result)
        # 'none' → nothing to do

    def _update_heartbeat(self, extra: str = ""):
        a = self.config.agent
        ctx_tokens = 0
        if a.context_limit and hasattr(self, '_buffer'):
            try:
                ctx_tokens = estimate_full_request_tokens(
                    self._buffer.view(), self._system_prompt,
                    tools=TOOLS, chars_per_token=a.chars_per_token)
            except Exception:
                pass
        elapsed = time.monotonic() - self._start_time if hasattr(self, '_start_time') else 0
        self._session.update_heartbeat(
            self._counters,
            max_rounds=self.max_rounds,
            model=self.model,
            best_str=self._best_metric_str(),
            extra=extra,
            phase=self._feedback.phase if hasattr(self, '_feedback') else "",
            context_tokens=ctx_tokens,
            context_limit=a.context_limit or 0,
            elapsed_sec=elapsed,
        )

    # -- Main loop --------------------------------------------------------

    async def run(self) -> dict:
        """Main agent loop. Guarantees cleanup of file logger and heartbeat."""
        self._file_logger.open()
        # Wire the Python logging module to the FileLogger stdout tee.
        # After this point any ``logger.info(...)`` under the
        # ``akg_agents.op.autoresearch`` namespace reaches both the
        # terminal AND agent.log — no more need for bare ``print()``.
        #
        # propagate=False prevents records from bubbling up to the root
        # logger, which may have its own handlers (e.g. from
        # ``basicConfig`` in the project's __init__.py). Without this,
        # every log line would appear twice on the terminal.
        #
        # Both the handler AND the level are saved/restored in finally
        # so a single AgentLoop.run() does not permanently mutate
        # process-global logger state.
        import sys
        _log_handler = logging.StreamHandler(sys.stdout)
        _log_handler.setFormatter(
            logging.Formatter("%(message)s"),
        )
        _autoresearch_logger = logging.getLogger(
            "akg_agents.op.autoresearch",
        )
        _prev_level = _autoresearch_logger.level
        _prev_propagate = _autoresearch_logger.propagate
        _autoresearch_logger.addHandler(_log_handler)
        _autoresearch_logger.setLevel(logging.INFO)
        _autoresearch_logger.propagate = False
        try:
            return await self._run_body()
        finally:
            _autoresearch_logger.removeHandler(_log_handler)
            _autoresearch_logger.setLevel(_prev_level)
            _autoresearch_logger.propagate = _prev_propagate
            self._session.remove_heartbeat()
            self._file_logger.close()

    async def _run_body(self) -> dict:
        """Core run logic — resource cleanup is guaranteed by run()."""
        self._start_time = time.monotonic()
        self._session.check_lock()

        clean_exit = False

        if self.verbose:
            logger.info(f"\n[AgentLoop] Starting: {self.config.name}")
            logger.info(f"[AgentLoop] task_dir: {self.task_dir}")
            logger.info(f"[AgentLoop] max_rounds={self.max_rounds}, "
                  f"device_id={self.device_id}, model={self.model}")
            logger.info(f"[AgentLoop] editable_files={self.config.editable_files}")

        await self._llm.check_connection(
            timeout=self.config.agent.llm_connection_check_timeout,
            verbose=self.verbose)

        # Resume
        session_restored = False
        if self.resume:
            state = self._session.load()
            if state:
                session_restored = True
                # RunCounters.from_dict accepts both the new "counters" key
                # and the legacy top-level layout — see its docstring.
                self._counters = RunCounters.from_dict(
                    state.get("counters") or state
                )
                # Both handlers were constructed in __init__ with the
                # pre-resume counters object. Resume swaps the counters
                # wholesale, so rebind each handler's reference or they
                # would continue to read gating / retry state from the
                # stale instance while _save_session serializes the fresh
                # one.
                self._diagnose_handler.rebind_counters(self._counters)
                self._baseline_commit = state.get("baseline_commit")
                if "plan_state" in state:
                    self._feedback.plan_state_from_dict(state["plan_state"])
                elif "plan" in state and state["plan"]:
                    # Legacy fallback: old session only has plan text
                    self._feedback.plan = state["plan"]
                # last_diagnosis text lives on FeedbackBuilder; restore
                # via its setter (paired with must_replan, which
                # plan_state_from_dict already restored above).
                self._feedback.last_diagnosis = state.get("last_diagnosis")

                # Restore skill state (empty dict → empty SkillBuilder,
                # which is the backward-compat no-op path).
                self._skill_builder.skill_state_from_dict(
                    state.get("skill_state", {})
                )

        # Seed the SkillPool concurrently with baseline measurement.
        # SkillPool.refill is the only entry point — it owns the
        # diagnostic prints, dedup, registry registration, and the
        # mode='replace' vs 'append' switch. Agent-driven refills
        # happen later via the search_skills tool handler.
        refill_task = asyncio.create_task(
            self._feedback.skill_pool.refill(
                llm=self._llm,
                config=self.config,
                mode="replace",
                plan_version=self._feedback.plan_version,
                include_categories=["guide"],
            )
        )
        await self._ensure_baseline()
        try:
            await refill_task
        except Exception as exc:
            logger.info(
                f"[AgentLoop] skill_pool refill FAILED: {exc!r}",
            )

        # Resume rehydration: the refill above only restocks skills whose
        # category is in include_categories. Any guide/example/case that
        # was added by the previous session's search_skills() call would
        # be missing, leaving plan_items with skill names the pool can't
        # resolve for content/description lookup. Walk plan_items, collect
        # referenced names, and pull them in by name from the catalog.
        if session_restored:
            self._rehydrate_pool_from_plan()

        # Build initial user message (or rebuild from summary/messages file)
        _resumed_from_compact = False
        sd = self.config.agent.session_dir
        # ``plan_analysis.md`` is written at every compact cycle (see
        # compress._analyze_plan_md), so its presence is the new
        # "previously compacted" signal (replacing the removed
        # summary.md).
        sp = _plan_analysis_path(self.task_dir, sd)
        if session_restored and _read_safe(sp):
            # Previously compacted — rebuild from disk (bootstrap has plan/diagnosis)
            _resumed_from_compact = True
            self._force_rebuild()
            self._buffer.append({
                "role": "user",
                "content": (
                    f"[Resumed from session. "
                    f"Eval {self._counters.eval_calls_made}/{self.max_rounds}. "
                    f"Best: {self._best_metric_str()}]"
                ),
            })
        elif session_restored:
            # No summary.md but have saved messages — restore full conversation.
            # ``load_latest`` internally replaces the buffer and triggers
            # on_buffer_rebuilt; we don't need to reset tracking manually.
            if self._buffer.load_latest(self.task_dir, session_dir=sd):
                _resumed_from_compact = True  # skip append-resume-info path
                self._buffer.append({
                    "role": "user",
                    "content": (
                        f"[Resumed from saved messages. "
                        f"Eval {self._counters.eval_calls_made}/{self.max_rounds}. "
                        f"Best: {self._best_metric_str()}]"
                    ),
                })
                if self.verbose:
                    logger.info(f"[AgentLoop] Restored {len(self._buffer)} messages "
                          f"from messages_latest.jsonl")
            else:
                self._buffer.replace([{"role": "user",
                                       "content": build_initial_message(
                                           self.config, self.task_dir, self.runner,
                                           self._feedback,
                                           self.max_rounds,
                                           session_restored=True,
                                       )}])
        else:
            self._buffer.replace([{"role": "user",
                                   "content": build_initial_message(
                                       self.config, self.task_dir, self.runner,
                                       self._feedback,
                                       self.max_rounds,
                                       session_restored=False,
                                   )}])

        if session_restored and not _resumed_from_compact:
            # Only append resume info to initial message (not to compact boundary)
            resume_info = (
                f"\n\n## Resumed Session"
                f"\nEval rounds used: {self._counters.eval_calls_made}/{self.max_rounds}"
                f"\nAPI calls so far: {self._counters.total_api_calls}"
            )
            resume_info += f"\n\n{self._feedback.format_status()}"
            if self._feedback.plan:
                resume_info += f"\n\n## Original Plan Text\n{self._feedback.plan}"
            best = self.runner.best_result
            if best:
                bv = best.metrics.get(self.config.primary_metric)
                resume_info += f"\nCurrent best: {self.config.primary_metric}={bv}"
            self._buffer.append_to_first(resume_info)
            if self._feedback.must_replan and self._feedback.last_diagnosis:
                self._buffer.append({
                    "role": "user",
                    "content": (
                        f"[System] ⚠ DIRECTION CHANGE (from prior session). "
                        f"Submit replacement item(s) via update_plan(items=[...]) "
                        f"based on the diagnostic report below."
                        f"\n\n{self._feedback.last_diagnosis}"
                    ),
                })

        # The SkillPool is rebuilt fresh by ``skill_pool.refill`` above
        # (it runs concurrently with ``_ensure_baseline``); the skill
        # registry itself was restored via skill_state_from_dict earlier
        # in this function. Because no skill is terminal, the pool
        # doesn't need to exclude anything on resume — priority is
        # derived from SkillRecord.tier() (applied_versions /
        # unbound_at_versions badges).

        self._save_session()
        self._update_heartbeat(extra="baseline complete")

        max_turns = self.max_rounds * self.config.agent.max_turns_multiplier

        # Transcript files, updated each turn:
        #   messages_full.jsonl   — append-only, all messages ever (survives compact)
        #   messages_latest.jsonl — overwrite, current buffer state
        _prev_msg_count = 0

        def _save_messages_to_disk():
            nonlocal _prev_msg_count
            sd = self.config.agent.session_dir
            try:
                _prev_msg_count = self._buffer.save_full_increment(
                    self.task_dir, session_dir=sd, since_idx=_prev_msg_count,
                )
                self._buffer.save_latest(self.task_dir, session_dir=sd)
            except Exception:
                pass

        # -- The core loop --
        while (self._counters.eval_calls_made < self.max_rounds
                and self._counters.total_api_calls < max_turns):
            if self.verbose:
                logger.info(f"\n[AgentLoop] Turn {self._counters.total_api_calls + 1} "
                      f"(eval: {self._counters.eval_calls_made}/{self.max_rounds})")
            self._update_heartbeat(extra=f"turn {self._counters.total_api_calls + 1}")

            # Context management
            a = self.config.agent
            self._buffer.microcompact(
                min_chars=a.microcompact_min_chars,
                keep_recent=a.microcompact_keep_recent,
            )
            # Skill injection cleanup: KEEP'd + current active stay,
            # (skill auto-injection and reference injection removed —
            # the agent pulls skills/<name>/SKILL.md via read_file on
            # demand; buffer.unload_item_reads evicts stale reads at
            # settle time.)
            if a.context_limit:
                tokens = estimate_full_request_tokens(
                    self._buffer.view(), self._system_prompt,
                    tools=TOOLS, chars_per_token=a.chars_per_token)
                threshold = int(a.context_limit * a.compression_threshold)
                if tokens > threshold:
                    if self.verbose:
                        logger.info(f"[AgentLoop] Auto-compact "
                              f"({tokens} > {threshold}) …")
                    compacted = False
                    try:
                        compacted = await self._buffer.auto_compact(
                            self._llm, self.task_dir,
                            config=self.config, tools=TOOLS,
                            feedback=self._feedback,
                            last_diagnosis=self._feedback.last_diagnosis,
                            keep_recent_rounds=a.compact_keep_recent_rounds,
                            best_metric_str=self._best_metric_str())
                    except Exception as e:
                        if self.verbose:
                            logger.warning(f"[AgentLoop] Compact failed ({e}), "
                                  f"force rebuild")
                        self._force_rebuild()
                        compacted = True
                    if compacted:
                        # Actually compacted — post-check + reset
                        # (on_buffer_rebuilt is called automatically by
                        # auto_compact / force_rebuild)
                        post = estimate_full_request_tokens(
                            self._buffer.view(), self._system_prompt,
                            tools=TOOLS, chars_per_token=a.chars_per_token)
                        if post > int(a.context_limit * a.compact_post_check_ratio):
                            if self.verbose:
                                logger.info(f"[AgentLoop] Still high ({post}), "
                                      f"force rebuild")
                            self._force_rebuild()
                        _prev_msg_count = 0

            # Auto-inject backing_skill content for the active item.
            # Dedup is per (plan_version, item_id, skill_name) so this
            # is idempotent within a single item's lifecycle. After
            # auto_compact / force_rebuild, ``on_buffer_rebuilt``
            # rescans surviving messages: if the inject marker is
            # still in the carried-forward recent rounds, the dedup
            # set is repopulated and this call is a no-op; if the
            # rebuild dropped the marker, the dedup set stays empty
            # and we re-inject onto the fresh buffer.
            _active = self._feedback.get_active_item()
            if _active and _active.get("backing_skill"):
                _pool = getattr(self._feedback, "skill_pool", None)
                _bs = _active["backing_skill"]
                _content = _pool.get_skill_content(_bs) if _pool else ""
                if _content:
                    self._buffer.inject_backing_skill(
                        _active["id"], _bs, _content,
                        plan_version=self._feedback.plan_version,
                        max_chars=getattr(
                            a, "skill_inject_max_chars", 6_000,
                        ),
                    )

            # Call LLM
            try:
                response = await self._llm.call(
                    self._system_prompt, self._buffer.view(), tools=TOOLS)
                self._counters.record_compact_success()
            except Exception as e:
                if _is_prompt_too_long(e):
                    if self._counters.compact_failures >= a.compact_max_failures:
                        if self.verbose:
                            logger.warning(f"[AgentLoop] ABORT: "
                                  f"{a.compact_max_failures} compact failures")
                        break
                    self._counters.record_compact_failure()
                    if self.verbose:
                        logger.info(f"[AgentLoop] PTL — level "
                              f"{self._counters.compact_failures}/"
                              f"{a.compact_max_failures}")
                    if self._counters.compact_failures == 1:
                        try:
                            compacted = await self._buffer.auto_compact(
                                self._llm, self.task_dir,
                                config=self.config, tools=TOOLS,
                                feedback=self._feedback,
                                last_diagnosis=self._feedback.last_diagnosis,
                                keep_recent_rounds=a.compact_emergency_keep_rounds,
                                best_metric_str=self._best_metric_str())
                            if not compacted:
                                # No-op signal: nothing to compress, fall back
                                self._force_rebuild()
                        except Exception:
                            self._force_rebuild()
                    else:
                        self._force_rebuild()
                    _prev_msg_count = 0
                    continue
                if self.verbose:
                    logger.warning(f"[AgentLoop] LLM error: {e}")
                self._save_session()
                break

            self._counters.record_api_call()
            self._llm.append_assistant(self._buffer, response)

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
                        logger.info(f"  [LLM] {text[:200]}")

                self._counters.record_no_tool_use()
                if self._feedback.phase == "active":
                    self._counters.record_turn_with_no_edits()

                nudge = self._feedback.build_phase_nudge()
                if nudge is None:
                    clean_exit = True
                    break
                # In replanning with no budget or too many idle turns, exit
                if (self._feedback.phase == "replanning"
                        and (self.max_rounds - self._counters.eval_calls_made <= 0
                             or self._counters.consecutive_no_tool_turns
                                 > a.replanning_max_idle_turns)):
                    clean_exit = True
                    break

                self._buffer.append({"role": "user", "content": nudge})
                if self.verbose:
                    logger.info(f"  [nudge] {nudge}")
                continue

            self._counters.record_tool_use()

            # Delegate turn execution to TurnExecutor.
            # TurnExecutor mutates self._counters in place via record_*
            # methods, so there's no counter state to copy back from
            # TurnResult after this returns.
            try:
                turn_result = await self._turn.execute(
                    tool_calls=tool_calls,
                    buffer=self._buffer,
                    counters=self._counters,
                    max_rounds=self.max_rounds,
                    baseline_commit=self._baseline_commit,
                )
            except Exception as e:
                import traceback as _tb
                if self.verbose:
                    logger.warning(f"[AgentLoop] Turn crashed: {e}\n{_tb.format_exc()}")
                # Rollback editable files to prevent dirty state from
                # leaking into the next round.
                self.runner.file_state.rollback_to_head()
                self._counters.record_edit_attempt(ok=False)
                self._buffer.append({
                    "role": "user",
                    "content": f"[System] Turn crashed: {e}. Files rolled back. Continuing.",
                })
                self._save_session()
                _save_messages_to_disk()
                continue

            # If manual compact replaced messages, reset message cursor
            if self._turn.compacted_this_turn:
                _prev_msg_count = 0

            self._save_session()
            _save_messages_to_disk()

            if turn_result.has_finish:
                clean_exit = True
                break

            # Post-eval action: pick at most ONE thing to do, with strict
            # priority. See PostEvalDecision at the top of this file for
            # the full priority semantics.
            decision = await self._decide_post_eval_action(turn_result)
            await self._execute_post_eval_action(decision, turn_result)

            if (self._counters.consecutive_failures
                    >= self.config.agent.max_consecutive_failures):
                if self.verbose:
                    logger.warning(f"[AgentLoop] ABORT: {self._counters.consecutive_failures} "
                          f"consecutive failures")
                clean_exit = True
                break

        # -- Done --
        if (self._counters.eval_calls_made >= self.max_rounds
                or self._counters.total_api_calls >= max_turns):
            clean_exit = True

        # Run-end summary: always printed regardless of verbose flag.
        # These are observability-critical and must reach agent.log even
        # in quiet mode.
        logger.info(
            f"\n[AgentLoop] Done. eval={self._counters.eval_calls_made}, "
            f"api_calls={self._counters.total_api_calls}",
        )
        best = self.runner.best_result
        ref = self.runner.ref_latency
        if best and ref and ref > 0:
            best_lat = best.metrics.get("latency_us")
            if isinstance(best_lat, (int, float)) and best_lat > 0:
                logger.info(
                    f"[AgentLoop] Best: {best_lat:.2f} us | "
                    f"Ref: {ref:.2f} us | "
                    f"Speedup: {ref / best_lat:.2f}x",
                )
        skill_status = self._feedback.skill_builder.format_short_status()
        if skill_status:
            logger.info(f"[skill-state] final: {skill_status}")
        else:
            logger.info("[skill-state] final: (no skills registered)")
        if self.verbose:
            logger.info(f"\n{self.runner.status()}")

        if clean_exit:
            self._finalize_clean()
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
            "total_api_calls": self._counters.total_api_calls,
            "eval_rounds": self._counters.eval_calls_made,
            "best_metrics": best_metrics or None,
            "final_status": self.runner.status(),
        }
