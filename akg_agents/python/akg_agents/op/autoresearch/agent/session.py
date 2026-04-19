"""
SessionStore — Persistence layer for agent state.

Manages:
  - Session save/load for --resume
  - Heartbeat file (PID lock + status)
  - Turn-level JSONL logging
  - Plan archival

Git operations (head check, dirty-file detection) are delegated to a
``GitRepo`` injected via the constructor — SessionStore no longer
imports git helpers directly.

Editable-file snapshot/restore moved out of this class entirely as
part of the P5 unification: it now lives on
``ExperimentRunner.file_state`` (a ``FileStateManager``), so callers
get one rollback owner instead of "snapshot here, git rollback there".

File logging (stdout tee) is handled by FileLogger (agent/file_logger.py).
"""

import json
import logging
import os
import shutil
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..framework.git_repo import GitRepo

logger = logging.getLogger(__name__)


class SessionStore:
    """Manages agent session persistence and heartbeat.

    The constructor takes a ``GitRepo`` instance for the head/dirty
    checks performed during save/load. Tests can inject a stub
    ``GitRepo`` (no monkey-patching of imports needed).
    """

    def __init__(self, task_dir: str, config, git: "GitRepo",
                 verbose: bool = True):
        self.task_dir = task_dir
        self.config = config
        self.git = git
        self.verbose = verbose

    # -- Session directory --------------------------------------------------

    @property
    def session_dir(self) -> str:
        path = os.path.join(self.task_dir, self.config.agent.session_dir)
        os.makedirs(path, exist_ok=True)
        return path

    def _session_path(self, filename: str) -> str:
        return os.path.join(self.session_dir, filename)

    # -- Session save/load --------------------------------------------------

    def save(self, state: dict):
        """Persist session state for --resume."""
        def _atomic_write(path, data):
            tmp = path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)
            except Exception as e:
                logger.warning(f"Failed to write {path}: {e}")

        session_data = {
            "version": 3,
            "task_name": self.config.name,
            "model": state.get("model", ""),
            # P3: all counters live under a single "counters" key. Old
            # sessions had counters as top-level fields; RunCounters.from_dict
            # handles both formats on load.
            "counters": state.get("counters") or {},
            "baseline_commit": state.get("baseline_commit"),
            "head_commit": self.git.current_commit(),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if "plan_state" in state:
            session_data["plan_state"] = state["plan_state"]
        # SkillBuilder registry — persist when non-empty so --resume
        # can restore the per-skill applied_versions and
        # unbound_at_versions badges (there are no terminal states;
        # the badges drive binding priority via SkillRecord.tier()).
        if state.get("skill_state"):
            session_data["skill_state"] = state["skill_state"]
        if state.get("last_diagnosis"):
            session_data["last_diagnosis"] = state["last_diagnosis"]
        _atomic_write(self._session_path("session.json"), session_data)

    def load(self) -> Optional[dict]:
        """Restore session. Returns state dict on success, None on failure."""
        session_path = self._session_path("session.json")
        if not os.path.exists(session_path):
            return None
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                session = json.load(f)

            if session.get("task_name") != self.config.name:
                logger.warning("Session task mismatch — ignoring")
                return None

            saved_head = session.get("head_commit")
            if saved_head:
                current_head = self.git.current_commit()
                if current_head is None or saved_head != current_head:
                    logger.warning("HEAD mismatch — ignoring session")
                    return None

            semantic_files = list(self.config.editable_files)
            for f in [self.config.eval_script, self.config.smoke_test_script,
                       self.config.program_file, self.config.ref_file, "task.yaml"]:
                if f and f not in semantic_files:
                    semantic_files.append(f)
            dirty = self.git.dirty_files(semantic_files)
            if dirty is None or dirty:
                logger.warning("Dirty files — ignoring session")
                return None

            state = {
                "baseline_commit": session.get("baseline_commit"),
                "last_diagnosis": session.get("last_diagnosis"),
            }
            # P3: counters live under "counters" in v3+. The full session
            # dict is also passed through so RunCounters.from_dict can fall
            # back to legacy top-level fields for v2 sessions.
            if "counters" in session:
                state["counters"] = session["counters"]
            else:
                # Legacy v2 schema — pass the whole session dict; the
                # caller (RunCounters.from_dict) filters for known fields.
                state["counters"] = session

            # SkillBuilder state — empty dict means "no skills tracked"
            # which SkillBuilder.skill_state_from_dict treats as a no-op,
            # so it's safe to default the missing key to {}.
            state["skill_state"] = session.get("skill_state", {})

            # plan_state is the sole source of truth when present
            if "plan_state" in session:
                state["plan_state"] = session["plan_state"]
            else:
                # Legacy fallback: read the single plan.md that
                # FeedbackBuilder._persist_plan writes to task_dir.
                plan_path = os.path.join(self.task_dir, "plan.md")
                if os.path.exists(plan_path):
                    with open(plan_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        state["plan"] = content

            if self.verbose:
                ctr = state["counters"]
                print(f"[AgentLoop] Resumed: eval={ctr.get('eval_calls_made', 0)}, "
                      f"turns={ctr.get('total_api_calls', 0)}", flush=True)
            return state

        except Exception as e:
            logger.warning(f"Failed to load session: {e}")
            return None

    def cleanup(self):
        """Delete session dir after clean exit."""
        session_dir = os.path.join(self.task_dir, self.config.agent.session_dir)
        if os.path.isdir(session_dir):
            try:
                shutil.rmtree(session_dir)
            except Exception as e:
                logger.warning(f"Failed to remove session: {e}")

    # -- Heartbeat ----------------------------------------------------------

    def update_heartbeat(self, counters, *,
                         max_rounds: int, model: str, best_str: str = "",
                         extra: str = "", phase: str = "",
                         context_tokens: int = 0, context_limit: int = 0,
                         elapsed_sec: float = 0):
        c = counters
        lines = [
            "── run ──",
            f"pid:           {os.getpid()}",
            f"task:          {self.config.name}",
            f"model:         {model}",
            f"phase:         {phase or 'unknown'}",
            "",
            "── progress ──",
            f"eval_rounds:   {c.eval_calls_made}/{max_rounds}",
            f"api_calls:     {c.total_api_calls}",
            f"total_keeps:   {c.total_keeps}",
            f"best:          {best_str or 'N/A'}",
            "",
            "── health ──",
            f"consec_fail:   {c.consecutive_failures}",
            f"no_improve:    {c.consecutive_no_improvement}",
            f"no_edit_turns: {c.consecutive_no_edit_turns}",
            f"no_tool_turns: {c.consecutive_no_tool_turns}",
            f"compact_fail:  {c.compact_failures}",
        ]
        if context_limit:
            pct = int(100 * context_tokens / context_limit) if context_tokens else 0
            lines.append(f"context:       {context_tokens}/{context_limit} ({pct}%)")
        if elapsed_sec > 0:
            m, s = divmod(int(elapsed_sec), 60)
            lines.append("")
            lines.append(f"elapsed:       {m}m{s:02d}s")
        lines.append(f"updated_at:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if extra:
            lines.append(f"last_action:   {extra}")
        hb_path = os.path.join(self.task_dir, self.config.agent.heartbeat_file)
        try:
            with open(hb_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass

    def remove_heartbeat(self):
        hb_path = os.path.join(self.task_dir, self.config.agent.heartbeat_file)
        if os.path.exists(hb_path):
            try:
                os.remove(hb_path)
            except Exception:
                pass

    def check_lock(self):
        """Abort if another agent process is already running on this task."""
        hb_path = os.path.join(self.task_dir, self.config.agent.heartbeat_file)
        if not os.path.exists(hb_path):
            return
        try:
            with open(hb_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("pid:"):
                        other_pid = int(line.split(":", 1)[1].strip())
                        if other_pid == os.getpid():
                            return
                        try:
                            os.kill(other_pid, 0)
                        except OSError:
                            return
                        raise RuntimeError(
                            f"Another agent (PID {other_pid}) is already running "
                            f"on this task. Kill it first or wait for it to finish."
                        )
        except (ValueError, IOError):
            pass

    # -- Turn-level logging ------------------------------------------------

    def log_turn(self, turn_num: int, tool_calls: list, tool_results: list,
                 outcome: str, eval_calls: int, detail: dict = None):
        arg_trunc = self.config.agent.log_arg_truncate
        res_trunc = self.config.agent.log_result_truncate
        entry = {
            "turn": turn_num,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "eval_calls": eval_calls,
            "tools": [
                {"tool": tc["tool_name"],
                 "args": {k: (v[:arg_trunc] if isinstance(v, str) and len(v) > arg_trunc else v)
                          for k, v in tc["arguments"].items()}}
                for tc in tool_calls
            ],
            "results": [r[:res_trunc] if isinstance(r, str) and len(r) > res_trunc else r
                        for r in tool_results],
            "outcome": outcome,
        }
        if detail:
            entry["detail"] = detail
        log_path = self._session_path("log.jsonl")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

