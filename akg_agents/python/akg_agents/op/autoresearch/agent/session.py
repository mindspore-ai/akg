"""
SessionStore — Persistence layer for agent state.

Manages:
  - Session save/load for --resume
  - Heartbeat file (PID lock + status)
  - Turn-level JSONL logging
  - Editable file snapshot/restore for atomic rollback
  - Plan archival

File logging (stdout tee) is handled by FileLogger (agent/file_logger.py).
"""

import json
import logging
import os
import shutil
import time
from typing import Optional

from ..framework.runner import git_current_commit, git_dirty_files

logger = logging.getLogger(__name__)


class SessionStore:
    """Manages agent session persistence, heartbeat, and file snapshots."""

    def __init__(self, task_dir: str, config, verbose: bool = True):
        self.task_dir = task_dir
        self.config = config
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
            "version": 2,
            "task_name": self.config.name,
            "model": state.get("model", ""),
            "eval_calls_made": state.get("eval_calls_made", 0),
            "total_api_calls": state.get("total_api_calls", 0),
            "consecutive_failures": state.get("consecutive_failures", 0),
            "consecutive_no_edit_turns": state.get("consecutive_no_edit_turns", 0),
            "baseline_commit": state.get("baseline_commit"),
            "head_commit": git_current_commit(self.task_dir),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if "plan_state" in state:
            session_data["plan_state"] = state["plan_state"]
        if state.get("last_diagnosis"):
            session_data["last_diagnosis"] = state["last_diagnosis"]
        _atomic_write(self._session_path("session.json"), session_data)

        plan = state.get("plan")
        plan_path = self._session_path("plan.md")
        if plan:
            try:
                with open(plan_path, "w", encoding="utf-8") as f:
                    f.write(plan)
            except Exception:
                pass
        else:
            # Remove stale plan.md so load() won't revive a cleared plan
            if os.path.exists(plan_path):
                try:
                    os.remove(plan_path)
                except Exception:
                    pass

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
                current_head = git_current_commit(self.task_dir)
                if current_head is None or saved_head != current_head:
                    logger.warning("HEAD mismatch — ignoring session")
                    return None

            semantic_files = list(self.config.editable_files)
            for f in [self.config.eval_script, self.config.smoke_test_script,
                       self.config.program_file, self.config.ref_file, "task.yaml"]:
                if f and f not in semantic_files:
                    semantic_files.append(f)
            dirty = git_dirty_files(self.task_dir, semantic_files)
            if dirty is None or dirty:
                logger.warning("Dirty files — ignoring session")
                return None

            state = {
                "eval_calls_made": session.get("eval_calls_made", 0),
                "consecutive_failures": session.get("consecutive_failures", 0),
                "consecutive_no_edit_turns": session.get("consecutive_no_edit_turns", 0),
                "baseline_commit": session.get("baseline_commit"),
                "total_api_calls": session.get("total_api_calls", 0),
                "last_diagnosis": session.get("last_diagnosis"),
            }

            # plan_state is the sole source of truth when present
            if "plan_state" in session:
                state["plan_state"] = session["plan_state"]
            else:
                # Legacy fallback: old session only has plan.md
                plan_path = self._session_path("plan.md")
                if os.path.exists(plan_path):
                    with open(plan_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        state["plan"] = content

            if self.verbose:
                print(f"[AgentLoop] Resumed: eval={state['eval_calls_made']}, "
                      f"turns={state['total_api_calls']}", flush=True)
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

    def update_heartbeat(self, total_api_calls: int, eval_calls_made: int,
                         max_rounds: int, model: str, best_str: str = "",
                         extra: str = ""):
        lines = [
            f"pid:         {os.getpid()}",
            f"task:        {self.config.name}",
            f"model:       {model}",
            f"eval_rounds: {eval_calls_made}/{max_rounds}",
            f"api_calls:   {total_api_calls}",
            f"best:        {best_str or 'N/A'}",
            f"updated_at:  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        if extra:
            lines.append(f"last_action: {extra}")
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

    # -- File snapshots ----------------------------------------------------

    def snapshot_editable_files(self) -> dict:
        """Snapshot all editable files before edits."""
        snapshots = {}
        for fname in self.config.editable_files:
            fpath = os.path.join(self.task_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    snapshots[fname] = f.read()
        return snapshots

    def restore_snapshots(self, snapshots: dict):
        """Restore editable files to pre-turn state."""
        for fname in self.config.editable_files:
            fpath = os.path.join(self.task_dir, fname)
            if fname in snapshots:
                try:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(snapshots[fname])
                except Exception as e:
                    logger.warning(f"Failed to restore {fname}: {e}")
            else:
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        logger.warning(f"Failed to remove {fname}: {e}")
