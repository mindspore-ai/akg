# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sync helpers over ``akg_agents.op.autoresearch.framework.git_repo.GitRepo``.

Workspace callers (``scaffold``, ``workflow.round``, ``workflow.baseline``,
``phase_machine``, ``resume``, ``hooks.post_bash``) use these as 1-liner
``commit_in_task(task_dir, paths, msg)`` style calls. The body just wraps
``GitRepo(task_dir)``.
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel
from __future__ import annotations

import os
import subprocess
import sys
from typing import Iterable, List, Optional, Tuple

from akg_agents.op.autoresearch.framework.git_repo import GitRepo


def ensure_git_identity(task_dir: str) -> None:
    """Idempotent local user.name/user.email — closes 'undefined identity'
    on fresh CI boxes. akg's GitRepo doesn't set identity on its own."""
    subprocess.run(["git", "config", "user.name", "autoresearch"],
                   cwd=task_dir, capture_output=True, check=False)
    subprocess.run(["git", "config", "user.email", "auto@research"],
                   cwd=task_dir, capture_output=True, check=False)


def commit_in_task(task_dir: str, paths: Iterable[str],
                   message: str) -> Tuple[bool, str]:
    """Stage `paths` under `task_dir` and commit. Returns
    ``(True, "<short hash>")``, ``(True, "noop")``, or ``(False, "<reason>")``.
    """
    try:
        ensure_git_identity(task_dir)
        repo = GitRepo(task_dir)
        result = repo.commit(message=message, files=list(paths))
    except Exception as e:
        return False, f"unexpected error: {e}"
    if getattr(result, "error", None):
        return False, result.error
    if getattr(result, "nothing_to_commit", False):
        return True, "noop"
    return True, getattr(result, "hash", None) or "ok"


def is_working_tree_clean(task_dir: str) -> bool:
    """True iff git tree is clean. Errors → False (better to leave a stale
    marker than falsely declare clean)."""
    try:
        repo = GitRepo(task_dir)
        return not bool(repo.dirty_files())
    except Exception:
        return False


def current_head_short(task_dir: str) -> Optional[str]:
    """Short HEAD hash, or None if rev-parse fails."""
    try:
        repo = GitRepo(task_dir)
        return repo.current_commit()
    except Exception:
        return None


def auto_rollback(task_dir: str) -> None:
    """Revert editable_files to HEAD. Silent on git failures — rollback
    is a recovery path."""
    try:
        _scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from task_config import load_task_config
        config = load_task_config(task_dir)
        if config is None:
            return
        repo = GitRepo(task_dir)
        repo_root = repo.repo_root
        rel_paths: List[str] = []
        for f in config.editable_files:
            rel_paths.append(os.path.relpath(os.path.join(task_dir, f),
                                             repo_root))
        repo.rollback_files(rel_paths)
    except Exception as e:
        print(f"[AR] Rollback failed: {e}", file=sys.stderr)
