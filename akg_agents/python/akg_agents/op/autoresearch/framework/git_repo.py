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

"""
GitRepo — git operations bound to a single task directory.

Before this module the autoresearch layer talked to git via 8
module-level functions in ``runner.py`` (``git_commit``,
``git_rollback_files``, ``git_current_commit``, ``git_dirty_files``,
``git_diff``, ``git_current_branch``, ``git_ensure_branch``,
``git_cleanup_branch``). Each one re-resolved the repo root via
``git rev-parse --show-toplevel`` on every call, and consumers in
session.py / loop.py / turn.py / autoresearch_workflow.py imported
them à la carte.

GitRepo collects them into one class with a cached ``repo_root``
property and a single ``task_dir`` instance, so callers hold one
handle instead of importing N helpers.
"""


# pylint: disable=broad-exception-caught
import os
import shutil
import subprocess
from typing import Optional

from .config import CommitResult


def _git_repo_root(task_dir: str) -> str:
    """Resolve the git repo root for ``task_dir``. Raises on failure.

    Module-private helper because every GitRepo instance caches its
    own resolved root and never calls this in the hot path — only
    during construction or first access.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, cwd=task_dir, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"Not a git repository: {task_dir}\n{result.stderr}")
    return result.stdout.strip()


def _git_add(repo_root: str, rel_path: str) -> bool:
    """``git add`` a single path. Returns True on success.

    On WSL2 with /mnt/c/ the stat cache can be stale, causing ``git
    add`` to skip genuinely modified files. The pre-add ``git diff``
    forces an index refresh for that path.
    """
    subprocess.run(
        ["git", "diff", "--", rel_path],
        cwd=repo_root, capture_output=True, check=False
    )
    result = subprocess.run(
        ["git", "add", rel_path],
        cwd=repo_root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        print(f"[git] WARNING: git add failed for {rel_path}: {result.stderr.strip()}")
        return False
    return True


class GitRepo:
    """Git operations bound to a single task directory.

    All methods that take a path are interpreted relative to
    ``task_dir`` and resolved against the cached ``repo_root``.
    Construction is lazy — ``repo_root`` is resolved on first access
    and cached for the rest of the instance lifetime.
    """

    def __init__(self, task_dir: str):
        self.task_dir = os.path.abspath(task_dir)
        self._repo_root: Optional[str] = None

    @property
    def repo_root(self) -> str:
        """Cached git repo root for ``self.task_dir``. Resolves lazily."""
        if self._repo_root is None:
            self._repo_root = _git_repo_root(self.task_dir)
        return self._repo_root

    def _rel(self, fname: str) -> str:
        """Convert a task-relative file path to a repo-root-relative path."""
        return os.path.relpath(os.path.join(self.task_dir, fname), self.repo_root)

    # -- Read operations ---------------------------------------------------

    def current_commit(self) -> Optional[str]:
        """Return HEAD short hash, or None on any failure."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=self.repo_root, check=False
            )
            if result.returncode == 0:
                return result.stdout.strip()
            return None
        except Exception:
            return None

    def current_branch(self) -> Optional[str]:
        """Return current branch name, or None on any failure."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, cwd=self.repo_root, check=False
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def diff(self, base_commit: str, head: str = "HEAD",
             paths: Optional[list[str]] = None) -> Optional[str]:
        """Return ``git diff base_commit..head`` output, optionally scoped."""
        try:
            cmd = ["git", "diff", f"{base_commit}..{head}"]
            if paths:
                cmd.append("--")
                for p in paths:
                    cmd.append(self._rel(p))
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.repo_root, check=False
            )
            if result.returncode == 0:
                return result.stdout
            return None
        except Exception:
            return None

    def dirty_files(self, files: list[str]) -> Optional[list[str]]:
        """Return the subset of ``files`` that have uncommitted changes
        or are untracked. Returns None on any unexpected exception.
        """
        try:
            dirty = set()
            for f in files:
                fpath = os.path.join(self.task_dir, f)
                rel_path = self._rel(f)

                result = subprocess.run(
                    ["git", "diff", "HEAD", "--", rel_path],
                    capture_output=True, text=True, cwd=self.repo_root, check=False
                )
                if result.returncode == 0 and result.stdout.strip():
                    dirty.add(f)
                    continue

                if os.path.exists(fpath):
                    result = subprocess.run(
                        ["git", "ls-files", "--others", "--exclude-standard",
                         "--", rel_path],
                        capture_output=True, text=True, cwd=self.repo_root, check=False
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        dirty.add(f)

            return list(dirty)
        except Exception:
            return None

    # -- Write operations --------------------------------------------------

    def commit(self, message: str, files: Optional[list[str]] = None,
               push: bool = False, task_name: Optional[str] = None,
               expected_branch: Optional[str] = None) -> CommitResult:
        """Stage files and create a commit. Returns a CommitResult.

        Three outcomes:
          - committed: hash non-empty, commit succeeded
          - nothing_to_commit: nothing was staged (e.g. baseline / no-op)
          - error: commit command failed

        If ``expected_branch`` is set, refuses to commit when the current
        branch differs (defends against committing on the wrong branch
        after a manual checkout).
        """
        try:
            repo_root = self.repo_root

            if expected_branch:
                current = self.current_branch()
                if current and current != expected_branch:
                    raise RuntimeError(
                        f"Branch mismatch: on '{current}' but expected "
                        f"'{expected_branch}'. Aborting to prevent commits "
                        "on the wrong branch."
                    )

            add_failures = []
            if files:
                for f in files:
                    rel_path = self._rel(f)
                    if not _git_add(repo_root, rel_path):
                        add_failures.append(rel_path)
            else:
                rel_dir = os.path.relpath(self.task_dir, repo_root)
                if not _git_add(repo_root, rel_dir):
                    add_failures.append(rel_dir)

            diff_result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=repo_root, capture_output=True, check=False
            )
            if diff_result.returncode == 0:
                if add_failures:
                    err = f"git add failed for: {', '.join(add_failures)}"
                    print(f"[git] ERROR: {err}")
                    return CommitResult(error=err)
                return CommitResult(nothing_to_commit=True)

            author_name = task_name or "agent"
            git_cmd = [
                "git",
                "-c", f"user.name={author_name}",
                "-c", "user.email=agent@autoresearch",
                "commit", "-m", message,
            ]
            commit_result = subprocess.run(
                git_cmd,
                cwd=repo_root, capture_output=True, text=True, check=False
            )
            if commit_result.returncode != 0:
                err = commit_result.stderr.strip()
                print(f"[git] ERROR: commit failed: {err}")
                return CommitResult(error=err)

            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=repo_root, check=False
            )
            commit_hash = result.stdout.strip()

            if push:
                push_result = subprocess.run(
                    ["git", "push", "-u", "origin", "HEAD"],
                    cwd=repo_root, capture_output=True, text=True, check=False
                )
                if push_result.returncode != 0:
                    print(f"[git] WARNING: push failed: {push_result.stderr.strip()}")
                else:
                    print(f"[git] Pushed {commit_hash} to remote")

            return CommitResult(hash=commit_hash)
        except Exception as e:
            print(f"[git] ERROR: GitRepo.commit exception: {e}")
            return CommitResult(error=str(e))

    def rollback_files(self, files: list[str]) -> None:
        """Roll back ``files`` to HEAD; remove untracked files entirely.

        For each file:
          - tracked: ``git checkout HEAD -- <path>`` restores to last commit
          - untracked: file is deleted from disk
        """
        try:
            repo_root = self.repo_root
            for f in files:
                fpath = os.path.join(self.task_dir, f)
                rel_path = self._rel(f)

                ls_result = subprocess.run(
                    ["git", "ls-files", "--error-unmatch", rel_path],
                    cwd=repo_root, capture_output=True, check=False
                )
                if ls_result.returncode != 0:
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        print(f"[git] Removed untracked file: {rel_path}")
                    continue

                result = subprocess.run(
                    ["git", "checkout", "HEAD", "--", rel_path],
                    cwd=repo_root, capture_output=True, text=True, check=False
                )
                if result.returncode != 0:
                    print(f"[git] WARNING: rollback failed for {rel_path}: {result.stderr.strip()}")
        except Exception as e:
            print(f"[git] ERROR: GitRepo.rollback_files exception: {e}")

    def ensure_branch(self, branch_name: str) -> str:
        """Switch to (or create) the named experiment branch.

        Stale branches from previous runs are deleted and recreated
        from current HEAD, so each experiment starts from a clean
        state. Returns the branch name on success, raises RuntimeError
        on failure.
        """
        repo_root = self.repo_root

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=repo_root, check=False
        )
        current_branch = result.stdout.strip()

        result = subprocess.run(
            ["git", "rev-parse", "--verify", branch_name],
            capture_output=True, text=True, cwd=repo_root, check=False
        )
        branch_exists = result.returncode == 0

        if branch_exists:
            if current_branch == branch_name:
                # On the stale branch — switch away first
                for candidate in ["main", "master"]:
                    check = subprocess.run(
                        ["git", "rev-parse", "--verify", candidate],
                        capture_output=True, cwd=repo_root, check=False
                    )
                    if check.returncode == 0:
                        subprocess.run(
                            ["git", "checkout", candidate],
                            capture_output=True, cwd=repo_root, check=False
                        )
                        break
            subprocess.run(
                ["git", "branch", "-D", branch_name],
                capture_output=True, text=True, cwd=repo_root, check=False
            )
            print(f"[git] Deleted stale branch '{branch_name}'")

        result = subprocess.run(
            ["git", "checkout", "-b", branch_name],
            capture_output=True, text=True, cwd=repo_root, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create branch '{branch_name}': {result.stderr.strip()}"
            )

        print(f"[git] Created and switched to branch '{branch_name}'")
        return branch_name

    def cleanup_branch(self, exp_branch: str, original_branch: str,
                       session_dir: str = "agent_session",
                       heartbeat_file: str = "RUNNING") -> None:
        """Switch back to the original branch and clean experiment artifacts.

        The exp branch is preserved for inspection. Artifacts created
        during the run (logs, plan.md, session dir, heartbeat) are
        removed before checkout to avoid dirty-tree conflicts.
        """
        repo_root = self.repo_root
        rel_dir = os.path.relpath(self.task_dir, repo_root)

        # 1. Remove experiment artifacts FIRST (before checkout)
        experiment_artifacts = [
            "agent.log", "log.jsonl", "perf_log.md",
            "report.md", "plan.md", heartbeat_file,
        ]
        for fname in experiment_artifacts:
            fpath = os.path.join(self.task_dir, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass
        for dname in [session_dir, "__pycache__"]:
            dpath = os.path.join(self.task_dir, dname)
            if os.path.isdir(dpath):
                try:
                    shutil.rmtree(dpath)
                except Exception:
                    pass

        # 2. Discard remaining uncommitted changes so checkout won't fail
        subprocess.run(
            ["git", "checkout", "--", rel_dir],
            capture_output=True, text=True, cwd=repo_root, check=False
        )

        # 3. Switch back to original branch (exp branch preserved)
        current = self.current_branch()
        if current == exp_branch:
            result = subprocess.run(
                ["git", "checkout", original_branch],
                capture_output=True, text=True, cwd=repo_root, check=False
            )
            if result.returncode != 0:
                print(f"[git] WARNING: checkout {original_branch} failed: {result.stderr.strip()}")
                return
            print(f"[git] Switched back to '{original_branch}' (exp branch '{exp_branch}' preserved)")
        else:
            print(f"[git] WARNING: not on exp branch '{exp_branch}' (on '{current}'), skipping checkout")
