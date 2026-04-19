# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
Tests for GitRepo and FileStateManager (P5/P8 refactor).

These exercise the real git CLI against a tmp_path repo. The fixture
sets local-only user.name / user.email so the test doesn't depend on
the developer's global git config and never writes there.

Skipped on hosts without git on PATH (resolved at module import time).
"""

import inspect
import os
import shutil
import subprocess

import pytest

from akg_agents.op.autoresearch.framework.file_state import FileStateManager
from akg_agents.op.autoresearch.framework.git_repo import GitRepo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git not on PATH — GitRepo tests require a real git CLI",
)


def _run_git(cwd, *args, check=True):
    """Run a git command in cwd, raising on failure if check=True."""
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result


@pytest.fixture
def fresh_repo(tmp_path):
    """Initialize a real git repo in tmp_path with one initial commit.

    Yields the absolute path to the repo. Local user.name/user.email
    are set so commits don't depend on global git config.
    """
    repo = str(tmp_path)
    _run_git(repo, "init", "-q", "-b", "main")
    _run_git(repo, "config", "user.email", "test@autoresearch")
    _run_git(repo, "config", "user.name", "Autoresearch Test")
    # Initial commit so HEAD exists
    initial = os.path.join(repo, "README.md")
    with open(initial, "w") as f:
        f.write("# test repo\n")
    _run_git(repo, "add", "README.md")
    _run_git(repo, "commit", "-q", "-m", "initial")
    return repo


@pytest.fixture
def git_repo(fresh_repo):
    """A GitRepo bound to fresh_repo."""
    return GitRepo(fresh_repo)


def _write(repo, name, content):
    fpath = os.path.join(repo, name)
    os.makedirs(os.path.dirname(fpath) or repo, exist_ok=True)
    with open(fpath, "w") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# GitRepo — read operations
# ---------------------------------------------------------------------------


class TestGitRepoReadOps:

    def test_repo_root_resolves_to_init_dir(self, git_repo, fresh_repo):
        # Cached lazily, so first access populates it
        assert os.path.realpath(git_repo.repo_root) == os.path.realpath(fresh_repo)

    def test_repo_root_cached_after_first_access(self, git_repo, monkeypatch):
        """repo_root is resolved once and cached. Subsequent accesses
        must NOT call git rev-parse again."""
        first = git_repo.repo_root  # warms the cache

        called = [0]
        original = subprocess.run
        def counting_run(*args, **kwargs):
            if args and isinstance(args[0], list) and "rev-parse" in args[0]:
                called[0] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", counting_run)
        second = git_repo.repo_root
        assert second == first
        assert called[0] == 0, "repo_root should be cached, not re-resolved"

    def test_current_commit_returns_short_hash(self, git_repo):
        h = git_repo.current_commit()
        assert h is not None
        # Short hash is 7+ hex chars by default
        assert len(h) >= 7
        assert all(c in "0123456789abcdef" for c in h)

    def test_current_branch_returns_main(self, git_repo):
        assert git_repo.current_branch() == "main"

    def test_dirty_files_clean_repo_returns_empty(self, fresh_repo, git_repo):
        _write(fresh_repo, "kernel.py", "x = 1\n")
        _run_git(fresh_repo, "add", "kernel.py")
        _run_git(fresh_repo, "commit", "-q", "-m", "add kernel")
        # No outstanding changes after commit
        assert git_repo.dirty_files(["kernel.py"]) == []

    def test_dirty_files_detects_modified(self, fresh_repo, git_repo):
        _write(fresh_repo, "kernel.py", "x = 1\n")
        _run_git(fresh_repo, "add", "kernel.py")
        _run_git(fresh_repo, "commit", "-q", "-m", "add kernel")
        # Now modify it
        _write(fresh_repo, "kernel.py", "x = 2\n")
        assert git_repo.dirty_files(["kernel.py"]) == ["kernel.py"]

    def test_dirty_files_detects_untracked(self, fresh_repo, git_repo):
        _write(fresh_repo, "new.py", "print('hi')\n")
        # Never added to git
        assert git_repo.dirty_files(["new.py"]) == ["new.py"]

    def test_diff_returns_changes_between_commits(self, fresh_repo, git_repo):
        _write(fresh_repo, "kernel.py", "x = 1\n")
        _run_git(fresh_repo, "add", "kernel.py")
        _run_git(fresh_repo, "commit", "-q", "-m", "add kernel")
        base = git_repo.current_commit()

        _write(fresh_repo, "kernel.py", "x = 999\n")
        _run_git(fresh_repo, "add", "kernel.py")
        _run_git(fresh_repo, "commit", "-q", "-m", "modify kernel")

        diff = git_repo.diff(base, paths=["kernel.py"])
        assert diff is not None
        assert "x = 1" in diff
        assert "x = 999" in diff


# ---------------------------------------------------------------------------
# GitRepo — write operations
# ---------------------------------------------------------------------------


class TestGitRepoWriteOps:

    def test_commit_creates_new_commit(self, fresh_repo, git_repo):
        _write(fresh_repo, "kernel.py", "x = 1\n")
        result = git_repo.commit("test commit", files=["kernel.py"])
        assert result.committed
        assert result.hash
        # File is now committed
        assert git_repo.dirty_files(["kernel.py"]) == []

    def test_commit_nothing_to_commit_when_no_changes(self, git_repo):
        result = git_repo.commit("noop", files=["README.md"])
        assert result.nothing_to_commit
        assert not result.committed

    def test_commit_branch_guard_rejects_wrong_branch(self, fresh_repo, git_repo):
        _write(fresh_repo, "kernel.py", "x = 1\n")
        # We're on main; ask for "other" → should error, not commit
        result = git_repo.commit(
            "test", files=["kernel.py"], expected_branch="other",
        )
        assert result.error
        assert "Branch mismatch" in result.error
        # File still uncommitted
        assert git_repo.dirty_files(["kernel.py"]) == ["kernel.py"]

    def test_rollback_files_restores_modified_to_head(self, fresh_repo, git_repo):
        _write(fresh_repo, "kernel.py", "x = 1\n")
        _run_git(fresh_repo, "add", "kernel.py")
        _run_git(fresh_repo, "commit", "-q", "-m", "add kernel")

        _write(fresh_repo, "kernel.py", "x = 999\n")
        git_repo.rollback_files(["kernel.py"])

        with open(os.path.join(fresh_repo, "kernel.py")) as f:
            assert f.read() == "x = 1\n"

    def test_rollback_files_removes_untracked(self, fresh_repo, git_repo):
        _write(fresh_repo, "scratch.py", "tmp = 1\n")
        # Untracked
        assert os.path.exists(os.path.join(fresh_repo, "scratch.py"))
        git_repo.rollback_files(["scratch.py"])
        # Removed
        assert not os.path.exists(os.path.join(fresh_repo, "scratch.py"))


# ---------------------------------------------------------------------------
# FileStateManager — snapshot/restore (pure file I/O, no git needed)
# ---------------------------------------------------------------------------


class _NullGit:
    """A no-op git stub — snapshot/restore tests don't touch git."""

    def rollback_files(self, files):
        pass


class TestFileStateManagerSnapshot:

    def test_snapshot_then_restore_preserves_content(self, tmp_path):
        _write(str(tmp_path), "kernel.py", "original\n")
        fs = FileStateManager(
            str(tmp_path), ["kernel.py"], git=_NullGit(),
        )
        snap = fs.snapshot()
        # Mutate disk
        _write(str(tmp_path), "kernel.py", "mutated\n")
        # Restore
        fs.restore(snap)
        with open(os.path.join(str(tmp_path), "kernel.py")) as f:
            assert f.read() == "original\n"

    def test_restore_removes_files_created_after_snapshot(self, tmp_path):
        """Files in editable_files but missing from the snapshot are
        removed on restore — they were created during the failed turn
        and shouldn't survive the rollback."""
        # Initially: only kernel.py exists
        _write(str(tmp_path), "kernel.py", "x\n")
        fs = FileStateManager(
            str(tmp_path), ["kernel.py", "extra.py"], git=_NullGit(),
        )
        snap = fs.snapshot()  # extra.py NOT in snap (didn't exist)

        # During the "turn" the agent creates extra.py
        _write(str(tmp_path), "extra.py", "tmp\n")
        assert os.path.exists(os.path.join(str(tmp_path), "extra.py"))

        # Restore should remove it
        fs.restore(snap)
        assert not os.path.exists(os.path.join(str(tmp_path), "extra.py"))
        # And kernel.py is unchanged
        with open(os.path.join(str(tmp_path), "kernel.py")) as f:
            assert f.read() == "x\n"

    def test_snapshot_skips_nonexistent_files(self, tmp_path):
        """Files in editable_files that don't exist on disk are
        excluded from the snapshot dict."""
        _write(str(tmp_path), "kernel.py", "x\n")
        fs = FileStateManager(
            str(tmp_path), ["kernel.py", "missing.py"], git=_NullGit(),
        )
        snap = fs.snapshot()
        assert "kernel.py" in snap
        assert "missing.py" not in snap


# ---------------------------------------------------------------------------
# FileStateManager — rollback_to_head (delegates to GitRepo.rollback_files)
# ---------------------------------------------------------------------------


class TestFileStateManagerRollback:

    def test_rollback_to_head_delegates_to_git(self, fresh_repo):
        """rollback_to_head should call git.rollback_files with the
        editable_files list, no extra logic."""
        _write(fresh_repo, "kernel.py", "x = 1\n")
        _run_git(fresh_repo, "add", "kernel.py")
        _run_git(fresh_repo, "commit", "-q", "-m", "init kernel")

        git = GitRepo(fresh_repo)
        fs = FileStateManager(fresh_repo, ["kernel.py"], git=git)

        # Mutate, then rollback
        _write(fresh_repo, "kernel.py", "x = 999\n")
        fs.rollback_to_head()

        with open(os.path.join(fresh_repo, "kernel.py")) as f:
            assert f.read() == "x = 1\n"


# ---------------------------------------------------------------------------
# Workflow timeout salvage path — must use the runner's GitRepo,
# never construct a fresh one inline
# ---------------------------------------------------------------------------


class TestFinalizeTimeoutGitOwnership:
    """Pin that AutoresearchWorkflow.finalize_on_timeout routes git
    access through an ExperimentRunner-owned GitRepo, not a fresh
    inline ``GitRepo(task_dir)`` construction.

    This is the only place outside ExperimentRunner that historically
    constructed its own git owner. If commit policy, branch guards,
    or extra-files conventions evolve in the live path, the salvage
    path used to silently drift; routing through the same runner
    eliminates that drift surface.

    Structural test against finalize_on_timeout source: a behavioral
    test would need to spin up a workflow + tmp git repo + minimal
    task.yaml, which is borderline integration testing for a
    low-severity invariant.
    """

    def test_finalize_uses_runner_git_not_fresh_gitrepo(self):
        from akg_agents.op.workflows.autoresearch_workflow import (
            AutoresearchWorkflow,
        )
        src = inspect.getsource(AutoresearchWorkflow.finalize_on_timeout)

        # Must NOT construct a fresh GitRepo — that bypasses the runner's
        # owner and re-introduces the scattered-git-entry-point smell.
        assert "GitRepo(" not in src, (
            "finalize_on_timeout should route git access through "
            "runner.git (the same owner the live agent loop uses), "
            "not construct a fresh GitRepo(task_dir) inline."
        )

        # Must construct an ExperimentRunner in salvage mode and use
        # its sub-owners (runner.git, runner.logger, runner.config).
        assert "ExperimentRunner(" in src, (
            "finalize_on_timeout should construct an ExperimentRunner "
            "(skip_branch_switch=True) and borrow its git/logger owners."
        )
        assert "skip_branch_switch=True" in src, (
            "finalize_on_timeout's runner construction must pass "
            "skip_branch_switch=True so the salvage path doesn't "
            "create branches or print branch banners."
        )
        assert "runner.git.commit(" in src, (
            "finalize_on_timeout must commit via runner.git.commit, "
            "not via a fresh GitRepo or the deleted module-level "
            "git_commit helper."
        )
