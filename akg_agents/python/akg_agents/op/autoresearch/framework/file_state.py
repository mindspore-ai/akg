"""
FileStateManager — single owner of editable-file state restoration.

Before this module the autoresearch layer had two parallel rollback
mechanisms living in different files:

  1. **In-memory snapshots** — ``SessionStore.snapshot_editable_files``
     captured pre-edit file contents before each turn, and
     ``restore_snapshots`` wrote them back on edit_fail / quick_check_fail.
     Lived in ``agent/session.py`` because session.py was the closest
     "system layer" file at the time.

  2. **Git rollback** — ``runner.git_rollback_files`` ran ``git checkout
     HEAD --`` against editable_files. Used by ExperimentRunner after
     a DISCARD/FAIL eval and by AgentLoop's turn-crash safety net.

Both paths land on the same on-disk state in the happy path (HEAD ==
turn-start, because every successful eval is committed and every
failed eval rolls back), so the duplication was historical, not
intentional. FileStateManager unifies them behind one owner with two
explicit named methods so callers pick the right one for their context:

  - ``snapshot()`` / ``restore()`` for the in-turn fast path (no git fork)
  - ``rollback_to_head()`` for the post-eval / crash recovery path

The class also owns the ``editable_files`` list (previously read from
``self.config.editable_files`` at every call site), so callers no
longer need to thread config through just to know which files are in
scope for rollback.
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .git_repo import GitRepo


class FileStateManager:
    """Single owner of editable-file state restoration.

    Two complementary rollback paths, both bound to the same
    ``editable_files`` list and ``task_dir``:

      - ``snapshot()`` / ``restore(snap)``: in-memory rollback to
        "the state when this turn started". Used by TurnExecutor's
        atomic edit-fail and quick-check-fail paths — fast, no fork.
      - ``rollback_to_head()``: git-based rollback to the last
        committed state. Used by ExperimentRunner after DISCARD/FAIL
        evals and by AgentLoop's turn-crash safety net.

    In the happy path both land on the same on-disk state because
    every successful eval is committed and every failed eval rolls
    back, so HEAD == turn-start.
    """

    def __init__(self, task_dir: str, editable_files, git: "GitRepo"):
        self.task_dir = task_dir
        # Defensive copy — caller's list could mutate later (TaskConfig
        # is conventionally immutable but the field is a plain list).
        self.editable_files = list(editable_files)
        self.git = git

    # -- Snapshot path (fast, in-memory) -----------------------------------

    def snapshot(self) -> dict:
        """Capture current contents of editable_files for later restore.

        Returns an opaque dict the caller passes back to ``restore()``.
        Files that don't exist on disk are simply not included — the
        restore path interprets "missing key" as "file should not exist
        after restore" and removes it.
        """
        snapshots = {}
        for fname in self.editable_files:
            fpath = os.path.join(self.task_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    snapshots[fname] = f.read()
        return snapshots

    def restore(self, snapshots: dict) -> None:
        """Restore editable_files to the state captured by ``snapshot()``.

        Files present in the snapshot are written back; files in
        editable_files but missing from the snapshot are removed
        from disk (they were created during the failed turn and
        shouldn't survive the rollback).
        """
        import logging
        log = logging.getLogger(__name__)
        for fname in self.editable_files:
            fpath = os.path.join(self.task_dir, fname)
            if fname in snapshots:
                try:
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write(snapshots[fname])
                except Exception as e:
                    log.warning(f"Failed to restore {fname}: {e}")
            else:
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        log.warning(f"Failed to remove {fname}: {e}")

    # -- Git path (authoritative, slower) ----------------------------------

    def rollback_to_head(self) -> None:
        """Discard all uncommitted changes to editable_files.

        Equivalent to ``git checkout HEAD -- <editable_files>`` plus
        removal of any untracked files. Used after eval DISCARD/FAIL
        (where the runner already committed any KEEP, so HEAD is the
        canonical "last good state") and from AgentLoop's turn-crash
        recovery path.
        """
        self.git.rollback_files(self.editable_files)
