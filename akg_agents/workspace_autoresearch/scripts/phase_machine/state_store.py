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

"""State storage layer.

Single per-task state file at ``<task_dir>/.ar_state/state.json``.
Every piece of "control state" (current phase, ownership, heartbeat,
all progress accounting, pending-settle sentinel) lives in this one
record. Atomic write of state.json IS the transaction commit; there's
no separate marker file. Cross-file consistency with the two durable
artifacts that DON'T live inside state.json (``plan.md`` and
``history.jsonl``) is checked by comparing their current shape against
``state.expected_plan_version`` / ``state.expected_history_round``.

Files this module owns (writes):
  - <task_dir>/.ar_state/state.json   single source of truth (this file)

Files this module reads (artifacts written elsewhere):
  - <task_dir>/.ar_state/history.jsonl   append-only round records
  - <task_dir>/.ar_state/plan.md         agent-facing plan
  - <task_dir>/.ar_state/diagnose_v<N>.md / plan_items.xml / .edit_started
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Union

# state_store is imported by hook code that may run before scripts/ is
# on sys.path (no editable install). Make the import work either way.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_HERE)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from utils.json_io import sanitize_floats  # noqa: E402

from .models import Progress


# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

INIT = "INIT"
BASELINE = "BASELINE"
PLAN = "PLAN"
EDIT = "EDIT"
DIAGNOSE = "DIAGNOSE"
REPLAN = "REPLAN"
FINISH = "FINISH"

ALL_PHASES = {INIT, BASELINE, PLAN, EDIT, DIAGNOSE, REPLAN, FINISH}


# ---------------------------------------------------------------------------
# Filenames inside <task_dir>/.ar_state/
# ---------------------------------------------------------------------------

STATE_FILE = "state.json"
HISTORY_FILE = "history.jsonl"
PLAN_FILE = "plan.md"
PLAN_ITEMS_FILE = "plan_items.xml"  # agent-written XML, validated by create_plan
EDIT_MARKER_FILE = ".edit_started"
# Journal / write-ahead intent for round and baseline transactions.
# Owners write it BEFORE appending bodies, clear it AFTER state.json
# commits. pipeline.py's replay branch reconstructs an in-flight
# transaction from this file. See `write_intent` / `replay_intent`.
INTENT_FILE = "intent.json"

# DIAGNOSE artifact — see CLAUDE.md invariant #10.
DIAGNOSE_ARTIFACT_TEMPLATE = "diagnose_v{}.md"
DIAGNOSE_MARKER_TEMPLATE = "[AR DIAGNOSE COMPLETE marker_v{}]"
# Pulled from config.yaml `defaults.diagnose_max_attempts` at import time
# so the 8 callers using the bare name don't need to change. Edit
# config.yaml to retune; restart the process to pick up.
from utils.settings import diagnose_max_attempts as _diagnose_max_attempts
DIAGNOSE_ATTEMPTS_CAP = _diagnose_max_attempts()


# ---------------------------------------------------------------------------
# Project root + per-op pointer (scaffold -> batch/run.py handoff)
# ---------------------------------------------------------------------------

def _find_project_root() -> str:
    """The autoresearch project root. Derived from this file's fixed
    location: <autoresearch_root>/scripts/phase_machine/state_store.py.
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


_PROJECT_ROOT = _find_project_root()
_TASK_DIR_POINTERS = os.path.join(_PROJECT_ROOT, ".task_dir_pointers")

# Repo-level session→task_dir index. Each session has its own file at
# <repo>/.session_tasks/<sha(session_id)>; its content is the absolute
# path of the task that session is currently driving. This covers tasks
# created with scaffold --output-dir pointing outside the repo root —
# an ar_tasks/-only scan wouldn't see them.
_SESSION_TASKS_DIR = os.path.join(_PROJECT_ROOT, ".session_tasks")


def task_dir_pointer_path(op_name: str) -> str:
    """Per-op pointer file path. scaffold writes the task_dir here
    immediately after creating <repo>/ar_tasks/<op>_<ts>_<rand>;
    batch/run.py reads it instead of mtime-scanning."""
    safe = op_name.replace("/", "_").replace("\\", "_")
    return os.path.join(_TASK_DIR_POINTERS, safe)


def write_task_dir_pointer(op_name: str, task_dir: str) -> None:
    """Atomic write."""
    path = task_dir_pointer_path(op_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(task_dir))
    os.replace(tmp, path)


def read_task_dir_pointer(op_name: str) -> Optional[str]:
    """Returns absolute task_dir, or None when missing / dangling."""
    path = task_dir_pointer_path(op_name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            td = f.read().strip()
    except OSError:
        return None
    return td if td and os.path.isdir(td) else None


# ---------------------------------------------------------------------------
# Path builders for .ar_state/ files
# ---------------------------------------------------------------------------

def state_path(task_dir: str, name: str) -> str:
    """Generic path builder for any file under <task_dir>/.ar_state/."""
    return os.path.join(task_dir, ".ar_state", name)


def state_record_path(task_dir: str) -> str:
    return state_path(task_dir, STATE_FILE)


def plan_path(task_dir: str) -> str:
    return state_path(task_dir, PLAN_FILE)


def history_path(task_dir: str) -> str:
    return state_path(task_dir, HISTORY_FILE)


def edit_marker_path(task_dir: str) -> str:
    return state_path(task_dir, EDIT_MARKER_FILE)


def intent_path(task_dir: str) -> str:
    return state_path(task_dir, INTENT_FILE)


def diagnose_artifact_path(task_dir: str, plan_version: int) -> str:
    return state_path(task_dir, DIAGNOSE_ARTIFACT_TEMPLATE.format(plan_version))


def diagnose_marker(plan_version: int) -> str:
    return DIAGNOSE_MARKER_TEMPLATE.format(plan_version)


# ---------------------------------------------------------------------------
# state.json — load / save / update primitives
# ---------------------------------------------------------------------------
# Schema (every key documented; missing keys at load → default value):
#
#   phase                      str    one of ALL_PHASES; defaults to INIT
#   owner                      dict|None  {session_id, pid, claimed_at};
#                                         None when no Claude session is
#                                         driving the task
#   last_touched               ISO    bumped by touch_heartbeat / save_state
#
#   # Progress accounting (Progress dataclass fields)
#   task, eval_rounds, max_rounds, consecutive_failures,
#   best_metric, best_commit, baseline_metric, baseline_source,
#   baseline_outcome, baseline_error_source, baseline_per_shape_us,
#   baseline_fingerprint, seed_metric, plan_version, next_pid,
#   num_cases, per_shape_descs, diagnose_attempts,
#   diagnose_attempts_for_version, last_diagnose_failure_reason
#
#   # Pending settle (None when no replay needed)
#   pending_settle             dict|None  kd_json from a round whose
#                                         settle hasn't committed yet
#
#   # Cross-file artifact expectations (subsumes the per-file _txn_id
#   # markers that lived in plan.md / history.jsonl / etc.)
#   expected_plan_version      int    plan.md's "# Plan vN" must match
#   expected_history_round     int    history.jsonl last row's "round"
#
# Atomic save_state(td, state) is the transaction commit. Body artifacts
# (plan.md / history.jsonl) are written FIRST, then state.json with the
# updated expected_* fields. A crash before save_state leaves the body
# artifacts ahead of state's expectations — check_state_consistency
# reports it, the recovery path is to re-run the writer (idempotent).


_PROGRESS_FIELD_NAMES = {f.name for f in Progress.__dataclass_fields__.values()}


def load_state(task_dir: str) -> Optional[dict]:
    """Read state.json into a dict, or None when missing / corrupt.
    Single canonical reader."""
    path = state_record_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"[state_store] WARNING: {path} corrupt ({e}); treating "
              f"as missing. Delete the file and re-run to recover.",
              file=sys.stderr)
        return None
    if not isinstance(data, dict):
        return None
    return data


def save_state(task_dir: str, state: dict) -> None:
    """Atomic write of the full state record. Bumps last_touched.
    Callers pass the COMPLETE new state dict — partial updates go
    through update_state below."""
    state = dict(state)
    state["last_touched"] = _now_iso()
    path = state_record_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sanitize_floats(state), f, indent=2)
    os.replace(tmp, path)


def update_state(task_dir: str, **fields) -> dict:
    """Load → merge fields → atomic save. Returns the post-merge state.
    Convenience for single-field updates (touch_heartbeat, owner
    changes); transactional callers that mutate many fields should
    build the dict explicitly and call save_state."""
    state = load_state(task_dir) or _fresh_state()
    state.update(fields)
    save_state(task_dir, state)
    return state


def _fresh_state() -> dict:
    """Default state record for a task that has none yet on disk.
    Two orthogonal semantics live in state.json:
      - control fields (phase, owner, last_touched, expected_*) —
        these are meaningful from the first set_task_dir; defaults here.
      - progress fields (task, eval_rounds, max_rounds, best_metric,
        baseline_*, ...) — meaningful ONLY after baseline.py has
        committed the first real measurement. `progress_initialized`
        is the discriminator. load_progress() returns None when it's
        False, so resume / dashboard / scaffold can't mistake a
        freshly-claimed-but-not-yet-evaluated task for a Round 0/0
        task.
    """
    return {
        "phase": INIT,
        "owner": None,
        "last_touched": _now_iso(),
        "progress_initialized": False,
        "pending_settle": None,
        "expected_plan_version": 0,
        "expected_history_round": 0,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Ownership — embedded in state.owner
# ---------------------------------------------------------------------------
# "Which task is the current Claude session
# driving?" is answered by scanning ar_tasks/ and matching state.json's
# owner.session_id against the env's CLAUDE_CODE_SESSION_ID. Supervisors
# (batch/run.py) have no Claude session and pass expected_task_dir to
# clear_active_task to release a task they themselves spawned.

def _our_session_id() -> str:
    """Caller's Claude Code session id (empty when not inside an agent
    process — supervisors like batch/run.py)."""
    return os.environ.get("CLAUDE_CODE_SESSION_ID", "")


# ---- Session→task index ---------------------------------------------------
# Each Claude session points at exactly one task at a time. The index
# is a single small file per session under <repo>/.session_tasks/. Why
# files instead of a single .json: per-session files mean no inter-
# session write contention, and stale sessions just leave orphan files
# that are trivially gc-able (orphan = the pointed task no longer
# names this session in its state.owner).

def _session_index_path(session_id: str) -> str:
    """Filesystem-safe per-session path. session_id is opaque (Claude
    issues UUIDs); hash to keep filenames stable across OSes."""
    import hashlib
    h = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_SESSION_TASKS_DIR, h)


def _write_session_index(session_id: str, task_dir: str) -> None:
    if not session_id:
        return
    path = _session_index_path(session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(task_dir))
    os.replace(tmp, path)


def _read_session_index(session_id: str) -> Optional[str]:
    if not session_id:
        return None
    path = _session_index_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            td = f.read().strip()
    except OSError:
        return None
    return td if td and os.path.isdir(td) else None


def _clear_session_index(session_id: str) -> None:
    if not session_id:
        return
    try:
        os.remove(_session_index_path(session_id))
    except OSError:
        pass


def _heartbeat_fresh(state: dict) -> bool:
    """True iff state.last_touched is within heartbeat_fresh_seconds.
    Loud-fallback to 180s if settings is unreachable."""
    try:
        from utils.settings import heartbeat_fresh_seconds as _hb
        window = _hb()
    except Exception as e:
        print(f"[state_store] WARNING: heartbeat_fresh_seconds() "
              f"unavailable ({e}); falling back to 180s.", file=sys.stderr)
        window = 180
    age = _age_seconds(state.get("last_touched"))
    return age < window


def _age_seconds(iso_str: Optional[str]) -> float:
    if not iso_str:
        return float("inf")
    try:
        ts = datetime.fromisoformat(iso_str).timestamp()
        return time.time() - ts
    except (ValueError, TypeError):
        return float("inf")


def _iter_task_dirs():
    """Yield absolute paths of ar_tasks/<dir>/ whose .ar_state/state.json
    exists. Order is undefined; caller sorts as needed."""
    root = os.path.join(_PROJECT_ROOT, "ar_tasks")
    if not os.path.isdir(root):
        return
    try:
        names = os.listdir(root)
    except OSError:
        return
    for name in names:
        full = os.path.join(root, name)
        if os.path.isdir(full) and os.path.exists(state_record_path(full)):
            yield full


def current_session_task_dir() -> Optional[str]:
    """Task this Claude session is currently driving, or None.

    Single shared facade for resume / dashboard / find_active_task_dir /
    hooks. Lookup order:

      1. Per-session index (<repo>/.session_tasks/<sha(sid)>).
         O(1). Covers tasks created with scaffold --output-dir pointing
         OUTSIDE <repo>/ar_tasks, which the ar_tasks scan in step 2 can
         never see.
      2. Fallback: scan <repo>/ar_tasks and pick the task whose
         state.owner.session_id == our session. Useful when the index
         file was lost (manual deletion, fresh checkout that inherited
         only the task dir). The matched task gets re-written into the
         index so subsequent lookups are O(1) again.

    Returns None when no session_id is set (supervisor processes like
    batch/run.py have no Claude session) — caller decides what to do
    with that signal."""
    session = _our_session_id()
    if not session:
        return None

    pointed = _read_session_index(session)
    if pointed:
        st = load_state(pointed)
        owner = (st or {}).get("owner") or {}
        if owner.get("session_id") == session:
            return pointed
        # Stale index — clear it before falling through to scan, so
        # the scan's findings aren't overwritten by the stale entry.
        _clear_session_index(session)

    # Scan ar_tasks/ for a state.owner.session_id match — recovers the
    # ownership when the index file is missing (fresh checkout, manual
    # cleanup, etc.). Multiple matches shouldn't happen because
    # set_task_dir enforces single-task-per-session, but if they do
    # pick the most recently touched one and warn loudly.
    matches: list[tuple[float, str]] = []
    for td in _iter_task_dirs():
        st = load_state(td)
        if not st:
            continue
        owner = st.get("owner") or {}
        if owner.get("session_id") == session:
            matches.append((_age_seconds(st.get("last_touched")), td))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    if len(matches) > 1:
        others = ", ".join(td for _, td in matches[1:])
        print(f"[state_store] WARNING: session_id={session} owns "
              f"{len(matches)} tasks (single-task invariant violated). "
              f"Picking most-recently touched: {matches[0][1]}. "
              f"Others: {others}", file=sys.stderr)
    chosen = matches[0][1]
    # Rehydrate the index so the next call is O(1).
    _write_session_index(session, chosen)
    return chosen


def get_task_dir() -> str:
    """Return the task_dir owned by the current Claude session, or ""
    when none is found. Falls back to AR_TASK_DIR env var.

    Thin shim over current_session_task_dir — kept as a separate name
    because hooks call it dozens of times per session and the string
    return convention ("" not None) matches the env-var contract they
    expect."""
    td = current_session_task_dir()
    if td:
        return td
    return os.environ.get("AR_TASK_DIR", "")


def set_task_dir(task_dir: str, *, force: bool = False) -> bool:
    """Claim `task_dir` for the current session. Returns True on
    success, False when refused.

    Single-task-per-session invariant: if the current session already
    owns a DIFFERENT task, that task's state.owner gets cleared first
    (and its session index entry rewritten to point here). Without
    this, a session could end up listed as owner on multiple state
    files and get_task_dir would return whichever the OS happened to
    list first.

    Refuse-overwrite logic on the target:
      1. force=True                                          → write
      2. no existing owner on `task_dir`                     → write
      3. existing owner.session_id == our session            → write
         (legitimate re-claim by the same agent)
      4. existing owner.session_id != ours, but state's
         last_touched is older than heartbeat_fresh_seconds  → write
         (prior owner is silent → presumed dead)
      5. otherwise (different session, fresh heartbeat)      → refuse
    """
    if not os.path.isdir(task_dir):
        return False
    state = load_state(task_dir) or _fresh_state()
    our_session = _our_session_id()
    new_abs = os.path.abspath(task_dir)
    if not force:
        existing = state.get("owner") or {}
        existing_sid = existing.get("session_id") or ""
        same_session = our_session and existing_sid == our_session
        if existing_sid and not same_session and _heartbeat_fresh(state):
            print(f"[state_store] WARNING: refusing to claim {task_dir} "
                  f"— owned by session_id={existing_sid} "
                  f"(heartbeat fresh). Our session_id="
                  f"{our_session or '<none>'}. Stop the other session "
                  f"or rm state.json's owner to take over.",
                  file=sys.stderr)
            return False
    # Release the prior task this session was driving (if any), so
    # the session is never simultaneously listed as owner of two
    # tasks. Skip when the prior task IS this one (idempotent re-set).
    if our_session:
        prior = _read_session_index(our_session)
        if prior and os.path.abspath(prior) != new_abs:
            prior_state = load_state(prior)
            if prior_state and (prior_state.get("owner") or {}).get(
                    "session_id") == our_session:
                prior_state["owner"] = None
                save_state(prior, prior_state)
    state["owner"] = {
        "session_id": our_session,
        "pid":        os.getpid(),
        "claimed_at": _now_iso(),
    }
    save_state(task_dir, state)
    if our_session:
        _write_session_index(our_session, new_abs)
    return True


def clear_active_task(expected_task_dir: Optional[str] = None,
                      *, force: bool = False) -> bool:
    """Release ownership.

    Two caller patterns:
      - in-session hook releasing its own task: omit expected_task_dir
        (we look it up via the session index)
      - supervisor (batch/run.py) releasing a task it spawned: pass
        expected_task_dir = the task that just finished

    Decision (each step short-circuits to clear + True):
      1. force=True with a target → clear unconditionally
      2. session match            → clear (mine via session)
      3. expected_task_dir match  → clear (mine via supervisor claim)
      4. heartbeat stale          → clear (prior owner dead)
      5. otherwise                → refuse (live different session)
    """
    our_session = _our_session_id()

    targets: list[str] = []
    if expected_task_dir:
        if os.path.isdir(expected_task_dir):
            targets = [os.path.abspath(expected_task_dir)]
    elif our_session:
        # Index lookup, not directory scan.
        pointed = _read_session_index(our_session)
        if pointed:
            targets = [os.path.abspath(pointed)]

    if not targets:
        # Nothing to clear from our point of view; also clear the
        # session index if it dangled.
        _clear_session_index(our_session)
        return True

    for td in targets:
        state = load_state(td)
        if not state or not state.get("owner"):
            # Already cleared on disk; just drop the index.
            if our_session:
                _clear_session_index(our_session)
            continue

        owner = state["owner"] or {}
        owner_sid = owner.get("session_id") or ""
        same_session = our_session and owner_sid == our_session
        supervisor_claim = (expected_task_dir
                            and os.path.abspath(expected_task_dir) == td)

        if force or same_session or supervisor_claim or not _heartbeat_fresh(state):
            state["owner"] = None
            save_state(td, state)
            if our_session and (same_session or force):
                _clear_session_index(our_session)
            continue

        print(f"[state_store] WARNING: refusing to clear ownership of "
              f"{td} — owned by session_id={owner_sid} (heartbeat "
              f"fresh, neither session-match nor supervisor claim). "
              f"Pass force=True only if you've verified that session "
              f"is truly done.", file=sys.stderr)
        return False

    return True


def find_active_task_dir() -> Optional[str]:
    """Pick the "current" task — used by dashboard / resume / batch
    monitor when no specific task is in mind.

    Priority:
      1. The current session's task via current_session_task_dir
         (covers external --output-dir paths that an ar_tasks-only
         scan would miss).
      2. The task with the most-recent last_touched among those with
         an owner (= last live activation, regardless of which agent)
      3. The most-recently touched task overall (no owner anywhere,
         dashboard fallback)
      4. None
    """
    pinned = current_session_task_dir()
    if pinned:
        return pinned
    owned_freshest: Optional[tuple] = None
    any_freshest: Optional[tuple] = None
    for td in _iter_task_dirs():
        st = load_state(td)
        if not st:
            continue
        last_touched = _age_seconds(st.get("last_touched"))
        cand = (last_touched, td, st)
        if any_freshest is None or last_touched < any_freshest[0]:
            any_freshest = cand
        owner = st.get("owner") or {}
        if owner.get("session_id"):
            if owned_freshest is None or last_touched < owned_freshest[0]:
                owned_freshest = cand
    for pick in (owned_freshest, any_freshest):
        if pick is not None:
            return pick[1]
    return None


def touch_heartbeat(task_dir: str):
    """Bump state.last_touched. Cheap atomic write — every hook fire
    calls this. Failed touch is reported to stderr — silently
    swallowing would make the session look dead in a hard-to-debug
    way."""
    try:
        state = load_state(task_dir) or _fresh_state()
        save_state(task_dir, state)
    except Exception as e:
        print(f"[AR] WARNING: heartbeat write failed ({e}); resume.py "
              f"may misreport this task as inactive.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public facade — task_summary
# ---------------------------------------------------------------------------
# Single helper that bundles the things outside callers (batch, resume,
# dashboard, scaffold) actually need: phase, ownership/freshness,
# progress numbers, baseline outcome, pending sentinel, consistency
# status. Routing all reads through here prevents the regression class
# where adding a field or changing semantics breaks a different file's
# direct poke.

def is_task_active(task_dir: str) -> bool:
    """True iff someone currently OWNS the task AND has touched it
    recently. The conjunction matters:

      - clear_active_task / save_state bumps last_touched every time,
        so freshness stays True for heartbeat_fresh_seconds after
        the supervisor explicitly released the task — long enough
        for a batch restart to see "fresh, refuse takeover" and pin a
        no-owner task.
      - A crashed agent leaves state.owner non-None but stops bumping
        last_touched; freshness goes False after the window and
        recovery callers can take over safely.

    Resume's "another session is running this" gate and batch's
    "demote stale running" check both want this stronger predicate."""
    state = load_state(task_dir)
    if state is None:
        return False
    owner = state.get("owner") or {}
    if not owner.get("session_id"):
        return False
    return _heartbeat_fresh(state)


def task_owner_info(task_dir: str) -> Optional[dict]:
    """Return the owner record `{session_id, pid, claimed_at}` or
    None when nobody owns this task."""
    state = load_state(task_dir)
    if state is None:
        return None
    return state.get("owner")


def task_summary(task_dir: str) -> Optional[dict]:
    """One-stop "what's the state of this task" view. Designed for
    callers that need to display / decide based on task state
    (batch / resume / dashboard / scaffold) without reaching into
    the state.json schema themselves.

    Returns None when no state.json exists for this task.
    """
    state = load_state(task_dir)
    if state is None:
        return None
    consistency = check_state_consistency(task_dir)
    owner = state.get("owner") or {}
    fresh = _heartbeat_fresh(state)
    return {
        "phase":         state.get("phase", INIT),
        "owner":         state.get("owner"),
        "last_touched":  state.get("last_touched"),
        # is_fresh = recently touched (time only). is_active = somebody
        # owns it AND time is recent. Callers that decide "can I take
        # this over" want is_active; callers that decide "is this
        # responsive at all" want is_fresh.
        "is_fresh":      fresh,
        "is_active":     fresh and bool(owner.get("session_id")),
        "progress_initialized": bool(state.get("progress_initialized")),
        # Progress fields are meaningful only when progress_initialized.
        # Callers can still read them when False but should expect
        # zeros / Nones (they're scaffold defaults from baseline init).
        "task":          state.get("task") or "",
        "eval_rounds":   state.get("eval_rounds") or 0,
        "max_rounds":    state.get("max_rounds") or 0,
        "best_metric":   state.get("best_metric"),
        "baseline_metric":       state.get("baseline_metric"),
        "baseline_outcome":      state.get("baseline_outcome"),
        "baseline_error_source": state.get("baseline_error_source"),
        "consecutive_failures":  state.get("consecutive_failures") or 0,
        "plan_version":          state.get("plan_version") or 0,
        # Replay sentinel — None when no in-flight settle.
        "pending_settle": state.get("pending_settle"),
        # Cross-file consistency vs plan.md / history.jsonl artifacts.
        "consistent":    consistency["consistent"],
        "issues":        consistency["issues"],
    }


# ---------------------------------------------------------------------------
# Phase R/W (state.json.phase field)
# ---------------------------------------------------------------------------

def read_phase(task_dir: str) -> str:
    """Return the current phase from state.json, or INIT when state
    is missing / phase value is corrupt (with a stderr warning so the
    corrupt-state path is visible during recovery)."""
    state = load_state(task_dir)
    if state is None:
        return INIT
    phase = state.get("phase")
    if phase in ALL_PHASES:
        return phase
    print(f"[state_store] WARNING: state.json has unrecognised phase "
          f"{phase!r}; treating as INIT. Recovery options: re-run "
          f"baseline.py / pipeline.py to advance, or delete "
          f"{state_record_path(task_dir)} to start over.",
          file=sys.stderr)
    return INIT


def write_phase(task_dir: str, phase: str):
    """Write phase into state.json. Atomic single-file commit; no
    cross-file coordination needed here."""
    assert phase in ALL_PHASES, f"Invalid phase: {phase}"
    state = load_state(task_dir) or _fresh_state()
    state["phase"] = phase
    save_state(task_dir, state)


# ---------------------------------------------------------------------------
# Progress R/W (Progress fields embedded in state)
# ---------------------------------------------------------------------------

def load_progress(task_dir: str) -> Optional[Progress]:
    """Read the Progress dataclass view from state.json, or None when
    state is missing OR `progress_initialized` is False.

    The False case keeps "task has been claimed by a session but
    baseline hasn't run yet" distinct from "task has measured
    progress". Without this gate resume/dashboard would treat a
    freshly-claimed empty state as Round 0/0 and offer to resume it.
    """
    state = load_state(task_dir)
    if state is None or not state.get("progress_initialized"):
        return None
    progress_fields = {k: v for k, v in state.items()
                       if k in _PROGRESS_FIELD_NAMES}
    return Progress.from_dict(progress_fields)


def save_progress(task_dir: str, progress: Union[Progress, dict],
                  *, stamp: bool = True):
    """Merge progress fields into state.json and atomically save.
    `stamp=True` updates the in-state `last_updated` field (Progress
    schema's own timestamp, distinct from state.last_touched)."""
    state = load_state(task_dir) or _fresh_state()
    if isinstance(progress, Progress):
        if stamp:
            progress = progress.apply(last_updated=_now_iso())
        payload = progress.to_dict()
    else:
        payload = dict(progress)
        if stamp:
            payload["last_updated"] = _now_iso()
    for k, v in payload.items():
        if k in _PROGRESS_FIELD_NAMES:
            state[k] = v
    save_state(task_dir, state)


def append_history(task_dir: str, record: dict):
    """Append one JSON record to history.jsonl. Append-only artifact;
    each row is self-contained and immutable. Cross-file consistency
    with state.json's `expected_history_round` is the caller's
    responsibility (typically: bump expected_history_round in the
    same save_state that wraps this round's body writes)."""
    path = history_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sanitize_floats(record), ensure_ascii=False) + "\n")


def update_progress(task_dir: str, **fields) -> Optional[Progress]:
    """Load Progress, .apply(**fields), save. Returns the new Progress.

    Field-name validation is delegated to Progress.apply, so a typo
    here becomes TypeError instead of a silently-dropped attribute.

    Returns None only when state.json doesn't exist (pre-scaffold,
    legitimate no-op). Save failures re-raise after a loud stderr
    warning — earlier callers silently lost DIAGNOSE attempt counts
    and consecutive_failures resets when the write failed, producing
    infinite-retry loops the operator couldn't trace back."""
    progress = load_progress(task_dir)
    if progress is None:
        return None
    new_progress = progress.apply(**fields)
    try:
        save_progress(task_dir, new_progress, stamp=False)
    except Exception as e:
        print(f"[state_store] CRITICAL: failed to save state.json for "
              f"{task_dir}: {type(e).__name__}: {e}. fields="
              f"{list(fields)}. The in-memory update is lost; the next "
              f"round may see stale state. Free disk space / fix "
              f"permissions and re-run the failed action.",
              file=sys.stderr)
        raise
    return new_progress


# ---------------------------------------------------------------------------
# Cross-file consistency check
# ---------------------------------------------------------------------------
# state.json is the commit barrier. plan.md and history.jsonl are
# durable artifacts written outside state.json — a writer that landed
# them but didn't commit state.json leaves a detectable gap. The check
# below compares state.expected_plan_version / expected_history_round
# against the actual artifact contents.

def _read_plan_version_from_disk(task_dir: str) -> Optional[int]:
    """Parse plan.md's `# Plan vN` header; None when plan.md missing
    or unparseable. (PlanStore.parse_version_on_disk exists too but
    we keep this local to avoid the workflow→phase_machine import
    cycle.)"""
    path = plan_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
    except OSError:
        return None
    import re as _re
    m = _re.match(r"^#\s*Plan\s+v(\d+)\b", first)
    return int(m.group(1)) if m else None


def _read_last_history_round(task_dir: str) -> Optional[int]:
    """Parse history.jsonl's last row's `round` field; None when no
    history yet or the last row is unparseable."""
    path = history_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            try:
                size = os.path.getsize(path)
                if size == 0:
                    return None
                f.seek(-1, os.SEEK_END)
                while f.tell() > 0:
                    if f.read(1) == b"\n":
                        if f.tell() == size:
                            f.seek(-2, os.SEEK_END)
                            continue
                        break
                    f.seek(-2, os.SEEK_CUR)
                last = f.readline().decode("utf-8", errors="replace").strip()
            except OSError:
                return None
        if not last:
            return None
        row = json.loads(last)
        v = row.get("round")
        return int(v) if isinstance(v, (int, float)) else None
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def check_state_consistency(task_dir: str) -> dict:
    """Return a report:
        {"consistent": bool,
         "state": <full state dict or None>,
         "issues": [<human-readable issue strings>]}

    issues is empty when state is consistent; otherwise lists each
    artifact-vs-state mismatch.
    """
    state = load_state(task_dir)
    if state is None:
        # No state → nothing to compare. Treat as consistent (fresh
        # task or pre-write state).
        return {"consistent": True, "state": None, "issues": []}

    issues = []
    expected_plan = int(state.get("expected_plan_version") or 0)
    plan_on_disk = _read_plan_version_from_disk(task_dir)
    if plan_on_disk is not None and plan_on_disk != expected_plan:
        issues.append(
            f"plan.md is at v{plan_on_disk} but state.expected_plan_"
            f"version={expected_plan}. Normal crashes are healed by "
            f"replay_intent at pipeline entry; reaching this point "
            f"means either an off-flow edit to plan.md or a "
            f"create_plan.py run that failed AFTER landing plan.md "
            f"but BEFORE its save_state. Re-run create_plan.py with "
            f"the same input — it's idempotent on the target version.")

    expected_round = int(state.get("expected_history_round") or 0)
    last_round = _read_last_history_round(task_dir)
    if last_round is not None and last_round != expected_round:
        issues.append(
            f"history.jsonl last row is round {last_round} but state."
            f"expected_history_round={expected_round} and no intent."
            f"json journal is present. Normal crashes are healed by "
            f"replay_intent; reaching this point means either an "
            f"off-flow append to history.jsonl or an intent.json that "
            f"was manually deleted before the next pipeline run. "
            f"Inspect the orphan row — if it's safe to discard, "
            f"delete it and re-run; if it's the missing settle, "
            f"hand-set state.pending_settle from the row and re-run "
            f"pipeline.py.")

    return {"consistent": not issues, "state": state, "issues": issues}


def format_state_inconsistency(report: dict) -> str:
    """Render a check_state_consistency report into a recovery
    message suitable for hook stderr / agent transcript."""
    if report["consistent"]:
        return ""
    state = report.get("state") or {}
    phase = state.get("phase") or "<unknown>"
    head = (f".ar_state is inconsistent (phase={phase}). Writer landed "
            f"body artifacts but never committed state.json:")
    return head + "\n  - " + "\n  - ".join(report["issues"])


# ---------------------------------------------------------------------------
# Journal / write-ahead intent
# ---------------------------------------------------------------------------
# Round and baseline transactions write body artifacts (history.jsonl,
# plan.md) BEFORE the matching state.json save_state. A crash in the
# window leaves bodies ahead of state.expected_*; the previous design
# refused pipeline.py with "pending_settle path will reconcile" — but
# pending_settle was also written by that same final save_state, so
# the message lied. The journal closes the loop:
#
#   1. caller writes intent.json (kind + minimal payload to reconstruct
#      the next state.json) BEFORE touching bodies.
#   2. caller writes bodies + final save_state.
#   3. caller clears intent.json.
#
# On crash recovery, replay_intent at pipeline.py entry inspects the
# leftover intent.json and the existing artifacts to decide:
#   - state already caught up (expected_* match bodies) → just clear
#     the intent file.
#   - bodies landed but state didn't → rebuild state.json from the
#     intent (set pending_settle so the existing replay branch runs).
#   - bodies didn't land → caller can safely redo the whole action.

_INTENT_KIND_ROUND = "round"
_INTENT_KIND_BASELINE = "baseline"
_INTENT_KIND_PLAN = "plan"


def write_intent(task_dir: str, intent: dict) -> None:
    """Atomic intent commit. Caller passes a dict carrying enough to
    reconstruct the post-action state.json (kind + kd_json for
    rounds; kind + progress dict for baseline). Bumping state.last_
    touched is intentionally NOT done here — touch_heartbeat owns
    that and the intent layer should be invisible to "is this task
    fresh" callers."""
    path = intent_path(task_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sanitize_floats(intent), f, indent=2)
    os.replace(tmp, path)


def read_intent(task_dir: str) -> Optional[dict]:
    """Returns the intent dict, or None when no journal exists / it's
    corrupt (corruption is logged loudly — same WARNING pattern as
    load_state, since acting on a corrupt journal could double-write
    history)."""
    path = intent_path(task_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"[state_store] WARNING: intent.json at {path} corrupt "
              f"({e}); treating as missing. Inspect and rm to recover.",
              file=sys.stderr)
        return None
    if not isinstance(data, dict):
        return None
    return data


def clear_intent(task_dir: str) -> None:
    path = intent_path(task_dir)
    try:
        os.remove(path)
    except OSError:
        pass


def replay_intent(task_dir: str) -> Optional[dict]:
    """Reconcile an in-flight transaction journaled before the crash.
    Returns a status dict for callers that want to surface what
    happened, or None when no intent exists.

    Status dict shape: {"kind": str, "action": str, "detail": str}
    action ∈ {
       "cleared":      intent existed but state had already caught up.
       "rebuilt":      bodies landed pre-crash; we rebuilt state.json
                       (pending_settle / progress fields) so the
                       normal replay branch in pipeline.py can finish.
       "discarded":    bodies didn't land pre-crash; intent dropped.
                       Caller can safely redo the whole action.
       "skipped":      intent unrecognised; left in place. Loud
                       warning to stderr — caller should inspect.
    }
    """
    intent = read_intent(task_dir)
    if intent is None:
        return None
    kind = intent.get("kind")
    if kind == _INTENT_KIND_ROUND:
        return _replay_round_intent(task_dir, intent)
    if kind == _INTENT_KIND_BASELINE:
        return _replay_baseline_intent(task_dir, intent)
    if kind == _INTENT_KIND_PLAN:
        return _replay_plan_intent(task_dir, intent)
    print(f"[state_store] WARNING: intent.json has unknown kind "
          f"{kind!r}; leaving in place. Inspect "
          f"{intent_path(task_dir)} and rm if safe to discard.",
          file=sys.stderr)
    return {"kind": str(kind), "action": "skipped",
            "detail": f"unknown kind {kind!r}"}


def _replay_round_intent(task_dir: str, intent: dict) -> dict:
    """Round intent payload:
        {"kind":"round", "kd_json":{...}, "round":N,
         "progress_fields":{<Progress.to_dict subset>}}

    On crash recovery the three possible disk states are:
      (a) state already at round N (expected_history_round==N AND
          pending_settle ≈ kd_json): intent is leftover, clear it.
      (b) history.jsonl last row.round == N but state isn't caught
          up: rebuild state from the intent so the existing
          pending_settle replay branch in pipeline.py runs.
      (c) history.jsonl last row.round != N (or no history yet):
          the body never landed; drop the intent so the action can
          be redone.
    """
    target_round = int(intent.get("round") or 0)
    state = load_state(task_dir) or {}

    if state.get("expected_history_round") == target_round:
        # State caught up. pending_settle may or may not still be
        # there depending on where we crashed; either way, the
        # journal has nothing to add. Drop it.
        clear_intent(task_dir)
        return {"kind": "round", "action": "cleared",
                "detail": f"state.expected_history_round already {target_round}"}

    last_round = _read_last_history_round(task_dir)
    if last_round == target_round:
        # Body landed; rebuild state so consistency gate passes and
        # the replay branch reconstructs the settle.
        kd_json = intent.get("kd_json") or {}
        progress_fields = intent.get("progress_fields") or {}
        for k, v in progress_fields.items():
            if k in _PROGRESS_FIELD_NAMES:
                state[k] = v
        state["pending_settle"] = kd_json
        state["expected_history_round"] = target_round
        save_state(task_dir, state)
        clear_intent(task_dir)
        return {"kind": "round", "action": "rebuilt",
                "detail": f"reconstructed pending_settle for round "
                          f"{target_round} from journal"}

    # Body didn't land — discard.
    clear_intent(task_dir)
    return {"kind": "round", "action": "discarded",
            "detail": f"history.jsonl last_round={last_round} != "
                      f"intent.round={target_round} (body never landed)"}


def _replay_baseline_intent(task_dir: str, intent: dict) -> dict:
    """Baseline intent payload:
        {"kind":"baseline", "progress_fields":{...},
         "expected_history_round": 0}

    Three disk shapes:
      (a) state already progress_initialized=True AND
          expected_history_round=0: intent is leftover; clear it.
      (b) history.jsonl already has a round=0 SEED row but state
          isn't caught up: rebuild progress + flip
          progress_initialized so dashboards / pipeline see the
          baseline as committed.
      (c) no SEED row yet: discard — caller redoes baseline cleanly.
    """
    state = load_state(task_dir) or {}
    if state.get("progress_initialized") and \
            state.get("expected_history_round") == 0:
        clear_intent(task_dir)
        return {"kind": "baseline", "action": "cleared",
                "detail": "state already shows baseline committed"}

    last_round = _read_last_history_round(task_dir)
    if last_round == 0:
        for k, v in (intent.get("progress_fields") or {}).items():
            if k in _PROGRESS_FIELD_NAMES:
                state[k] = v
        state["progress_initialized"] = True
        state["expected_history_round"] = 0
        save_state(task_dir, state)
        clear_intent(task_dir)
        return {"kind": "baseline", "action": "rebuilt",
                "detail": "reconstructed progress fields from journal "
                          "(SEED row already in history.jsonl)"}

    clear_intent(task_dir)
    return {"kind": "baseline", "action": "discarded",
            "detail": "no SEED row on disk; intent dropped"}


def _replay_plan_intent(task_dir: str, intent: dict) -> dict:
    """Plan intent payload:
        {"kind":"plan", "version":N, "progress_fields":{...}}

    create_plan writes plan.md (the body) first, then save_state with
    expected_plan_version=N. The window: SIGKILL after ps.write,
    before save_state, leaves plan.md at vN but state still at vN-1.
    Three disk shapes:
      (a) state already at vN: intent leftover; clear it.
      (b) plan.md vN on disk, state at vN-1: rebuild state from the
          intent so consistency gate passes and downstream readers
          see the new plan_version + next_pid.
      (c) plan.md not at vN (still vN-1 because ps.write never
          flushed): discard the intent so the next create_plan starts
          from a clean slate.
    """
    target_version = int(intent.get("version") or 0)
    state = load_state(task_dir) or {}

    if int(state.get("expected_plan_version") or 0) == target_version:
        clear_intent(task_dir)
        return {"kind": "plan", "action": "cleared",
                "detail": f"state.expected_plan_version already {target_version}"}

    plan_on_disk = _read_plan_version_from_disk(task_dir)
    if plan_on_disk == target_version:
        for k, v in (intent.get("progress_fields") or {}).items():
            if k in _PROGRESS_FIELD_NAMES:
                state[k] = v
        state["expected_plan_version"] = target_version
        save_state(task_dir, state)
        clear_intent(task_dir)
        return {"kind": "plan", "action": "rebuilt",
                "detail": f"reconstructed plan_version + next_pid for "
                          f"v{target_version} from journal"}

    clear_intent(task_dir)
    return {"kind": "plan", "action": "discarded",
            "detail": f"plan.md on disk is at v{plan_on_disk} != "
                      f"intent.version={target_version} "
                      f"(body never landed)"}


def require_state_consistency(task_dir: str,
                              *, on_inconsistent: str = "raise",
                              auto_replay: bool = True) -> dict:
    """Single "I'm about to act on this task" gate for activation
    hooks / resume / pipeline. Heals first, then checks.

    Healing: replay_intent runs at entry by default. The journal owns
    the "bodies-without-state" crash window — replay turns intent +
    leftover artifacts into a consistent state.json (or discards an
    orphan intent). Folding it here means every caller that asks
    "is this task safe to act on" implicitly heals in-flight
    transactions, instead of each caller remembering to call
    replay_intent before this. Set auto_replay=False for read-only
    callers (dashboards, debug tools) that must NOT touch disk.

    On the post-replay inconsistency:
      - on_inconsistent="raise" (default): RuntimeError with the
        recovery message. Pipelines fail loud.
      - on_inconsistent="report": return the report; caller surfaces
        the message its own way.
    """
    if auto_replay:
        replay = replay_intent(task_dir)
        if replay is not None:
            print(f"[state_store] replay_intent {replay['action']}: "
                  f"{replay['detail']}", file=sys.stderr)
    report = check_state_consistency(task_dir)
    if report["consistent"]:
        return report
    if on_inconsistent == "raise":
        raise RuntimeError(format_state_inconsistency(report))
    return report
