#!/usr/bin/env python3
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

"""opencode headless driver: re-invoke `opencode run` until decide() says stop
(opencode has no Stop hook to self-loop like `claude --print`). Per-turn
guardrails are the plugin's job; this is just the cross-turn loop. Pins turn
1's session id and reuses it via `--session`.

    python .opencode/run_loop.py --resume <task_dir> [--force]
    python .opencode/run_loop.py --ref X.py --kernel Y.py --op-name foo --devices 0

Env: AR_OPENCODE_BIN, AR_OPENCODE_MODEL, AR_OPENCODE_ENV_FILE,
DEEPSEEK_API_KEY, NODE_TLS_REJECT_UNAUTHORIZED.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # workspace_autoresearch/
sys.path.insert(0, str(ROOT / "scripts"))
from phase_machine import read_phase, FINISH  # noqa: E402
from decide import AgentEvent, decide  # noqa: E402  (the single stop verdict)
_bin = os.environ.get("AR_OPENCODE_BIN", "opencode")
OPENCODE = shutil.which(_bin) or _bin   # resolve like run.py._resolve_claude_bin (Windows PATHEXT)
# No hardcoded default — mirror build_claude_cmd: pass --model only when given,
# else the agent uses its own configured default (opencode.jsonc / claude).
MODEL = os.environ.get("AR_OPENCODE_MODEL", "")

_SUBPROC_TIMEOUT_SEC = 15  # taskkill helper cap


def _last_json_obj(text: str):
    """Last stdout line that parses as a JSON object. parse_args/scaffold emit
    their machine-readable record on one line, but `--run-baseline` pulls in
    CANN, which prints `[Warning]: tiling struct ...` to stdout AFTER it — so
    the record is NOT reliably the last line. Scan bottom-up for the real one."""
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _scaffold(passthrough) -> str:
    """Resolve a task_dir the same way the /autoresearch command does: run
    parse_args.py, then run the dispatch command it returns, and read the
    task_dir from its JSON record (robust to trailing library noise)."""
    pa = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "engine" / "parse_args.py"),
         *passthrough],
        capture_output=True, text=True, cwd=str(ROOT))
    rec = _last_json_obj(pa.stdout)
    if rec is None:
        sys.exit(f"[run_loop] parse_args produced no JSON record:\n"
                 f"{pa.stdout}\n{pa.stderr}")
    if rec.get("mode") == "ask":
        sys.exit(f"[run_loop] missing flags: {rec.get('missing')}")
    argv = rec.get("argv")
    if not (isinstance(argv, list)
            and argv and all(isinstance(x, str) for x in argv)):
        sys.exit(f"[run_loop] parse_args produced no argv: {rec}")
    res = subprocess.run(argv, capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode != 0:
        sys.exit(f"[run_loop] scaffold failed:\n{res.stdout}\n{res.stderr}")
    rec = _last_json_obj(res.stdout)
    task_dir = rec.get("task_dir") if rec else ""
    if not task_dir or not os.path.isdir(task_dir):
        sys.exit(f"[run_loop] scaffold did not return a valid task_dir.\n"
                 f"stdout tail:\n{res.stdout[-600:]}")
    return task_dir


def _resume_task(task_dir: str, *, force: bool = False) -> str:
    """Enter the canonical resume transaction before starting OpenCode.

    Merely pointing ``AR_TASK_DIR`` at an existing task is insufficient: a
    fresh OpenCode process has a fresh session id, while state.owner may still
    name the process that was interrupted.  ``scripts/resume.py`` is the one
    lifecycle entry that heals state, validates resumability, and transfers
    ownership.  Run it here instead of duplicating those state-machine rules
    in the headless adapter.  ``--force`` stays explicit so a live foreign
    session is never taken over accidentally.
    """
    requested = os.path.abspath(task_dir)
    cmd = [sys.executable, str(ROOT / "scripts" / "resume.py"), requested]
    if force:
        cmd.append("--force")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode != 0:
        sys.exit(f"[run_loop] resume failed:\n{res.stdout}\n{res.stderr}")

    # resume.py prints the canonical task path on its last non-empty stdout
    # line.  Consume that contract so path normalisation has one owner too.
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    resumed = os.path.abspath(lines[-1]) if lines else requested
    if not os.path.isdir(resumed):
        sys.exit(f"[run_loop] resume returned no valid task_dir:\n{res.stdout}")
    return resumed


def _turn_prompt(task_dir: str) -> str:
    # re-export every turn (idempotent activation), then defer to plugin guidance
    return (
        f'First run this bash command verbatim: export AR_TASK_DIR="{task_dir}"  '
        f'. Then follow the [AR Phase: ...] guidance appended to each tool '
        f'result and perform the current phase\'s action via the prescribed '
        f'scripts. Keep working through tool calls; do not end your turn until '
        f'you have completed at least one phase action (or a tool blocks you '
        f'with a reason to address).'
    )


def _kill_opencode(proc) -> None:
    """Best-effort kill of the opencode child on standalone Ctrl-C. opencode
    shares run_loop's process group (see _opencode_turn), so on POSIX we must
    NOT killpg — that would SIGKILL run_loop itself; a tty Ctrl-C has already
    SIGINT'd the shared group's descendants, so killing the direct child is
    enough. On Windows opencode is a child PID (no shared-group hazard), so
    taskkill /T can reap its whole subtree. (The batch's outer-timeout kill is
    handled by run.py over the whole group/PID-tree, not here.)"""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=_SUBPROC_TIMEOUT_SEC)
        else:
            proc.kill()
    except Exception:
        pass


# OpenCode log layouts differ by version:
#   old: `... message=created id=ses_xxx slug=...`
#   1.14: `... service=session id=ses_xxx ... created`
_SESSION_RE = re.compile(
    r"(?:created id=|service=session id=)(ses_[A-Za-z0-9]+)")
_FATAL_PROVIDER_MARKERS = (
    "ProviderModelNotFoundError",
    "AuthenticationError",
    "NoSuchModelError",
)


def _opencode_turn(task_dir: str, session_id):
    """Run one opencode turn; return ``(rc, session_id, fatal_provider)``.

    Explicit `--session <id>` (captured from turn 1's `--print-logs`), never
    `--continue` (= most-recent session, which races concurrent drivers)."""
    cmd = [OPENCODE, "run", "--print-logs"]
    if MODEL:
        cmd += ["--model", MODEL]
    if session_id:
        cmd += ["--session", session_id]
        label = f"--session {session_id[:12]}"
    else:
        label = "new session"
    cmd.append(_turn_prompt(task_dir))
    print(f"[run_loop] -> opencode run ({label}, phase={read_phase(task_dir)})",
          flush=True)

    # Inherit run_loop's process group (do NOT start_new_session): the batch
    # spawns run_loop in its own group and reaps that group on the outer
    # timeout (run.py _kill_process_tree). Keeping opencode in the SAME group
    # is what makes "the batch's outer wall-clock is the only bound" true — a
    # fresh session here would orphan opencode/shell/build out of the group the
    # batch kills.
    child_env = os.environ.copy()
    # The headless driver owns cross-turn continuation. Disable the plugin's
    # interactive session.idle re-prompt in this child, otherwise both layers
    # can enqueue the next turn and race/duplicate phase actions.
    child_env["AR_EXTERNAL_LOOP"] = "1"
    if os.name == "posix":
        # OpenCode/Bun 1.14 uses PWD to choose the project config. Popen(cwd=)
        # does not update an explicitly inherited PWD environment variable.
        child_env["PWD"] = str(ROOT)
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env=child_env)
    seen_sid = None
    fatal_provider = False
    try:
        for line in proc.stdout:           # tee + scan turn 1's session id
            sys.stdout.write(line)
            sys.stdout.flush()
            if seen_sid is None:
                m = _SESSION_RE.search(line)
                if m:
                    seen_sid = m.group(1)
            if any(marker in line for marker in _FATAL_PROVIDER_MARKERS):
                fatal_provider = True
        proc.wait()
        # OpenCode 1.14 can report an HTTP 500 provider/config error in logs
        # yet still exit 0. Surface it to the outer driver instead of spinning
        # forever on a session that can never produce a turn.
        rc = proc.returncode or (1 if fatal_provider else 0)
        return rc, seen_sid, fatal_provider
    except KeyboardInterrupt:
        _kill_opencode(proc)
        raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", metavar="TASK_DIR",
                    help="drive an already-scaffolded task")
    ap.add_argument("--force", action="store_true",
                    help="with --resume, take over after verifying the prior "
                         "session is no longer running")
    args, passthrough = ap.parse_known_args()

    if args.force and not args.resume:
        ap.error("--force requires --resume TASK_DIR")

    task_dir = (_resume_task(args.resume, force=args.force)
                if args.resume else _scaffold(passthrough))
    task_dir = os.path.abspath(task_dir)
    os.environ["AR_TASK_DIR"] = task_dir   # decide()/children resolve this task
    print(f"[run_loop] task_dir={task_dir}", flush=True)

    session_id = None      # pinned to turn 1's session (concurrency-safe)
    # Loop on the SAME verdict as the Claude Stop hook (block == keep going).
    # No iteration cap — like claude --print, the bounds are the stop verdict
    # plus the batch's outer wall-clock timeout.
    while decide(AgentEvent(kind="stop", stop_reason="loop")).block:
        rc, sid, fatal_provider = _opencode_turn(task_dir, session_id)
        if sid and not session_id:
            session_id = sid
            print(f"[run_loop] pinned opencode session={session_id}",
                  flush=True)
        if fatal_provider:
            print("[run_loop] opencode reported a fatal provider/model/auth "
                  "error; aborting instead of retrying the same session.",
                  flush=True)
            return rc
        if session_id is None:
            print("[run_loop] opencode emitted no session id; refusing an "
                  "unsafe --continue fallback.", flush=True)
            return rc or 2
        # Process-level failure with no session ever established (bad config /
        # API / auth): opencode can't even start a turn, so respawning just
        # loops into the same error until the outer timeout. Bail to the batch
        # like a failed `claude --print`. Once a session exists, a non-zero rc
        # is a within-turn guardrail/block failure → keep looping; decide()
        # owns the stop decision.
        if rc != 0 and session_id is None:
            print(f"[run_loop] opencode exited rc={rc} before starting a "
                  f"session — aborting (likely config/API/auth).", flush=True)
            return rc

    final = read_phase(task_dir)
    print(f"[run_loop] loop ended at phase={final} "
          f"({'FINISH reached' if final == FINISH else 'not FINISH'})",
          flush=True)
    return 0 if final == FINISH else 1


if __name__ == "__main__":
    sys.exit(main())
