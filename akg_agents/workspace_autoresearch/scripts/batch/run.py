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

"""Batch driver for /autoresearch.

Loads a manifest from <batch_dir>/manifest.{yaml,json}, resolves the op
list against the <op_name>_{ref,kernel}.py naming convention, then drives each
op end-to-end via headless `claude --print`. Streams stdout to console and
batch.log, updates batch_progress.json after every op.

Usage:
    python scripts/batch/run.py <batch_dir> --devices N \\
        [--max-rounds N] [--eval-timeout S] [--timeout-min M] \\
        [--only op1,op2] [--limit N] [--retry-errored] [--cooldown-sec S]
"""
from __future__ import annotations

import argparse
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Force UTF-8 on this script's own stdout/stderr. claude.cmd prints
# tokens like `µs`, box-drawing rules, and Chinese rationale text;
# on Chinese-locale Windows the default GBK codec can't encode them
# and `sys.stdout.write(line)` (line 405) raises mid-batch, killing
# the supervisor while ops are still queued. Sister fix to the
# subprocess-read encoding pin already on Popen below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _resolve_claude_bin(name: str) -> str:
    """Resolve `name` to a real path Popen can execute.

    Bare 'claude' on Windows fails: Popen(list) calls CreateProcess
    directly and CreateProcess does not apply PATHEXT, so it won't find
    `claude.cmd`. shutil.which DOES walk PATHEXT and returns the full
    path (including `.cmd` on Windows). On POSIX this is also fine —
    shutil.which returns the resolved absolute path, which Popen handles
    identically to the bare name."""
    if os.path.isabs(name) or os.sep in name:
        return name  # caller already gave a path
    resolved = shutil.which(name)
    return resolved or name  # fall through to original on miss (Popen will raise with the same FileNotFoundError as before — better diagnostic than swapping silently)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf
# Reach up one level (scripts/) for the shared config accessors so batch
# defaults match single-task /autoresearch instead of drifting.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase_machine  # noqa: E402
import task_handle  # noqa: E402
from akg_agents.utils.process_utils import (  # noqa: E402
    popen_process_group_kwargs, terminate_process_tree,
)
from utils.settings import (  # noqa: E402
    default_max_rounds, default_eval_timeout,
    batch_run_timeout_min, batch_cooldown_sec, batch_transient_retries,
    recorded_speedup,
)


# Force line-buffered stdout so logs flush in real time when run via nohup.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass
os.environ.setdefault("PYTHONUNBUFFERED", "1")


def _console_write(text: str, *, flush: bool = True) -> None:
    """Write to the interactive console without making it part of the
    batch transaction.

    A foreground batch is commonly launched through ``ssh ... | tee``.  If
    that controlling SSH connection disappears, the remote agent and its eval
    children can keep running, but the inherited stdout pipe eventually raises
    ``BrokenPipeError``.  Console output is only an observer; losing it must not
    prevent the driver from harvesting a completed task into
    ``batch_progress.json``.  After the first sink failure, replace stdout with
    ``os.devnull`` so ordinary ``print`` calls later in the batch are harmless.
    """
    try:
        sys.stdout.write(text)
        if flush:
            sys.stdout.flush()
    except (BrokenPipeError, OSError, ValueError):
        try:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
        except OSError:
            pass


def _emit(log_fp, text: str) -> None:
    """Persist output first, then mirror it to the best-effort console."""
    log_fp.write(text)
    log_fp.flush()
    _console_write(text)


# Keep the prompt to the slash command itself. Claude passes all text after
# `/autoresearch` as `$ARGUMENTS`; appending extra prose here corrupts the
# argument vector consumed by scripts/engine/parse_args.py.
PROMPT_TEMPLATE = (
    "/autoresearch --ref {ref} --kernel {kernel} --op-name {op} {hw} "
    "--max-rounds {rounds} --eval-timeout {timeout}"
)

LOCK_FILENAME = ".batch.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            SYNCHRONIZE = 0x00100000
            h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if not h:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        except Exception:
            # Can't tell — err on the safe side and assume alive so the user
            # has to confirm by removing the lock manually.
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def acquire_lock(batch_dir: Path) -> Path:
    """Prevent two run.py instances racing on the same batch_progress.json.
    Stale locks (PID gone) are auto-cleared; live locks abort with a hint.

    Uses os.open(O_CREAT|O_EXCL) — atomic create-or-fail on every OS
    Python supports (POSIX + Windows). The old "exists() then write_text"
    pattern was a check-then-act race: two run.py instances starting in
    parallel could both see no lock, both write their PID, and both
    proceed to corrupt batch_progress.json. Atomic create eliminates
    that race; the stale-lock retry path is bounded to one cycle so two
    racers both finding a stale lock can't ping-pong it forever — the
    loser of the second atomic create aborts with a clear hint.
    """
    lock = batch_dir / LOCK_FILENAME
    for attempt in range(2):
        try:
            fd = os.open(str(lock),
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                pid = int(lock.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                pid = -1
            if pid > 0 and _pid_alive(pid):
                sys.exit(
                    f"\nanother batch run is active on this batch dir "
                    f"(pid={pid}, lock={lock}).\n"
                    f"if you're sure no run.py is running, remove {lock} "
                    f"and retry.\n"
                )
            if attempt == 0:
                # stale; unlink and retry atomic create. A second racer
                # who also sees the stale lock may unlink between us, in
                # which case our retry's O_EXCL wins one of the two.
                try:
                    lock.unlink()
                except OSError:
                    pass
                continue
            # Second collision = real concurrent racer. Bail.
            sys.exit(
                f"\nbatch dir {batch_dir} is being claimed concurrently "
                f"(another run.py won the lock race).\n"
                f"retry in a moment.\n"
            )
        else:
            try:
                os.write(fd, str(os.getpid()).encode())
            finally:
                os.close(fd)
            return lock
    return lock  # unreachable, the loop either returns or sys.exits


def release_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except OSError:
        pass


def recover_stale_running(progress: dict) -> tuple[int, int]:
    """Demote 'running' cases that are demonstrably orphaned. We hold the
    batch dir lock when this fires, so anything still 'running' was left
    by a previous run.py. "Previous run.py" can mean any of:
    SIGKILLed, OOM-killed, machine rebooted (true orphans) OR another
    runner that's still alive but raced us through the (now atomic)
    lock OR a case whose runner is gone but whose claude --print is
    still finishing its last tool call (state.json keeps getting
    touched on the task_dir).

    Demoting all "running" indiscriminately means --retry-errored
    later re-launches a case that's still in flight, putting two
    Claude processes on the same task and the same worker — a silent
    double-run footgun. Check before demoting:
      - if the case carries a `runner_pid` and that pid is alive,
        it's not orphaned — skip.
      - if the task_dir is_task_active (owner + fresh heartbeat in
        state.json), /autoresearch is still writing — skip.
      - once both owners are dead, harvest an already-FINISH task from its
        authoritative state instead of launching the whole optimization again.
      - otherwise the case is a real incomplete orphan; demote with a note.

    Returns ``(demoted, harvested)``.
    """
    # Route through the phase_machine facade so the "is this task
    # still live" judgement has one owner (reads state.last_touched).
    import sys as _sys
    _scripts = str(Path(__file__).resolve().parent.parent)
    if _scripts not in _sys.path:
        _sys.path.insert(0, _scripts)
    from phase_machine import is_task_active  # noqa: E402

    cases = progress.get("cases", {})
    demoted = harvested = 0
    now = mf.now_iso()
    for c in cases.values():
        if c.get("status") != "running":
            continue
        # Skip if the previous runner is still alive.
        runner_pid = c.get("runner_pid")
        if isinstance(runner_pid, int) and runner_pid > 0 and _pid_alive(runner_pid):
            continue
        # Skip if the task is still active — claude --print may have
        # lost its runner pid (e.g. detached) but the agent loop is
        # still bumping state.last_touched.
        td = c.get("task_dir")
        if td and is_task_active(td):
            continue
        if td:
            task_dir = Path(td)
            if mf.read_phase(task_dir) == "FINISH":
                c.update({
                    "status": "done",
                    "finished_at": now,
                    "final_phase": "FINISH",
                    "rc": 0,
                    "result": mf.read_task_state(task_dir),
                })
                existing = (c.get("note") or "").strip()
                tag = ("harvested completed task on batch restart "
                       "after runner exit")
                c["note"] = f"{existing}; {tag}" if existing else tag
                harvested += 1
                continue
        c["status"] = "error"
        c["finished_at"] = now
        existing = (c.get("note") or "").strip()
        tag = ("stale running, demoted on batch restart "
               "(no live runner_pid, task not active in state.json)")
        c["note"] = f"{existing}; {tag}" if existing else tag
        demoted += 1
    return demoted, harvested


def build_prompt(case: dict, hw_arg: str,
                 max_rounds: int, eval_timeout: int) -> str:
    """Quote every value-bearing flag with shlex.quote so paths with
    spaces (e.g. batch dir under `C:\\Users\\Foo Bar\\...`, or
    `--output-dir "my tasks"`) reach /autoresearch as one argv each."""
    return PROMPT_TEMPLATE.format(
        ref=shlex.quote(case["ref"]),
        kernel=shlex.quote(case["kernel"]),
        op=shlex.quote(case["op_name"]),
        hw=hw_arg,
        rounds=max_rounds,
        timeout=eval_timeout,
    )


def build_claude_cmd(args: argparse.Namespace, prompt: str) -> list[str]:
    cmd = [
        args.claude_bin,
        "--print",
        "--permission-mode", "acceptEdits",
        "--output-format", "text",
    ]
    if args.model:
        cmd += ["--model", args.model]
    cmd += args.extra_claude_arg
    cmd += [prompt]
    return cmd


def env_with_no_proxy(extra: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = os.environ.copy()
    extras = "127.0.0.1,localhost"
    existing = env.get("NO_PROXY", "")
    env["NO_PROXY"] = f"{existing},{extras}".strip(",") if existing else extras
    env["no_proxy"] = env["NO_PROXY"]
    env["PYTHONIOENCODING"] = "utf-8"  # propagates UTF-8 to claude --print + its Bash-tool subprocs
    if extra:
        env.update(extra)
    return env


def _begin_case(batch_dir: Path, case: dict,
                prev_task_dir: Optional[str]):
    """Shared manifest/ownership setup for every agent driver."""
    op = case["op_name"]
    mf.update_case(
        batch_dir, op, status="running", started_at=mf.now_iso(),
        finished_at=None, task_dir=None, final_phase=None, rc=None,
        runner_pid=os.getpid(), note="",
    )
    pre_task_dirs = mf.snapshot_task_dirs()
    if phase_machine.clear_active_task(expected_task_dir=prev_task_dir):
        return op, mf.repo_root(), time.time(), pre_task_dirs
    sys.stdout.write(
        f"[run] op={op}: refusing to start — another session is active on "
        "this checkout. Stop it before retrying.\n")
    sys.stdout.flush()
    mf.update_case(batch_dir, op, status="error", finished_at=mf.now_iso(),
                   note="aborted: prior owner still active")
    return None


def _find_task_dir(batch_dir: Path, op: str, pre_task_dirs: set,
                   candidate: Optional[str] = None) -> Optional[Path]:
    """Resolve the task produced by either agent, newest evidence first."""
    recorded = (mf.load_progress(batch_dir).get("cases", {}).get(op, {})
                .get("task_dir"))
    for raw in (candidate, recorded):
        if not isinstance(raw, str):
            continue
        task_dir = Path(raw)
        if (task_dir.is_dir()
                and mf.task_dir_belongs_to_op(task_dir.name, op)):
            return task_dir.resolve()
    found = mf.pick_new_task_dir(pre_task_dirs, op)
    return found.resolve() if found is not None else None


def _read_case_result(task_dir: Path, interrupted: bool):
    """Replay an interrupted journal, then read one canonical outcome."""
    consistency_note = ""
    try:
        with task_handle.open_task(
                str(task_dir), role=task_handle.Role.SUPERVISOR):
            pass
    except task_handle.TaskConsistencyError as exc:
        consistency_note = f"; post-run heal refused: {exc}"
    phase = mf.read_phase(task_dir)
    result = mf.read_task_state(task_dir)
    status = ("done" if phase == "FINISH" and not interrupted
              and not consistency_note else "error")
    return phase, result, status, consistency_note


def _finish_case(batch_dir: Path, op: str, task_dir: Path, phase: str,
                 result: dict, status: str, rc: int, interrupted: bool,
                 consistency_note: str = "", retries: int = 0) -> int:
    note = ""
    if status == "error":
        note = f"phase={phase} rc={rc}"
        if interrupted:
            note += "; interrupted"
        note += consistency_note
    if retries:
        retry_note = f"transient_retries={retries}"
        note = f"{retry_note}; {note}" if note else retry_note
    mf.update_case(
        batch_dir, op, status=status, task_dir=str(task_dir.resolve()),
        finished_at=mf.now_iso(), final_phase=phase, rc=rc,
        result=result, note=note,
    )
    sys.stdout.write(
        f"[run] result: op={op} task_dir={task_dir} phase={phase} "
        f"status={status}\n")
    return 130 if interrupted else (0 if status == "done" else 1)


def _run_driver(cmd: list[str], repo_root: Path, batch_dir: Path, op: str,
                hw_arg: str, rounds: int, timeout_min: int, started: float,
                log_fp, launch_name: str, *, agent: str = "",
                line_cb=None):
    agent_tag = f" (agent={agent})" if agent else ""
    header = (
        f"\n{'=' * 72}\n"
        f"[run {datetime.now().isoformat(timespec='seconds')}] op={op} "
        f"{hw_arg} rounds={rounds}{agent_tag}\n"
        f"[run] launching: {launch_name} (cwd={repo_root}, "
        f"timeout={timeout_min}min)\n{'─' * 72}\n")
    _emit(log_fp, header)
    rc, interrupted = _stream_subprocess(
        cmd, str(repo_root), started, timeout_min * 60, log_fp, line_cb,
        extra_env={
            "AR_BATCH_DIR": str(batch_dir.resolve()),
            "AR_BATCH_OP": op,
        },
    )
    footer = (f"{'─' * 72}\n[run] {launch_name} exited rc={rc} after "
              f"{time.time() - started:.0f}s\n")
    _emit(log_fp, footer)
    return rc, interrupted


def run_one(batch_dir: Path, case: dict,
            args: argparse.Namespace, hw_arg: str, log_fp,
            prev_task_dir: Optional[str] = None) -> int:
    context = _begin_case(batch_dir, case, prev_task_dir)
    if context is None:
        return 2
    op, repo_root, started, pre_task_dirs = context
    prompt = build_prompt(case, hw_arg,
                          args.max_rounds, args.eval_timeout)
    cmd = build_claude_cmd(args, prompt)

    timeout_s = args.timeout_min * 60
    last_rc, interrupted = _run_driver(
        cmd, repo_root, batch_dir, op, hw_arg, args.max_rounds,
        args.timeout_min, started, log_fp, f"{args.claude_bin} --print")

    task_dir = _find_task_dir(batch_dir, op, pre_task_dirs)
    if task_dir is None:
        mf.update_case(batch_dir, op,
                       status="error",
                       finished_at=mf.now_iso(),
                       rc=last_rc,
                       note=f"no task_dir found; rc={last_rc}"
                            + ("; interrupted" if interrupted else ""))
        return 130 if interrupted else 2
    phase, result, final_status, consistency_note = _read_case_result(
        task_dir, interrupted)

    # (D) Transient claude-exe retry. claude.exe exits rc != 0 on
    # ECONNRESET / Stream idle timeout / other transient API failures
    # before the Stop hook ever sees the turn — supervisor's only
    # signal is rc + state.json. When framework progress is intact
    # (progress_initialized=True, baseline-ok), re-spawn
    # `claude --print /autoresearch --resume <td> --force` to roll the
    # task forward. Bounded by config.yaml batch.transient_retries.
    # rc=0 + phase != FINISH = LLM ended turn early (real fail, no retry);
    # rc != 0 + no progress = baseline never committed (real fail, no
    # retry).
    transient_attempts = 0
    if (final_status == "error" and last_rc != 0
            and not interrupted and not consistency_note
            and phase != "FINISH"
            and (result or {}).get("progress_initialized") is True):
        max_retries = batch_transient_retries()
        resume_prompt = (
            f"/autoresearch --resume {task_dir} --force"
        )
        resume_cmd = build_claude_cmd(args, resume_prompt)
        while (transient_attempts < max_retries
               and final_status == "error"
               and phase != "FINISH"):
            transient_attempts += 1
            r_started = time.time()
            r_msg = (f"[run] transient claude crash (rc={last_rc}, "
                     f"phase={phase}); resuming attempt "
                     f"{transient_attempts}/{max_retries} via "
                     f"--resume --force\n")
            _emit(log_fp, r_msg)
            r_rc, r_interrupted = _stream_subprocess(
                resume_cmd, str(repo_root), r_started, timeout_s, log_fp,
                extra_env={
                    "AR_BATCH_DIR": str(batch_dir.resolve()),
                    "AR_BATCH_OP": op,
                },
            )
            r_elapsed = time.time() - r_started
            _console_write(f"[run] resume attempt {transient_attempts} "
                           f"exited rc={r_rc} after "
                           f"{r_elapsed:.0f}s\n")
            # Reread phase + state; retain the latest process return code for
            # the final update_case() below.
            try:
                with task_handle.open_task(
                        str(task_dir), role=task_handle.Role.SUPERVISOR):
                    pass
            except task_handle.TaskConsistencyError:
                pass
            phase = mf.read_phase(task_dir)
            result = mf.read_task_state(task_dir)
            last_rc = r_rc
            interrupted = interrupted or r_interrupted
            final_status = ("done" if phase == "FINISH" and not interrupted
                            else "error")
            if r_interrupted:
                break

    return _finish_case(
        batch_dir, op, task_dir, phase, result, final_status, last_rc,
        interrupted, consistency_note, transient_attempts)


def _stream_subprocess(cmd, cwd, started, timeout_s, log_fp, line_cb=None,
                       extra_env: Optional[dict[str, str]] = None):
    """Run `cmd`, tee its combined stdout to console + log, enforce a
    wall-clock cap, and invoke line_cb(line) per line. Reader-thread + queue
    poll so a silent child still hits the deadline (Windows can't select on
    pipes). Returns (returncode, interrupted). This is the agent-neutral
    streaming primitive used by both Claude and OpenCode drivers.

    The child is spawned in its own process group/session so a wall-clock or
    Ctrl-C kill takes down the entire `run_loop.py -> opencode run -> shell ->
    pipeline/build` tree, not just the driver."""
    child_env = env_with_no_proxy(extra_env)
    if os.name == "posix":
        # Bun/OpenCode uses PWD for project-level config discovery.
        child_env["PWD"] = os.path.abspath(cwd)
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
        bufsize=1, env=child_env, **popen_process_group_kwargs(),
    )
    q: "queue.Queue[str]" = queue.Queue()
    done = threading.Event()

    def _reader():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line)
        finally:
            done.set()

    threading.Thread(target=_reader, daemon=True).start()
    interrupted = False
    try:
        while True:
            try:
                line = q.get(timeout=5)
            except queue.Empty:
                if time.time() - started > timeout_s:
                    msg = ("[run] WALL-CLOCK TIMEOUT, killing agent driver "
                           "+ its process tree\n")
                    _emit(log_fp, msg)
                    terminate_process_tree(proc)
                    break
                if done.is_set() and q.empty():
                    break
                continue
            _emit(log_fp, line)
            if line_cb:
                line_cb(line)
        proc.wait(timeout=30)
    except KeyboardInterrupt:
        interrupted = True
        terminate_process_tree(proc)
    return proc.returncode, interrupted


def run_one_opencode(batch_dir: Path, case: dict, args: argparse.Namespace,
                     hw_arg: str, log_fp, prev_task_dir: Optional[str] = None):
    """Drive one op to FINISH with opencode. opencode 1.17.7 has no Stop
    hook, so a single `opencode run` can't self-loop to FINISH like
    `claude --print`. We delegate to the proven headless driver
    `.opencode/run_loop.py`, which scaffolds the task and re-invokes
    `opencode run --session <id>` until the phase machine reaches FINISH. This
    wrapper supplies the same batch bookkeeping run_one does (ownership
    handoff, task_dir binding, post-run heal, status/result recording) so
    both agents share the manifest / queue / summary orchestration verbatim."""
    context = _begin_case(batch_dir, case, prev_task_dir)
    if context is None:
        return 2
    op, repo_root, started, pre_task_dirs = context

    run_loop = repo_root / ".opencode" / "run_loop.py"
    if not run_loop.is_file():
        mf.update_case(batch_dir, op, status="error", finished_at=mf.now_iso(),
                       note=f"opencode driver missing: {run_loop}")
        return 2
    # No iteration cap (symmetric with the claude path): run_loop stops on
    # decide(stop), and this op is bounded by the outer wall-clock timeout
    # below (args.timeout_min), the same one the claude path uses.
    cmd = [sys.executable, str(run_loop),
           "--ref", case["ref"], "--kernel", case["kernel"],
           "--op-name", op,
           "--max-rounds", str(args.max_rounds),
           "--eval-timeout", str(args.eval_timeout)]
    cmd += shlex.split(hw_arg)
    if args.model:
        os.environ["AR_OPENCODE_MODEL"] = args.model

    bound = {"td": None}

    def _cb(line: str):
        # run_loop prints `[run_loop] task_dir=<abs path>` once it resolves
        # the scaffolded task. scaffold.py also writes this into
        # batch_progress when it creates the task.
        if bound["td"] is None:
            s = line.strip()
            marker = "[run_loop] task_dir="
            if s.startswith(marker):
                td = s[len(marker):].strip()
                if td:
                    bound["td"] = td
                    mf.update_case(batch_dir, op,
                                   task_dir=str(Path(td).resolve()))

    rc, interrupted = _run_driver(
        cmd, repo_root, batch_dir, op, hw_arg, args.max_rounds,
        args.timeout_min, started, log_fp, "run_loop.py",
        agent="opencode", line_cb=_cb)

    task_dir = _find_task_dir(
        batch_dir, op, pre_task_dirs, bound["td"])
    if task_dir is None:
        mf.update_case(batch_dir, op, status="error", finished_at=mf.now_iso(),
                       rc=rc,
                       note=f"no task_dir from run_loop; rc={rc}"
                            + ("; interrupted" if interrupted else ""))
        return 130 if interrupted else 2
    phase, result, status, consistency_note = _read_case_result(
        task_dir, interrupted)
    return _finish_case(
        batch_dir, op, task_dir, phase, result, status, rc, interrupted,
        consistency_note)


def filter_queue(progress: dict, args: argparse.Namespace) -> list[dict]:
    statuses = {"pending"}
    if args.retry_errored:
        statuses.add("error")
    only = {s.strip() for s in (args.only or "").split(",") if s.strip()}
    out: list[dict] = []
    for v in progress.get("cases", {}).values():
        if v.get("status") not in statuses:
            continue
        if only and v.get("op_name") not in only:
            continue
        out.append(v)
    return out


def print_summary(batch_dir: Path, total_elapsed: float,
                  hw_arg: str) -> None:
    """Compact end-of-batch report + concrete next-step commands.

    Status lines: just done / error counts (skip / pending only shown when
    nonzero). Speedup distribution collapses into a single line — regress
    cases are part of `done`, not called out separately.

    Next-step commands echo back enough of the original invocation that
    the user can paste directly: batch dir path + the hardware flag we
    were called with. mode is read from the manifest by run.py so we
    don't repeat it.
    """
    progress = mf.load_progress(batch_dir)
    cases = progress.get("cases", {})
    counts = {"done": 0, "error": 0, "skip": 0, "pending": 0, "running": 0}
    speedups: list[float] = []
    for v in cases.values():
        s = v.get("status", "pending")
        counts[s] = counts.get(s, 0) + 1
        if s != "done":
            continue
        r = v.get("result") or {}
        sp = recorded_speedup(r)
        if sp is not None:
            speedups.append(sp)

    print()
    print("=" * 72)
    print(f"[batch done] elapsed={total_elapsed/60:.1f}min")

    if speedups:
        import statistics
        speed_note = (f"  (median {statistics.median(speedups):.2f}x, "
                      f"best {max(speedups):.2f}x, "
                      f"worst {min(speedups):.2f}x; "
                      f"{len(speedups)} with metric)")
    else:
        speed_note = ""
    print(f"  done : {counts['done']}{speed_note}")
    print(f"  error: {counts['error']}")
    if counts["skip"]:
        print(f"  skip : {counts['skip']}")
    if counts["pending"]:
        print(f"  pending: {counts['pending']}")

    # Resolve the batch dir path the way the user is most likely to type it
    # (relative to repo root if it's under there; absolute otherwise).
    repo_root = mf.repo_root()
    try:
        ws_str = str(batch_dir.relative_to(repo_root))
    except ValueError:
        ws_str = str(batch_dir)

    suggestions: list[tuple[str, str]] = []
    if counts["error"]:
        suggestions.append((
            f"retry {counts['error']} errored ops",
            f"python scripts/batch/run.py {ws_str} "
            f"{hw_arg} --retry-errored",
        ))
    if counts["pending"]:
        suggestions.append((
            f"resume {counts['pending']} pending ops",
            f"python scripts/batch/run.py {ws_str} {hw_arg}",
        ))

    if suggestions:
        print()
        print("next steps:")
        for label, cmd in suggestions:
            print(f"  {label}:")
            print(f"    {cmd}")
    print("=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch driver for /autoresearch.")
    ap.add_argument("batch_dir", help="dir containing manifest.yaml/json")
    ap.add_argument("--worker-url", default="",
                    help="Comma-separated worker URLs (host:port). "
                         "When set, eval routes to remote HTTP worker(s) "
                         "and --devices is not required (the orchestrator "
                         "machine can be GPU/NPU-free).")
    ap.add_argument("--devices", default="",
                    help="device ids, e.g. 0 or 0,1; required only for local eval. "
                         "With --worker-url, this is an optional expected-device filter.")
    ap.add_argument("--max-rounds", type=int, default=default_max_rounds())
    ap.add_argument("--eval-timeout", type=int, default=default_eval_timeout(),
                    help="per-shape verify/profile budget in seconds. The "
                         "eval call is capped at eval_timeout * num_cases "
                         "(num_cases comes from get_input_groups() / get_inputs()). "
                         "Single-shape ops keep the original semantics.")
    ap.add_argument("--timeout-min", type=int, default=batch_run_timeout_min(),
                    help="hard wall-clock cap per op in minutes")
    ap.add_argument("--only", default="", help="comma-separated op names")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N ops (0 = no limit)")
    ap.add_argument("--retry-errored", action="store_true",
                    help="also queue ops with status=error")
    ap.add_argument("--cooldown-sec", type=int, default=batch_cooldown_sec(),
                    help="seconds to sleep between ops")
    ap.add_argument("--agent", choices=["claude", "opencode"], default="claude",
                    help="which agent harness drives each op. claude: one "
                         "`claude --print` per op (Stop hook self-loops to "
                         "FINISH). opencode: `.opencode/run_loop.py` re-invokes "
                         "`opencode run --session <id>` to FINISH (opencode has "
                         "no Stop hook). Both share this batch's manifest/queue/"
                         "summary. For opencode set its model + creds in the "
                         "env (AR_OPENCODE_MODEL / DEEPSEEK_API_KEY / …).")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--model", default="",
                    help="model id. claude: --model. opencode: exported as "
                         "AR_OPENCODE_MODEL for run_loop.py.")
    ap.add_argument("--extra-claude-arg", action="append", default=[],
                    help="extra arg to pass to claude (repeatable)")
    args = ap.parse_args()
    # Resolve `claude` -> real executable (Windows needs `claude.cmd`,
    # POSIX returns the absolute path). subprocess.Popen(list, ...) does
    # NOT walk PATHEXT, so a bare 'claude' crashes on Windows even when
    # it's on PATH as a .cmd. (opencode resolves its own bin inside run_loop.)
    if args.agent == "claude":
        args.claude_bin = _resolve_claude_bin(args.claude_bin)

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")

    try:
        manifest_path = mf.find_manifest(batch_dir)
    except mf.ManifestError as e:
        sys.exit(str(e))

    try:
        manifest_data = mf.load_manifest(manifest_path)
    except mf.ManifestError as e:
        sys.exit(f"failed to load {manifest_path}: {e}")

    # ref-kernel is the only supported mode now. Ignore stale manifest.mode
    # values for backward compatibility instead of erroring out.
    mode = "ref-kernel"

    if not args.devices and not args.worker_url:
        sys.exit("--devices (local eval) or --worker-url (remote worker) is required")
    if args.worker_url:
        # Remote-eval path: --devices is optional. When omitted, the worker
        # daemon's /status + DevicePool declare and allocate the actual
        # device; when supplied, it is forwarded as an expected-device filter.
        hw_arg = f"--worker-url {args.worker_url}"
        if args.devices:
            hw_arg = f"--devices {args.devices} {hw_arg}"
    else:
        hw_arg = f"--devices {args.devices}"

    try:
        cases = mf.resolve_cases(batch_dir, manifest_data, mode)
    except mf.ManifestError as e:
        sys.exit(f"manifest validation failed: {e}")

    lock_path = acquire_lock(batch_dir)
    try:
        progress = mf.load_progress(batch_dir)
        demoted, harvested = recover_stale_running(progress)
        progress, dropped = mf.merge_cases(progress, cases, mode)
        mf.save_progress(batch_dir, progress)
        if demoted:
            print(f"[batch] demoted {demoted} stale 'running' op(s) "
                  f"from a previous run -> error")
        if harvested:
            print(f"[batch] harvested {harvested} completed op(s) from "
                  "authoritative task state")
        if dropped:
            preview = ", ".join(dropped[:5]) + (
                f", ... (+{len(dropped) - 5} more)" if len(dropped) > 5 else "")
            print(f"[batch] dropped {len(dropped)} op(s) no longer in manifest: "
                  f"{preview}")

        queue = filter_queue(progress, args)
        if not queue:
            print("nothing to run.")
            return 0
        if args.limit:
            queue = queue[: args.limit]

        print(f"[batch {datetime.now().isoformat(timespec='seconds')}] "
              f"batch_dir={batch_dir}  {hw_arg}\n"
              f"[batch] queue size: {len(queue)}  rounds={args.max_rounds}")

        log_path = batch_dir / mf.LOG_FILENAME
        log_fp = log_path.open("a", encoding="utf-8", buffering=1)

        succeeded = failed = skipped = 0
        total_started = time.time()
        rc_final = 0
        # Carry the previous op's task_dir into the next iteration so
        # run_one can pass it as expected_task_dir to clear_active_task
        # — the only signal we have that the task's owner record
        # belongs to us (a batch op we ourselves drove) versus to an
        # unrelated concurrent session.
        prev_task_dir: Optional[str] = None
        try:
            for i, case in enumerate(queue, 1):
                op = case["op_name"]
                current = filter_queue(mf.load_progress(batch_dir), args)
                if not any(c["op_name"] == op for c in current):
                    print(f"[{i}/{len(queue)}] {op}: status changed underfoot, skipping")
                    skipped += 1
                    continue

                print(f"\n[{i}/{len(queue)}] starting op={op}  "
                      f"elapsed_total={(time.time()-total_started)/60:.1f}min")

                try:
                    driver = (run_one_opencode if args.agent == "opencode"
                              else run_one)
                    rc = driver(batch_dir, case, args, hw_arg, log_fp,
                                prev_task_dir=prev_task_dir)
                    # Refresh prev_task_dir from whatever update_case
                    # eventually wrote (scaffold's batch-progress write or
                    # an agent marker). On abort
                    # without ever binding a task_dir, the field stays
                    # None and the next clear() will fall through to
                    # the heartbeat defence — which is what we want.
                    settled = mf.load_progress(batch_dir).get(
                        "cases", {}).get(op, {}).get("task_dir")
                    if settled:
                        prev_task_dir = settled
                except KeyboardInterrupt:
                    print("\n[batch] Ctrl-C — current op recorded, stopping.")
                    rc_final = 130
                    break

                if rc == 0:
                    succeeded += 1
                elif rc == 130:
                    failed += 1
                    print("\n[batch] op interrupted, stopping.")
                    rc_final = 130
                    break
                else:
                    failed += 1

                print(f"[{i}/{len(queue)}] {op} done rc={rc}  "
                      f"running totals: ok={succeeded} fail={failed} skipped={skipped}")

                if i < len(queue) and args.cooldown_sec > 0:
                    time.sleep(args.cooldown_sec)
        finally:
            log_fp.close()

        print_summary(batch_dir, time.time() - total_started, hw_arg)
        if rc_final:
            return rc_final
        return 0 if failed == 0 else 1
    finally:
        release_lock(lock_path)


if __name__ == "__main__":
    sys.exit(main())
