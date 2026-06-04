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
from utils.settings import (  # noqa: E402
    default_max_rounds, default_eval_timeout,
    batch_run_timeout_min, batch_cooldown_sec,
)

# Force line-buffered stdout so logs flush in real time when run via nohup.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass
os.environ.setdefault("PYTHONUNBUFFERED", "1")


PROMPT_TEMPLATE = """\
/autoresearch --ref {ref} --kernel {kernel} --op-name {op} {hw} --max-rounds {rounds} --eval-timeout {timeout}

CRITICAL rules — read carefully, this session is non-interactive:

1. After scaffold prints "Task directory created: <path>", your VERY FIRST
   subsequent action MUST be exactly:
       export AR_TASK_DIR="<that path>"
   The double quotes are required so paths with spaces or backslashes
   (e.g. C:\\Users\\Foo Bar\\...) survive shell parsing. The post-Bash
   hook reads AR_TASK_DIR and claims the task into state.json (owner +
   session_id). Every later hook keys off get_task_dir() / state.owner.
   THIS IS THE SINGLE MOST IMPORTANT STEP.

2. The kernel.py we passed via --kernel is a verified seed. Scaffold's
   --run-baseline runs it; on PASS the phase advances to PLAN in
   state.json immediately. Your job is PERFORMANCE OPTIMIZATION via
   PLAN -> EDIT -> VERIFY for the configured max-rounds: propose
   targeted incremental edits to ModelNew (block sizes, memory layout,
   vectorization, fewer DRAM round-trips) and let pipeline.py measure
   the speedup. If baseline fails on the seed, the hook routes to PLAN
   and the first plan items must fix/rewrite the seed kernel.

3. In EDIT phase use the Edit tool (or Write for full rewrites).
   PostToolUse validates kernel.py and auto-advances on pass.

4. Treat hook output as authoritative. Each hook prints the legal next
   action on stderr (or as additionalContext). Hooks gate every script
   to the right phase (e.g. baseline.py runs only in BASELINE); when a
   hook blocks something, the rejection reason is the next step.

5. Keep working through whatever phase the hooks indicate, until the
   framework itself writes phase=FINISH (which only happens when
   eval_rounds reaches max-rounds — settling all current plan items
   triggers REPLAN, not FINISH). The session is fully unattended; the
   orchestrator detects completion by reading state.json's phase. When
   the hooks have nothing more to ask of you, the session ends
   naturally on your last tool call.
"""

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


def recover_stale_running(progress: dict) -> int:
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
      - otherwise the case is a real orphan; demote with a note.
    """
    # Route through the phase_machine facade so the "is this task
    # still live" judgement has one owner (reads state.last_touched).
    import sys as _sys
    _scripts = str(Path(__file__).resolve().parent.parent)
    if _scripts not in _sys.path:
        _sys.path.insert(0, _scripts)
    from phase_machine import is_task_active  # noqa: E402

    cases = progress.get("cases", {})
    n = 0
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
        c["status"] = "error"
        c["finished_at"] = now
        existing = (c.get("note") or "").strip()
        tag = ("stale running, demoted on batch restart "
               "(no live runner_pid, task not active in state.json)")
        c["note"] = f"{existing}; {tag}" if existing else tag
        n += 1
    return n


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


def env_with_no_proxy() -> dict[str, str]:
    env = os.environ.copy()
    extras = "127.0.0.1,localhost"
    existing = env.get("NO_PROXY", "")
    env["NO_PROXY"] = f"{existing},{extras}".strip(",") if existing else extras
    env["no_proxy"] = env["NO_PROXY"]
    env["PYTHONIOENCODING"] = "utf-8"  # propagates UTF-8 to claude --print + its Bash-tool subprocs
    return env


def run_one(batch_dir: Path, case: dict,
            args: argparse.Namespace, hw_arg: str, log_fp,
            prev_task_dir: Optional[str] = None) -> int:
    op = case["op_name"]
    repo_root = mf.repo_root()
    prompt = build_prompt(case, hw_arg,
                          args.max_rounds, args.eval_timeout)
    cmd = build_claude_cmd(args, prompt)

    started = time.time()
    started_iso = mf.now_iso()
    mf.update_case(batch_dir, op,
                   status="running",
                   started_at=started_iso,
                   finished_at=None,
                   task_dir=None,
                   final_phase=None,
                   rc=None,
                   # Record the runner pid so a future recover_stale_
                   # running call can tell "this case is being driven by
                   # a live process" from "this case is an orphan".
                   runner_pid=os.getpid(),
                   note="")

    # Identity-bound task_dir from same-Popen scaffold markers; snapshot
    # is the post-process safety net only.
    pre_task_dirs = mf.snapshot_task_dirs()
    bound_task_dir: Path | None = None

    # Release the previous op's owner record before launching the
    # next claude --print. Ownership lives in <task_dir>/.ar_state/
    # state.json's owner field and the per-session index. Pass our
    # just-finished task_dir as expected_task_dir so the ownership
    # branch fires instead of the heartbeat-fresh defence (the
    # previous op's last hook touched state.last_touched seconds ago
    # and a heartbeat-only check would refuse every legitimate
    # transition). First op of the run (prev_task_dir is None) falls
    # through to the heartbeat defence, which protects against a
    # manual Claude session
    # already running against this checkout.
    from phase_machine import clear_active_task
    if not clear_active_task(expected_task_dir=prev_task_dir):
        sys.stdout.write(f"[run] op={op}: refusing to start — another "
                         f"session is active on this checkout. Stop it "
                         f"before retrying.\n")
        sys.stdout.flush()
        mf.update_case(batch_dir, op, status="error",
                       finished_at=mf.now_iso(),
                       note="aborted: prior owner still active")
        return 2

    header = (f"\n{'=' * 72}\n"
              f"[run {datetime.now().isoformat(timespec='seconds')}] op={op} "
              f"{hw_arg} rounds={args.max_rounds}\n"
              f"[run] launching: {args.claude_bin} --print "
              f"(cwd={repo_root}, timeout={args.timeout_min}min)\n"
              f"{'─' * 72}\n")
    sys.stdout.write(header)
    sys.stdout.flush()
    log_fp.write(header)
    log_fp.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # Force UTF-8 decode for subprocess stdout: claude (and our hooks)
        # write UTF-8; Python's `text=True` default is
        # `locale.getpreferredencoding(False)`, which on a Chinese-locale
        # Windows is GBK and crashes the _reader thread on the first
        # non-ASCII byte. errors='replace' keeps the stream readable even
        # if some downstream tool ever emits malformed bytes.
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        env=env_with_no_proxy(),
    )

    # Background reader thread + bounded queue.get poll. The earlier
    # `for line in proc.stdout` form blocks on readline indefinitely when
    # claude is alive but silent (API retry, deep IO wait), so the
    # wall-clock check inside the loop never fires and `--timeout-min`
    # becomes a no-op. Selectors aren't an option because Windows can't
    # select() on pipe handles, so we use a thread + queue (cross-platform).
    line_q: "queue.Queue[str]" = queue.Queue()
    reader_done = threading.Event()

    def _reader():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line_q.put(line)
        finally:
            reader_done.set()

    threading.Thread(target=_reader, daemon=True).start()

    timeout_s = args.timeout_min * 60
    interrupted = False
    try:
        while True:
            try:
                # Short poll so a silent claude still hits the wall-clock
                # check below within 5s of crossing the deadline.
                line = line_q.get(timeout=5)
            except queue.Empty:
                if time.time() - started > timeout_s:
                    msg = (f"[run] WALL-CLOCK TIMEOUT after "
                           f"{args.timeout_min}min, killing claude\n")
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    log_fp.write(msg)
                    log_fp.flush()
                    proc.kill()
                    break
                if reader_done.is_set() and line_q.empty():
                    break
                continue
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fp.write(line)
            log_fp.flush()
            if bound_task_dir is None:
                td = (mf.parse_scaffold_created_line(line)
                      or mf.parse_scaffold_result_line(line))
                # Reject paths claude might echo from prior context: must
                # be fresh AND a scaffold-formatted dir for THIS op (exact
                # match, not prefix — `op=avg` must not claim avg_pool2d_*).
                if (td is not None
                        and td not in pre_task_dirs
                        and mf.task_dir_belongs_to_op(td.name, op)):
                    bound_task_dir = td
                    mf.update_case(batch_dir, op, task_dir=str(td.resolve()))
        proc.wait(timeout=30)
    except KeyboardInterrupt:
        interrupted = True
        msg = "\n[run] Ctrl-C received, killing claude\n"
        sys.stdout.write(msg)
        log_fp.write(msg)
        try:
            proc.kill()
        except Exception:
            pass

    elapsed = time.time() - started
    footer = (f"{'─' * 72}\n"
              f"[run] claude exited rc={proc.returncode} after {elapsed:.0f}s\n")
    sys.stdout.write(footer)
    log_fp.write(footer)
    log_fp.flush()

    # Final pick: stdout-bound dir wins; snapshot diff is the safety net.
    td = bound_task_dir or mf.pick_new_task_dir(pre_task_dirs, op)
    if td is None:
        mf.update_case(batch_dir, op,
                       status="error",
                       finished_at=mf.now_iso(),
                       rc=proc.returncode,
                       note=f"no task_dir found; rc={proc.returncode}"
                            + ("; interrupted" if interrupted else ""))
        return 130 if interrupted else 2
    task_dir = td

    # Post-op heal + read. The supervisor never owned this task —
    # the claude --print process did, via post_bash activation. But a
    # crash inside claude can leave an in-flight journal that this
    # supervisor's `read_phase` / `read_task_state` would observe as
    # stale fields. Open a Task as SUPERVISOR (heal + check, no
    # claim) so the reads below see the post-replay state.
    import sys as _sys
    _scripts = str(Path(__file__).resolve().parent.parent)
    if _scripts not in _sys.path:
        _sys.path.insert(0, _scripts)
    from task_handle import (open_task as _open_task,
                              Role as _Role,
                              TaskConsistencyError as _Consistency)
    consistency_note = ""
    try:
        with _open_task(str(task_dir), role=_Role.SUPERVISOR):
            pass  # __enter__ ran replay + check
    except _Consistency as e:
        consistency_note = f"; post-run heal refused: {e}"

    phase = mf.read_phase(td)
    result = mf.read_task_state(task_dir)
    final_status = ("done" if phase == "FINISH" and not interrupted
                    and not consistency_note else "error")

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
    if (final_status == "error" and proc.returncode != 0
            and not interrupted and not consistency_note
            and phase != "FINISH"
            and (result or {}).get("progress_initialized") is True):
        from utils.settings import batch_transient_retries as _max_retries
        max_retries = _max_retries()
        resume_prompt = (
            f"/autoresearch --resume {task_dir} --force"
        )
        resume_cmd = build_claude_cmd(args, resume_prompt)
        while (transient_attempts < max_retries
               and final_status == "error"
               and phase != "FINISH"):
            transient_attempts += 1
            r_started = time.time()
            r_msg = (f"[run] transient claude crash (rc={proc.returncode}, "
                     f"phase={phase}); resuming attempt "
                     f"{transient_attempts}/{max_retries} via "
                     f"--resume --force\n")
            sys.stdout.write(r_msg); log_fp.write(r_msg)
            sys.stdout.flush(); log_fp.flush()
            r_proc = subprocess.Popen(
                resume_cmd,
                cwd=str(repo_root),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                bufsize=1, env=env_with_no_proxy(),
            )
            r_q: "queue.Queue[str]" = queue.Queue()
            r_done = threading.Event()
            def _r_reader(p=r_proc, q=r_q, done=r_done):
                try:
                    assert p.stdout is not None
                    for line in p.stdout:
                        q.put(line)
                finally:
                    done.set()
            threading.Thread(target=_r_reader, daemon=True).start()
            r_interrupted = False
            try:
                while True:
                    try:
                        line = r_q.get(timeout=5)
                    except queue.Empty:
                        if time.time() - r_started > timeout_s:
                            sys.stdout.write(f"[run] WALL-CLOCK TIMEOUT after "
                                             f"{args.timeout_min}min on "
                                             f"resume attempt "
                                             f"{transient_attempts}, killing\n")
                            r_proc.kill()
                            break
                        if r_done.is_set() and r_q.empty():
                            break
                        continue
                    sys.stdout.write(line); log_fp.write(line)
                    sys.stdout.flush(); log_fp.flush()
                r_proc.wait(timeout=30)
            except KeyboardInterrupt:
                r_interrupted = True
                r_proc.kill()
            r_elapsed = time.time() - r_started
            sys.stdout.write(f"[run] resume attempt {transient_attempts} "
                             f"exited rc={r_proc.returncode} after "
                             f"{r_elapsed:.0f}s\n")
            # Reread phase + state; rebind proc so the final
            # update_case() below sees the latest rc.
            try:
                with _open_task(str(task_dir), role=_Role.SUPERVISOR):
                    pass
            except _Consistency:
                pass
            phase = mf.read_phase(task_dir)
            result = mf.read_task_state(task_dir)
            proc = r_proc
            interrupted = interrupted or r_interrupted
            final_status = ("done" if phase == "FINISH" and not interrupted
                            else "error")
            if r_interrupted:
                break

    note = ""
    if final_status == "error":
        note = f"phase={phase} rc={proc.returncode}"
        if interrupted:
            note += "; interrupted"
        if consistency_note:
            note += consistency_note
    if transient_attempts > 0:
        rt_tag = f"transient_retries={transient_attempts}"
        note = f"{rt_tag}; {note}" if note else rt_tag

    mf.update_case(batch_dir, op,
                   status=final_status,
                   task_dir=str(task_dir.resolve()),
                   finished_at=mf.now_iso(),
                   final_phase=phase,
                   rc=proc.returncode,
                   result=result,
                   note=note)

    sys.stdout.write(
        f"[run] result: op={op} task_dir={task_dir} phase={phase} "
        f"status={final_status}\n"
    )
    if interrupted:
        return 130
    return 0 if final_status == "done" else 1


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
        bm, best = r.get("baseline_metric"), r.get("best_metric")
        if isinstance(bm, (int, float)) and isinstance(best, (int, float)) and best > 0:
            speedups.append(bm / best)

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
                    help="NPU device ids, e.g. 0 or 0,1 (required)")
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
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--model", default="")
    ap.add_argument("--extra-claude-arg", action="append", default=[],
                    help="extra arg to pass to claude (repeatable)")
    args = ap.parse_args()
    # Resolve `claude` -> real executable (Windows needs `claude.cmd`,
    # POSIX returns the absolute path). subprocess.Popen(list, ...) does
    # NOT walk PATHEXT, so a bare 'claude' crashes on Windows even when
    # it's on PATH as a .cmd.
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
        # Remote-eval path: still forward --devices to /autoresearch if the
        # caller supplied one (it's a required slash-command arg and gets
        # written into task.yaml for arch derivation), but bake a placeholder
        # `0` when only --worker-url was given. The actual NPU id the eval
        # runs on is decided by the worker daemon's device pool, not by this
        # CLI.
        dev_part = args.devices or "0"
        hw_arg = f"--devices {dev_part} --worker-url {args.worker_url}"
    else:
        hw_arg = f"--devices {args.devices}"

    try:
        cases = mf.resolve_cases(batch_dir, manifest_data, mode)
    except mf.ManifestError as e:
        sys.exit(f"manifest validation failed: {e}")

    lock_path = acquire_lock(batch_dir)
    try:
        progress = mf.load_progress(batch_dir)
        demoted = recover_stale_running(progress)
        progress, dropped = mf.merge_cases(progress, cases, mode)
        mf.save_progress(batch_dir, progress)
        if demoted:
            print(f"[batch] demoted {demoted} stale 'running' op(s) "
                  f"from a previous run -> error")
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
                    rc = run_one(batch_dir, case, args, hw_arg, log_fp,
                                 prev_task_dir=prev_task_dir)
                    # Refresh prev_task_dir from whatever update_case
                    # eventually wrote (run_one resolves the task_dir
                    # mid-run via scaffold markers). On run_one abort
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
