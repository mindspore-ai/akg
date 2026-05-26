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
    python .autoresearch/scripts/batch/run.py <batch_dir> \\
        [--dsl triton_ascend] \\
        [--devices N | --worker-url host:port] \\
        [--max-rounds 30] [--eval-timeout 600] [--timeout-min 180] \\
        [--only op1,op2] [--limit N] [--retry-errored] [--cooldown-sec 5]
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring,wrong-import-position
from __future__ import annotations

import argparse
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf

# Force line-buffered stdout so logs flush in real time when run via nohup.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass
os.environ.setdefault("PYTHONUNBUFFERED", "1")


PROMPT_TEMPLATE = """\
/autoresearch --ref {ref} --kernel {kernel} --op-name {op} --dsl {dsl} {hw} --max-rounds {rounds} --eval-timeout {timeout}

CRITICAL rules — read carefully, this session is non-interactive:

1. After scaffold prints "Task directory created: <path>", your VERY FIRST
   subsequent action MUST be exactly:
       export AKG_AGENTS_AR_TASK_DIR="<that path>"
   The double quotes are required so paths with spaces or backslashes
   (e.g. C:\\Users\\Foo Bar\\...) survive shell parsing. This single
   command writes .autoresearch/.active_task, which activates the hook
   chain — every PostToolUse gate keys off that file. THIS IS THE
   SINGLE MOST IMPORTANT STEP.

2. The kernel.py we passed via --kernel is a verified seed. Scaffold's
   --run-baseline runs it; on PASS .ar_state/.phase is set to PLAN
   immediately. Your job is PERFORMANCE OPTIMIZATION via
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
   orchestrator detects completion by reading .ar_state/.phase. When
   the hooks have nothing more to ask of you, the session ends
   naturally on your last tool call.
"""

def health_check_worker(worker_url: str) -> None:
    """Probe http://<host>:<port>/api/v1/status. Raises SystemExit on failure."""
    if "://" not in worker_url:
        url = f"http://{worker_url}/api/v1/status"
    else:
        url = worker_url.rstrip("/") + "/api/v1/status"
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status != 200:
                raise urllib.error.URLError(f"HTTP {resp.status}")
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        sys.exit(
            f"\nworker daemon at {worker_url} is unreachable ({e}).\n"
            f"start it on the worker host:\n"
            "    akg_cli worker --start --backend ascend --arch ascend910b3 "
            f"--devices 0 --port {worker_url.split(':')[-1] or '9111'}\n"
            f"or pass --devices N to use in-process eval (slower for batch runs).\n"
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
    Stale locks (PID gone) are auto-cleared; live locks abort with a hint."""
    lock = batch_dir / LOCK_FILENAME
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = -1
        if pid > 0 and _pid_alive(pid):
            sys.exit(
                f"\nanother batch run is active on this batch dir "
                f"(pid={pid}, lock={lock}).\n"
                f"if you're sure no run.py is running, remove {lock} and retry.\n"
            )
        # stale lock — overwrite below
    lock.write_text(str(os.getpid()), encoding="utf-8")
    return lock


def release_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except OSError:
        pass


def recover_stale_running(progress: dict) -> int:
    """Demote any 'running' cases to 'error'. We hold the batch dir lock by
    the time this is called, so anything still 'running' is an orphan from a
    previous run.py that died (SIGKILL, OOM, machine reboot)."""
    cases = progress.get("cases", {})
    n = 0
    now = mf.now_iso()
    for c in cases.values():
        if c.get("status") == "running":
            c["status"] = "error"
            c["finished_at"] = now
            existing = (c.get("note") or "").strip()
            tag = "stale running, demoted on batch restart"
            c["note"] = f"{existing}; {tag}" if existing else tag
            n += 1
    return n


def build_prompt(case: dict, dsl: str, hw_arg: str,
                 max_rounds: int, eval_timeout: int) -> str:
    """Quote every value-bearing flag with shlex.quote so paths with
    spaces (e.g. batch dir under `C:\\Users\\Foo Bar\\...`, or
    `--output-dir "my tasks"`) reach /autoresearch as one argv each.
    `hw_arg` is constructed by the caller from already-validated CLI
    flags — pass through unchanged."""
    return PROMPT_TEMPLATE.format(
        ref=shlex.quote(case["ref"]),
        kernel=shlex.quote(case["kernel"]),
        op=shlex.quote(case["op_name"]),
        dsl=shlex.quote(dsl),
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
    return env


def _start_claude_proc(cmd: list[str], repo_root: Path):
    return subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env_with_no_proxy(),
    )


def _try_bind_task_dir(line: str, pre_task_dirs: set,
                       op: str) -> Path | None:
    """Parse one claude-stdout line for the scaffold marker / result
    JSON, returning the bound task_dir or None. Rejects paths from
    prior context: must be fresh AND match THIS op's
    `<op>_<ts>_<hex6>` shape exactly (`op=avg` must not claim
    avg_pool2d_*)."""
    td = (mf.parse_scaffold_created_line(line)
          or mf.parse_scaffold_result_line(line))
    if (td is not None
            and td not in pre_task_dirs
            and mf.task_dir_belongs_to_op(td.name, op)):
        return td
    return None


def _stream_claude_output(proc, log_fp, op: str, batch_dir: Path,
                          pre_task_dirs: set, started: float,
                          timeout_min: int) -> tuple:
    """Drain proc.stdout via a background reader thread + bounded queue
    poll. The naive `for line in proc.stdout` blocks on readline
    indefinitely when claude is alive but silent (API retry, deep IO
    wait), making --timeout-min a no-op. select() isn't an option
    because Windows can't select on pipe handles. Returns
    (bound_task_dir, interrupted)."""
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

    timeout_s = timeout_min * 60
    bound_task_dir: Path | None = None
    interrupted = False
    try:
        while True:
            try:
                line = line_q.get(timeout=5)
            except queue.Empty:
                if time.time() - started > timeout_s:
                    msg = ("[run] WALL-CLOCK TIMEOUT after "
                           f"{timeout_min}min, killing claude\n")
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
                td = _try_bind_task_dir(line, pre_task_dirs, op)
                if td is not None:
                    bound_task_dir = td
                    mf.update_case(batch_dir, op,
                                   task_dir=str(td.resolve()))
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
    return bound_task_dir, interrupted


def _finalize_run(batch_dir: Path, op: str, proc,
                  bound_task_dir: Path | None, pre_task_dirs: set,
                  interrupted: bool) -> int:
    """Record final case state. Returns rc this run reports."""
    td = bound_task_dir or mf.pick_new_task_dir(pre_task_dirs, op)
    if td is None:
        mf.update_case(batch_dir, op,
                       status="error",
                       finished_at=mf.now_iso(),
                       rc=proc.returncode,
                       note=f"no task_dir found; rc={proc.returncode}"
                            + ("; interrupted" if interrupted else ""))
        return 130 if interrupted else 2
    phase = mf.read_phase(td)
    result = mf.read_task_state(td)
    final_status = ("done" if phase == "FINISH" and not interrupted
                    else "error")
    note = ""
    if final_status == "error":
        note = f"phase={phase} rc={proc.returncode}"
        if interrupted:
            note += "; interrupted"
    mf.update_case(batch_dir, op,
                   status=final_status,
                   task_dir=str(td.resolve()),
                   finished_at=mf.now_iso(),
                   final_phase=phase,
                   rc=proc.returncode,
                   result=result,
                   note=note)
    sys.stdout.write(
        f"[run] result: op={op} task_dir={td} phase={phase} "
        f"status={final_status}\n"
    )
    if interrupted:
        return 130
    return 0 if final_status == "done" else 1


def run_one(batch_dir: Path, case: dict, args: argparse.Namespace,
            dsl: str, hw_arg: str, log_fp) -> int:
    op = case["op_name"]
    repo_root = mf.repo_root()
    prompt = build_prompt(case, dsl, hw_arg,
                          args.max_rounds, args.eval_timeout)
    cmd = build_claude_cmd(args, prompt)

    started = time.time()
    mf.update_case(batch_dir, op,
                   status="running",
                   started_at=mf.now_iso(),
                   finished_at=None,
                   task_dir=None,
                   final_phase=None,
                   rc=None,
                   note="")

    # Clear stale .active_task left by a prior op / batch / interactive
    # session. The activation hook reads this file on claude startup and
    # silently resumes the pointed task, bypassing /autoresearch's
    # scaffold dispatch.
    (repo_root / ".autoresearch" / ".active_task").unlink(missing_ok=True)
    pre_task_dirs = mf.snapshot_task_dirs()

    header = (f"\n{'=' * 72}\n"
              f"[run {datetime.now().isoformat(timespec='seconds')}] "
              f"op={op} {hw_arg} rounds={args.max_rounds}\n"
              f"[run] launching: {args.claude_bin} --print "
              f"(cwd={repo_root}, timeout={args.timeout_min}min)\n"
              f"{'─' * 72}\n")
    sys.stdout.write(header)
    sys.stdout.flush()
    log_fp.write(header)
    log_fp.flush()

    proc = _start_claude_proc(cmd, repo_root)
    bound_task_dir, interrupted = _stream_claude_output(
        proc, log_fp, op, batch_dir, pre_task_dirs, started,
        args.timeout_min,
    )

    elapsed = time.time() - started
    footer = (f"{'─' * 72}\n"
              f"[run] claude exited rc={proc.returncode} after "
              f"{elapsed:.0f}s\n")
    sys.stdout.write(footer)
    log_fp.write(footer)
    log_fp.flush()

    return _finalize_run(batch_dir, op, proc, bound_task_dir,
                         pre_task_dirs, interrupted)


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
    were called with. mode / dsl are read from the manifest by run.py so
    we don't repeat them.
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
            f"python .autoresearch/scripts/batch/run.py {ws_str} "
            f"{hw_arg} --retry-errored",
        ))
    if counts["pending"]:
        suggestions.append((
            f"resume {counts['pending']} pending ops",
            f"python .autoresearch/scripts/batch/run.py {ws_str} {hw_arg}",
        ))

    if suggestions:
        print()
        print("next steps:")
        for label, cmd in suggestions:
            print(f"  {label}:")
            print(f"    {cmd}")
    print("=" * 72)


def _make_run_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch driver for /autoresearch.")
    ap.add_argument("batch_dir", help="dir containing manifest.yaml/json")
    ap.add_argument("--dsl", default="",
                    help="DSL passed to /autoresearch (overrides manifest.dsl)")
    ap.add_argument("--devices", default="",
                    help="NPU device ids, e.g. 0 or 0,1; mutually exclusive "
                         "with --worker-url")
    ap.add_argument("--worker-url", default="",
                    help="autoresearch worker URL; default 127.0.0.1:9111 if "
                         "neither --devices nor --worker-url is given")
    ap.add_argument("--max-rounds", type=int, default=30)
    ap.add_argument("--eval-timeout", type=int, default=600)
    ap.add_argument("--timeout-min", type=int, default=180,
                    help="hard wall-clock cap per op in minutes")
    ap.add_argument("--only", default="", help="comma-separated op names")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N ops (0 = no limit)")
    ap.add_argument("--retry-errored", action="store_true",
                    help="also queue ops with status=error")
    ap.add_argument("--cooldown-sec", type=int, default=5,
                    help="seconds to sleep between ops")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--model", default="")
    ap.add_argument("--extra-claude-arg", action="append", default=[],
                    help="extra arg to pass to claude (repeatable)")
    return ap


def _resolve_hw_arg(args: argparse.Namespace) -> str:
    """Return the `--devices N` / `--worker-url URL` flag to pass downstream
    to /autoresearch. Health-checks worker URLs (local --devices is a
    direct subprocess and needs no probe). Exits on mutual-exclusion
    violation or worker unreachable."""
    if args.devices and args.worker_url:
        sys.exit("--devices and --worker-url are mutually exclusive")
    if args.devices:
        return f"--devices {args.devices}"
    worker_url = args.worker_url or "127.0.0.1:9111"
    health_check_worker(worker_url)
    return f"--worker-url {worker_url}"


def _prep_batch_progress(batch_dir: Path, cases: list[dict], mode: str,
                         dsl: str) -> dict:
    """Load progress, demote stale 'running' rows, merge fresh cases.
    Logs the demote / drop counts so the user sees the batch state
    change. Returns the merged progress dict (also written to disk)."""
    progress = mf.load_progress(batch_dir)
    demoted = recover_stale_running(progress)
    progress, dropped = mf.merge_cases(progress, cases, mode, dsl)
    mf.save_progress(batch_dir, progress)
    if demoted:
        print(f"[batch] demoted {demoted} stale 'running' op(s) from a "
              "previous run -> error")
    if dropped:
        preview = ", ".join(dropped[:5]) + (
            f", ... (+{len(dropped) - 5} more)" if len(dropped) > 5 else "")
        print(f"[batch] dropped {len(dropped)} op(s) no longer in "
              f"manifest: {preview}")
    return progress


def _run_queue_loop(op_queue: list[dict], batch_dir: Path,
                    args: argparse.Namespace, dsl: str, hw_arg: str,
                    log_fp) -> tuple:
    """Drive the per-op loop. Returns (failed, rc_final, total_started).
    rc_final is 130 if interrupted, else 0."""
    succeeded = failed = skipped = 0
    total_started = time.time()
    rc_final = 0
    for i, case in enumerate(op_queue, 1):
        op = case["op_name"]
        current = filter_queue(mf.load_progress(batch_dir), args)
        if not any(c["op_name"] == op for c in current):
            print(f"[{i}/{len(op_queue)}] {op}: status changed underfoot, "
                  "skipping")
            skipped += 1
            continue
        print(f"\n[{i}/{len(op_queue)}] starting op={op}  "
              "elapsed_total="
              f"{(time.time() - total_started) / 60:.1f}min")
        try:
            rc = run_one(batch_dir, case, args, dsl, hw_arg, log_fp)
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
        print(f"[{i}/{len(op_queue)}] {op} done rc={rc}  running totals: "
              f"ok={succeeded} fail={failed} skipped={skipped}")
        if i < len(op_queue) and args.cooldown_sec > 0:
            time.sleep(args.cooldown_sec)
    return failed, rc_final, total_started


def main() -> int:
    args = _make_run_parser().parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")

    try:
        manifest_path = mf.find_manifest(batch_dir)
        manifest_data = mf.load_manifest(manifest_path)
    except mf.ManifestError as e:
        sys.exit(str(e))

    # ref-kernel is the only supported mode now. Ignore stale
    # manifest.mode values for backward compatibility.
    mode = "ref-kernel"
    dsl = args.dsl or manifest_data.get("dsl") or ""
    if not dsl:
        sys.exit("--dsl is required (also accepted as `dsl:` in manifest)")
    hw_arg = _resolve_hw_arg(args)

    try:
        cases = mf.resolve_cases(batch_dir, manifest_data, mode)
    except mf.ManifestError as e:
        sys.exit(f"manifest validation failed: {e}")

    lock_path = acquire_lock(batch_dir)
    try:
        progress = _prep_batch_progress(batch_dir, cases, mode, dsl)
        op_queue = filter_queue(progress, args)
        if not op_queue:
            print("nothing to run.")
            return 0
        if args.limit:
            op_queue = op_queue[: args.limit]

        print(f"[batch {datetime.now().isoformat(timespec='seconds')}] "
              f"batch_dir={batch_dir}  mode={mode}  dsl={dsl}  {hw_arg}\n"
              f"[batch] queue size: {len(op_queue)}  "
              f"rounds={args.max_rounds}")

        log_path = batch_dir / mf.LOG_FILENAME
        log_fp = log_path.open("a", encoding="utf-8", buffering=1)
        try:
            failed, rc_final, total_started = _run_queue_loop(
                op_queue, batch_dir, args, dsl, hw_arg, log_fp)
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
