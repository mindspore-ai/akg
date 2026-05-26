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

"""Live monitor for the batch run.

    python .autoresearch/scripts/batch/monitor.py <batch_dir>
        # auto-refreshing snapshot (default; Ctrl-C to stop)
    python .autoresearch/scripts/batch/monitor.py <batch_dir> --dashboard
        # exec autoresearch's own dashboard.py on the active task (full TUI)

The view shows:
  - queue counts + visual progress bar
  - active task: phase, rounds, baseline/best/speedup, heartbeat age
  - active task: latest 3 history.jsonl decisions + plan.md head
  - tail of batch.log
  - speedup distribution across done ops
  - errored ops summary

For a static, copy-pasteable end-of-batch report use summarize.py.
"""

# pylint: disable=broad-exception-caught,missing-function-docstring,wrong-import-position
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf

DASHBOARD_PY = mf.repo_root() / ".autoresearch" / "scripts" / "dashboard.py"


def task_state(task_dir: Path) -> dict:
    out: dict = {"task_dir": str(task_dir)}
    out["phase"] = mf.read_phase(task_dir)
    for name in ("progress.json", ".progress.json"):
        pf = task_dir / ".ar_state" / name
        if not pf.exists():
            continue
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            for k in ("baseline_metric", "best_metric", "eval_rounds",
                      "max_rounds", "consecutive_failures", "plan_version",
                      "status"):
                if k in data:
                    out[k] = data[k]
        except Exception as e:
            out["_progress_read_error"] = str(e)
        break
    hb = task_dir / ".ar_state" / ".heartbeat"
    if hb.exists():
        try:
            ts = int(hb.read_text(encoding="utf-8").strip())
            out["heartbeat_age_s"] = int(time.time() - ts)
        except Exception:
            pass
    hist = task_dir / ".ar_state" / "history.jsonl"
    if hist.exists():
        try:
            lines = [l for l in hist.read_text(encoding="utf-8").splitlines() if l.strip()]
            out["history_tail"] = [json.loads(l) for l in lines[-3:]]
        except Exception:
            pass
    plan = task_dir / ".ar_state" / "plan.md"
    if plan.exists():
        try:
            ls = [l.rstrip() for l in plan.read_text(encoding="utf-8").splitlines()]
            out["plan_head"] = [l for l in ls if l.strip()][:12]
        except Exception:
            pass
    return out


def tail_lines(path: Path, n: int = 8) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            block = 4096
            data = b""
            pos = size
            while pos > 0 and data.count(b"\n") <= n + 1:
                read = min(block, pos)
                pos -= read
                f.seek(pos)
                data = f.read(read) + data
            text = data.decode("utf-8", errors="replace")
        return text.splitlines()[-n:]
    except Exception:
        return []


def _render_header(batch_dir: Path, progress: dict) -> list:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        f"━━━ batch monitor  {now} ━━━",
        f"batch_dir  {batch_dir}",
        f"mode={progress.get('mode', '?')}  dsl={progress.get('dsl', '?')}",
        "",
    ]


def _render_queue(cases: dict) -> list:
    counts = {"done": 0, "error": 0, "skip": 0, "pending": 0, "running": 0}
    for v in cases.values():
        s = v.get("status", "pending")
        counts[s] = counts.get(s, 0) + 1
    total = sum(counts.values())
    bar = ("█" * counts["done"]
           + "▶" * counts["running"]
           + "▒" * counts["error"]
           + "·" * counts["skip"]
           + " " * counts["pending"])
    return [
        f"queue   total={total:3d}  done={counts['done']:3d}  "
        f"error={counts['error']:3d}  skip={counts['skip']:3d}  "
        f"pending={counts['pending']:3d}  running={counts['running']:3d}",
        f"        [{bar}]",
    ]


def _render_elapsed(cases: dict) -> list:
    """Cumulative elapsed across reboots/sessions (per-op timestamps in
    batch_progress.json)."""
    done_secs = 0.0
    running_secs = 0.0
    now_dt = datetime.now()
    for v in cases.values():
        s = v.get("started_at")
        f = v.get("finished_at")
        if not s:
            continue
        try:
            s_dt = datetime.fromisoformat(s)
            if f:
                done_secs += (datetime.fromisoformat(f) - s_dt).total_seconds()
            elif v.get("status") == "running":
                running_secs += (now_dt - s_dt).total_seconds()
        except (TypeError, ValueError):
            pass
    total_secs = done_secs + running_secs
    return [
        f"elapsed done={done_secs/60:.1f}min  "
        f"current_op={running_secs/60:.1f}min  "
        f"total={total_secs/60:.1f}min ({total_secs/3600:.2f}h)",
        "",
    ]


def _render_active_metrics(active: dict) -> list:
    bm = active.get("baseline_metric")
    best = active.get("best_metric")
    if (isinstance(bm, (int, float)) and isinstance(best, (int, float))
            and best > 0):
        return [f"        baseline={bm:.3f}  best={best:.3f}  "
                f"speedup={bm/best:.2f}x"]
    if bm is not None or best is not None:
        return [f"        baseline={bm}  best={best}"]
    return []


def _render_history_tail(history: list) -> list:
    if not history:
        return []
    out = ["", "        history (last 3 rounds):"]
    for rec in history:
        rnd = rec.get("round", "?")
        dec = rec.get("decision", "?")
        metrics = rec.get("metrics") or {}
        m_short = ""
        for k in ("latency_us", "metric"):
            if k in metrics:
                m_short = f" {k}={metrics[k]}"
                break
        corr = ("" if rec.get("correctness") is None
                else f" correct={rec['correctness']}")
        desc = (rec.get("description") or "")[:50]
        out.append(f"          R{rnd:>2} {dec}{m_short}{corr}  {desc}")
    return out


def _render_active(active: dict | None) -> list:
    if not active:
        return ["active  (no task in ar_tasks/)", ""]
    # task_dir name already starts with `<op>_<ts>_<uuid>`, so the op name
    # is on-screen without an explicit `op=` field.
    out = [
        f"active  {Path(active['task_dir']).name}",
        f"        phase={active.get('phase', '?')}  "
        f"rounds={active.get('eval_rounds', '?')}/"
        f"{active.get('max_rounds', '?')}  "
        f"failures={active.get('consecutive_failures', 0)}  "
        f"plan_v={active.get('plan_version', 0)}  "
        f"status={active.get('status', '?')}",
    ]
    out.extend(_render_active_metrics(active))
    hb = active.get("heartbeat_age_s")
    if hb is not None:
        stale = " (STALE)" if hb > 300 else ""
        out.append(f"        heartbeat: {hb}s ago{stale}")
    out.extend(_render_history_tail(active.get("history_tail") or []))
    plan_head = active.get("plan_head") or []
    if plan_head:
        out.append("")
        out.append("        plan.md head:")
        for line in plan_head[:8]:
            out.append(f"          {line[:90]}")
    out.append("")
    return out


def _render_log_tail(log_tail: list) -> list:
    if not log_tail:
        return []
    out = ["batch.log (last 6 lines):"]
    for line in log_tail:
        out.append(f"  {line[:100]}")
    out.append("")
    return out


def _render_speedup_summary(cases: dict) -> list:
    speedups: list[tuple[str, float]] = []
    for k, v in cases.items():
        if v.get("status") != "done":
            continue
        r = v.get("result") or {}
        bm, best = r.get("baseline_metric"), r.get("best_metric")
        if (isinstance(bm, (int, float))
                and isinstance(best, (int, float)) and best > 0):
            speedups.append((k, bm / best))
    if not speedups:
        return []
    vals = [s for _, s in speedups]
    improved = sum(1 for v in vals if v > 1.05)
    onpar = sum(1 for v in vals if 0.95 <= v <= 1.05)
    regr = sum(1 for v in vals if v < 0.95)
    return [
        f"done speedup  median={statistics.median(vals):.2f}x  "
        f"best={max(vals):.2f}x  worst={min(vals):.2f}x  (n={len(vals)})",
        f"              improved={improved}  on-par={onpar}  regress={regr}",
    ]


def _render_errored_ops(cases: dict) -> list:
    errored = [(k, v) for k, v in cases.items() if v.get("status") == "error"]
    if not errored:
        return []
    out = ["", f"errored ops ({len(errored)}):"]
    for k, v in errored[:5]:
        note = (v.get("note") or "")[:80]
        out.append(f"  - {k}: {note}")
    if len(errored) > 5:
        out.append(f"  ... and {len(errored) - 5} more")
    return out


def render(batch_dir: Path, progress: dict, active: dict | None,
           log_tail: list[str]) -> str:
    cases = progress.get("cases", {})
    out: list[str] = []
    out.extend(_render_header(batch_dir, progress))
    out.extend(_render_queue(cases))
    out.extend(_render_elapsed(cases))
    out.extend(_render_active(active))
    out.extend(_render_log_tail(log_tail))
    out.extend(_render_speedup_summary(cases))
    out.extend(_render_errored_ops(cases))
    return "\n".join(out)


def clear_screen() -> None:
    os.system("cls" if sys.platform == "win32" else "clear")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_dir")
    ap.add_argument("-n", "--interval", type=int, default=15,
                    help="refresh interval in seconds (default: 15)")
    ap.add_argument("--dashboard", action="store_true",
                    help="exec autoresearch's dashboard.py on the active task")
    ap.add_argument("--task-dir", default="",
                    help="for --dashboard: explicit task_dir (default: most recent)")
    args = ap.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")

    if args.dashboard:
        if args.task_dir:
            td = Path(args.task_dir).resolve()
        else:
            # Pull the running case's task_dir from THIS batch's
            # progress.json — not the repo-wide active pointer, which
            # could belong to a sibling batch or a manual session
            # sharing `ar_tasks/`.
            td = mf.find_running_case_task_dir(batch_dir)
            if td is None:
                sys.exit("no running case has a bound task_dir yet "
                         "(scaffolding, or batch already finished). "
                         "Pass --task-dir <path> to attach explicitly.")
        if not DASHBOARD_PY.exists():
            sys.exit(f"dashboard.py not found at {DASHBOARD_PY}")
        print(f"[monitor] launching autoresearch dashboard on {td}")
        cmd = [sys.executable, str(DASHBOARD_PY), str(td), "--watch", "5"]
        os.execvp(cmd[0], cmd)

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    log_path = batch_dir / mf.LOG_FILENAME

    def render_once() -> str:
        progress = mf.load_progress(batch_dir)
        # Scope active state to THIS batch's running case (or None while
        # it's still scaffolding). Earlier the monitor read the repo-wide
        # active-task pointer, which would project a sibling batch's
        # state into this batch's UI when both ran concurrently.
        active_dir = mf.find_running_case_task_dir(batch_dir)
        active = task_state(active_dir) if active_dir else None
        log_tail = tail_lines(log_path, n=6)
        return render(batch_dir, progress, active, log_tail)

    try:
        while True:
            body = render_once()
            footer = (f"\n(refresh every {args.interval}s; Ctrl-C to stop  |  "
                      "full TUI: monitor.py --dashboard  |  "
                      "static report: summarize.py)")
            clear_screen()
            print(body + footer, flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
