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

"""Live dashboard for autoresearch progress."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_machine as _pm
from utils.json_io import load_jsonl as _shared_load_jsonl
from utils.json_io import _read_whole_file as _shared_read_whole_file
from utils.settings import default_max_rounds as _default_max_rounds

# ---------------------------------------------------------------------------
# Non-blocking keyboard input (cross-platform)
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import msvcrt

    def read_key_nonblocking():
        """Return key name or None. Handles arrows/page keys via escape prefix."""
        if not msvcrt.kbhit():
            return None
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):  # Arrow/function key prefix
            if not msvcrt.kbhit():
                return None
            code = msvcrt.getch()
            return {
                b"H": "UP", b"P": "DOWN",
                b"I": "PGUP", b"Q": "PGDN",
                b"G": "HOME", b"O": "END",
            }.get(code)
        if ch == b"\x1b":
            return "ESC"
        if ch == b"q":
            return "QUIT"
        return None

    _old_tty = None

    def setup_keyboard(): pass
    def restore_keyboard(): pass

else:
    import select
    import termios
    import tty

    _old_tty = None

    def setup_keyboard():
        global _old_tty
        _old_tty = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def restore_keyboard():
        if _old_tty:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _old_tty)

    def read_key_nonblocking():
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Could be ESC or arrow key sequence
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not r:
                return "ESC"
            seq = sys.stdin.read(2)
            return {
                "[A": "UP", "[B": "DOWN",
                "[5": "PGUP", "[6": "PGDN",
                "[H": "HOME", "[F": "END",
            }.get(seq)
        if ch == "q":
            return "QUIT"
        return None

# ANSI colors
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RESET = "\033[0m"


_read_raw = _shared_read_whole_file  # canonical loader lives in utils.json_io


def load_json(path):
    if not os.path.exists(path):
        return None
    return json.loads(_read_raw(path))


load_jsonl = _shared_load_jsonl


def load_plan(path):
    if not os.path.exists(path):
        return "(no plan yet)", None
    mtime = os.path.getmtime(path)
    return _read_raw(path), mtime


def bar(fraction, width=30):
    filled = int(fraction * width)
    return f"[{'#' * filled}{'.' * (width - filled)}]"


# Visible prefix widths for the two table rows (ANSI colour codes excluded).
# History row:  "  {rnd:>3}  │ {dec:8} │ {metric:>13} │ "
#               2 + 3 + 4 + 8 + 3 + 13 + 3  = 36
# Plan row:     "  {item_id:>4}  │ {status:9} │ "
#               2 + 4 + 4 + 9 + 2 = 21
_HIST_PREFIX_VIS = 36
_PLAN_PREFIX_VIS = 21


def _fit(text: str, avail: int) -> str:
    """Truncate with single-char ellipsis only when the text would overflow
    the available column width. Every description column across the dashboard
    routes through this helper so behaviour stays consistent — render as much
    as the terminal can fit, truncate just enough to land in the column.
    """
    if avail <= 0:
        return ""
    if len(text) <= avail:
        return text
    if avail == 1:
        return "…"
    return text[: avail - 1] + "…"


def render(task_dir, history_offset=0, history_window=None):
    """Render dashboard.

    history_offset: how many rounds to skip from the END (0 = latest).
    history_window: how many rounds to show (None = auto based on terminal height).
    """
    # Progress fields now live in state.json (single per-task record).
    # load_state returns the Progress dict embedded in state.json,
    # plus the new control fields (owner / phase / pending_settle).
    # The dashboard only reads progress-related keys, so this swap is
    # one-line.
    progress = _pm.load_state(task_dir)
    history_all = load_jsonl(_pm.history_path(task_dir))
    plan_text, plan_mtime = load_plan(_pm.plan_path(task_dir))

    # Get terminal width for responsive layout. Tables render as wide as the
    # terminal allows; descriptions are truncated only when necessary.
    try:
        term_width = os.get_terminal_size().columns
    except Exception:
        term_width = 100
    hist_desc_avail = max(10, term_width - _HIST_PREFIX_VIS - 2)
    plan_desc_avail = max(10, term_width - _PLAN_PREFIX_VIS - 2)
    divider_width = max(40, term_width - 2)

    lines = []
    lines.append(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗{RESET}")
    lines.append(f"{BOLD}{CYAN}║          AUTORESEARCH DASHBOARD                             ║{RESET}")
    lines.append(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════╝{RESET}")

    if progress is None:
        lines.append(f"\n  {RED}No state.json found at "
                     f"{_pm.state_record_path(task_dir)}{RESET}")
        lines.append(f"  Run /autoresearch --ref ... --op-name ... first.")
        return "\n".join(lines)

    # progress_initialized=False means a session has claimed this task
    # but baseline has not yet committed any measurement. Showing the
    # default zero/None scaffold values here would look like genuine
    # data ("Round 0/999, Baseline None"); render a fresh-task banner
    # instead. Once baseline lands (sets progress_initialized=True),
    # the normal layout below takes over.
    if not progress.get("progress_initialized"):
        owner = progress.get("owner") or {}
        lines.append(f"\n  {BOLD}{CYAN}Task scaffolded; baseline not yet "
                     f"run.{RESET}")
        if owner.get("session_id"):
            lines.append(f"  Owner session: {DIM}"
                         f"{owner.get('session_id')}{RESET}")
        lines.append(f"  {DIM}This dashboard will populate once "
                     f"baseline.py commits the first measurement.{RESET}")
        return "\n".join(lines)

    task = progress.get("task", "?")
    rounds = progress.get("eval_rounds", 0)
    # Fallback for incomplete state files missing the field — defer to
    # settings so the display stays consistent with what the scheduler
    # actually uses.
    max_rounds = progress.get("max_rounds", _default_max_rounds())
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    best_commit = progress.get("best_commit", "?")
    failures = progress.get("consecutive_failures", 0)
    plan_ver = progress.get("plan_version", 0)
    # Status was a redundant field on Progress that just tracked "has a
    # plan been validated yet?". Derive it here from plan.md presence so
    # the dashboard label stays identical for users.
    status = "active" if os.path.exists(_pm.plan_path(task_dir)) else "no_plan"
    updated_raw = progress.get("last_updated", "?")
    # Convert UTC to local time
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(updated_raw)
        if dt.tzinfo is not None:
            dt = dt.astimezone()  # Convert to local timezone
        updated = dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        updated = updated_raw

    # Improvement (baseline is PyTorch reference latency; best is kernel latency)
    if best is not None and baseline is not None and baseline != 0 and best != 0:
        improv_pct = (baseline - best) / abs(baseline) * 100
        speedup = baseline / best
        color = GREEN if improv_pct > 0 else RED
        # Anchor is always "ref" for committed tasks (the baseline gate
        # refuses to commit without a valid PyTorch reference).
        src = progress.get("baseline_source")
        anchor_label = "vs ref" if src == "ref" else "vs baseline"
        improv_str = f"{color}{speedup:.2f}x {anchor_label} ({improv_pct:+.1f}%){RESET}"
    else:
        improv_str = f"{DIM}N/A{RESET}"

    frac = rounds / max_rounds if max_rounds > 0 else 0
    budget_bar = bar(frac)
    budget_color = GREEN if frac < 0.5 else (YELLOW if frac < 0.8 else RED)

    status_color = GREEN if status == "active" else (YELLOW if status == "no_plan" else CYAN)
    fail_color = RED if failures >= 3 else (YELLOW if failures > 0 else GREEN)

    lines.append("")
    lines.append(f"  {BOLD}Task:{RESET}     {task}")
    lines.append(f"  {BOLD}Status:{RESET}   {status_color}{status}{RESET}  (plan v{plan_ver})")
    lines.append(f"  {BOLD}Updated:{RESET}  {DIM}{updated}{RESET}")

    # Task-level abort banner. Fires only for INFRA_FAIL (agent can't fix
    # from EDIT). The error_source axis picks ref-broken vs other.
    outcome = progress.get("baseline_outcome")
    err_src = progress.get("baseline_error_source") or ""
    if outcome == "infra_fail":
        lines.append("")
        if err_src == "ref":
            lines.append(f"  {BOLD}{RED}ABORTED:{RESET}  {RED}REF BROKEN{RESET}"
                         f"  reference.py is invalid.")
            lines.append(f"           {DIM}Fix the source --ref file and "
                         f"re-run /autoresearch from scratch.{RESET}")
        else:
            lines.append(f"  {BOLD}{YELLOW}ABORTED:{RESET}  "
                         f"{YELLOW}EVAL PIPELINE BROKEN{RESET}  "
                         f"no per-shape data produced.")
            lines.append(f"           {DIM}Check device / env / eval.timeout, "
                         f"then retry baseline.py.{RESET}")

    lines.append("")
    lines.append(f"  {BOLD}Budget:{RESET}   {budget_color}{budget_bar} {rounds}/{max_rounds}{RESET}")

    # Baseline (PyTorch reference timing). Always present on committed
    # tasks — the gate refuses to commit without a valid ref.
    baseline_tags = {
        "ref": f"{DIM}(PyTorch reference){RESET}",
    }
    if baseline is None:
        lines.append(f"  {BOLD}Baseline:{RESET} {DIM}— (not measured){RESET}")
    else:
        baseline_tag = baseline_tags.get(progress.get("baseline_source"),
                                          f"{DIM}(source unknown){RESET}")
        lines.append(f"  {BOLD}Baseline:{RESET} {baseline}  {baseline_tag}")

    # Seed (initial kernel timing).
    seed = progress.get("seed_metric")
    if seed is not None:
        if seed != baseline:
            lines.append(f"  {BOLD}Seed:{RESET}     {seed}  {DIM}(initial kernel){RESET}")
    elif outcome == "kernel_fail":
        lines.append(f"  {BOLD}Seed:{RESET}     {RED}FAILED{RESET}  "
                     f"{DIM}(kernel verify or profile failed; timing dropped){RESET}")
    else:
        lines.append(f"  {BOLD}Seed:{RESET}     {DIM}— (no timing recorded){RESET}")
    lines.append(f"  {BOLD}Best:{RESET}     {GREEN}{best}{RESET}  ({improv_str})")
    lines.append(f"  {BOLD}Commit:{RESET}   {best_commit}")
    lines.append(f"  {BOLD}Failures:{RESET} {fail_color}{failures}{RESET} consecutive" +
                 (f"  {RED}⚠ DIAGNOSIS WILL TRIGGER{RESET}" if failures >= 3 else ""))

    # History table — windowed view
    n_total = len(history_all)
    if history_window is None:
        # Auto: reserve ~25 lines for header+plan, use rest for history
        try:
            term_h = os.get_terminal_size().lines
        except Exception:
            term_h = 40
        history_window = max(5, term_h - 28)

    # offset=0 means show latest; offset=5 means skip last 5 rounds
    history_offset = max(0, min(history_offset, max(0, n_total - history_window)))
    end = n_total - history_offset
    start = max(0, end - history_window)
    history = history_all[start:end]

    scroll_info = ""
    if n_total > history_window:
        scroll_info = f" [{start+1}-{end} of {n_total}, ↑↓ PgUp/PgDn Home/End q=quit]"

    lines.append("")
    lines.append(f"  {BOLD}History{RESET}{DIM}{scroll_info}{RESET}")
    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")
    lines.append(f"  {BOLD}  #  │ Decision │ Metric        │ Description{RESET}")
    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")

    for rec in history:
        rnd = rec.get("round")
        rnd = "?" if rnd is None else str(rnd)
        decision = rec.get("decision", "?")
        metrics = rec.get("metrics", {})
        raw_desc = rec.get("description", "")
        pid = rec.get("plan_item")
        # Prefix description with plan-item id when we have one, so every row
        # is unambiguously traceable back to plan.md. Older rounds (pre-fix)
        # may lack plan_item; render without the prefix.
        if pid:
            desc = f"{pid}: {raw_desc}"
        else:
            desc = raw_desc
        desc = _fit(desc, hist_desc_avail)

        # Find primary metric value. Limited to known performance keys -
        # earlier we fell back to `next(iter(metrics.values()))` when neither
        # was present, which on FAIL rows displayed `num_cases` (e.g. 72) as
        # if it were the kernel's latency. FAIL rows now show "—" cleanly.
        metric_val = "—"
        for k in ["latency_us", "score"]:
            if k in metrics and metrics[k] is not None:
                metric_val = f"{metrics[k]:.1f}" if isinstance(metrics[k], float) else str(metrics[k])
                break

        if decision == "KEEP":
            dec_str = f"{GREEN}  KEEP  {RESET}"
        elif decision == "DISCARD":
            dec_str = f"{YELLOW}DISCARD {RESET}"
        elif decision == "FAIL":
            dec_str = f"{RED}  FAIL  {RESET}"
        elif decision == "SEED":
            dec_str = f"{CYAN}  SEED  {RESET}"
        else:
            # Non-canonical decision string — render dim without colour.
            dec_str = f"{DIM}{decision:^8}{RESET}"

        lines.append(f"  {rnd:>3}  │ {dec_str} │ {metric_val:>13} │ {desc}")

    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")

    # Plan summary — structured table
    lines.append("")
    plan_age = ""
    if plan_mtime:
        age_sec = time.time() - plan_mtime
        if age_sec < 60:
            plan_age = f"{DIM}(updated {int(age_sec)}s ago){RESET}"
        else:
            plan_age = f"{DIM}(updated {int(age_sec/60)}m ago){RESET}"
    lines.append(f"  {BOLD}Current Plan:{RESET} {plan_age}")
    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")
    lines.append(f"  {BOLD}  #   │ Status    │ Description{RESET}")
    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")

    # Plan parsing goes through phase_machine.parse_plan_text — single
    # source of truth shared with hook validators, so the dashboard
    # can't drift on plan.md format.
    for item in _pm.parse_plan_text(plan_text):
        item_id = item["id"]
        is_active = item["active"]
        tag = item["tag"]
        # tag carries the leading bracket content like "KEEP, metric=..."
        # or "DISCARD" or "FAIL" — collapse to the keyword for display.
        outcome = ""
        for kw in ("KEEP", "DISCARD", "FAIL"):
            if tag.startswith(kw):
                outcome = kw
                break

        desc = _fit(item["description"], plan_desc_avail)

        if is_active:
            status_str = f"{CYAN}> ACTIVE {RESET}"
            desc_str = f"{CYAN}{desc}{RESET}"
        elif outcome == "KEEP":
            status_str = f"{GREEN}  KEEP   {RESET}"
            desc_str = f"{DIM}{desc}{RESET}"
        elif outcome == "DISCARD":
            status_str = f"{YELLOW} DISCARD {RESET}"
            desc_str = f"{DIM}{desc}{RESET}"
        elif outcome == "FAIL":
            status_str = f"{RED}  FAIL   {RESET}"
            desc_str = f"{DIM}{desc}{RESET}"
        else:
            status_str = " pending "  # 9 visible chars, matches other statuses
            desc_str = desc

        lines.append(f"  {item_id:>4}  │ {status_str}│ {desc_str}")

    lines.append(f"  {BOLD}{'─' * divider_width}{RESET}")

    lines.append("")
    lines.append(f"  {DIM}Press Ctrl+C to stop watching{RESET}")

    return "\n".join(lines)


def _auto_detect_task_dir() -> str:
    """Auto-detect task_dir via phase_machine.find_active_task_dir — the
    single shared rule used by resume / dashboard / batch (was three
    slightly different rules; see find_active_task_dir docstring)."""
    return _pm.find_active_task_dir() or ""


def main():
    parser = argparse.ArgumentParser(
        description="AutoResearch live dashboard. Auto-detects task if no path given.",
    )
    parser.add_argument("task_dir", nargs="?", default=None,
                        help="Path to task directory (auto-detected if omitted)")
    parser.add_argument("--watch", type=int, nargs="?", const=5, default=5,
                        help="Refresh interval in seconds (default: 5, use 0 for one-shot)")
    args = parser.parse_args()

    if args.task_dir:
        task_dir = os.path.abspath(args.task_dir)
    else:
        task_dir = _auto_detect_task_dir()
        if not task_dir:
            print("No task found. Pass a task_dir or start /autoresearch first.")
            sys.exit(1)
        print(f"Auto-detected: {task_dir}", file=sys.stderr)

    # Force UTF-8 output on Windows + enable ANSI escape codes
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        # Enable ANSI escapes on Windows 10+
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass

    if args.watch and args.watch > 0:
        def clear_screen():
            # Use native clear (reliably clears both visible screen + scrollback)
            os.system("cls" if sys.platform == "win32" else "clear")
        history_offset = 0
        last_render = 0

        # Detect if stdin is a real TTY; if not, fallback to pure refresh (no keyboard)
        interactive = False
        try:
            interactive = sys.stdin.isatty()
        except Exception:
            interactive = False

        if interactive:
            try:
                setup_keyboard()
            except Exception:
                interactive = False

        try:
            while True:
                now = time.time()
                needs_render = False

                # Keyboard handling (only if interactive)
                if interactive:
                    try:
                        key = read_key_nonblocking()
                    except Exception:
                        key = None
                    if key == "QUIT" or key == "ESC":
                        break
                    elif key == "UP":
                        history_offset += 1
                        needs_render = True
                    elif key == "DOWN":
                        history_offset = max(0, history_offset - 1)
                        needs_render = True
                    elif key == "PGUP":
                        history_offset += 10
                        needs_render = True
                    elif key == "PGDN":
                        history_offset = max(0, history_offset - 10)
                        needs_render = True
                    elif key == "HOME":
                        history_offset = 999999
                        needs_render = True
                    elif key == "END":
                        history_offset = 0
                        needs_render = True

                # Auto-refresh
                if now - last_render >= args.watch:
                    needs_render = True

                if needs_render:
                    clear_screen()
                    print(render(task_dir, history_offset=history_offset), flush=True)
                    last_render = now

                time.sleep(0.1 if interactive else max(0.5, args.watch / 2))
        except KeyboardInterrupt:
            pass
        finally:
            if interactive:
                try:
                    restore_keyboard()
                except Exception:
                    pass
            print(f"\n{DIM}Dashboard stopped.{RESET}")
    else:
        print(render(task_dir))


if __name__ == "__main__":
    main()
