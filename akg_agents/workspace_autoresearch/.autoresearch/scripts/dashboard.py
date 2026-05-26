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

# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring,wrong-import-position

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_machine as _pm
from utils.json_io import load_jsonl as _shared_load_jsonl
from utils.json_io import _read_whole_file as _shared_read_whole_file
from task_config.metric_policy import STUCK_BASELINE_OUTCOMES

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

    def setup_keyboard():

        pass
    def restore_keyboard():
        pass

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


def _compute_layout() -> tuple:
    try:
        term_width = os.get_terminal_size().columns
    except Exception:
        term_width = 100
    return (
        max(10, term_width - _HIST_PREFIX_VIS - 2),
        max(10, term_width - _PLAN_PREFIX_VIS - 2),
        max(40, term_width - 2),
    )


def _format_updated(raw: str) -> str:
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return raw


def _improvement_text(best, baseline, src) -> str:
    """Speedup vs baseline. The ANCHOR depends on baseline_source —
    "ref" means PyTorch reference; "seed_fallback" means baseline IS
    the seed timing itself (ratio is self-relative). Mislabeling
    seed_fallback as "vs ref" is a lie the user couldn't catch from
    the Best line alone."""
    if not (best is not None and baseline is not None
            and baseline != 0 and best != 0):
        return f"{DIM}N/A{RESET}"
    improv_pct = (baseline - best) / abs(baseline) * 100
    speedup = baseline / best
    color = GREEN if improv_pct > 0 else RED
    if src == "ref":
        anchor_label = "vs ref"
    elif src == "seed_fallback":
        anchor_label = "vs seed (no ref measured)"
    else:
        anchor_label = "vs baseline"
    return (f"{color}{speedup:.2f}x {anchor_label} "
            f"({improv_pct:+.1f}%){RESET}")


def _render_task_header(progress, task_dir) -> list:
    plan_ver = progress.get("plan_version", 0)
    status = ("active" if os.path.exists(_pm.plan_path(task_dir))
              else "no_plan")
    status_color = (GREEN if status == "active"
                    else (YELLOW if status == "no_plan" else CYAN))
    return [
        "",
        f"  {BOLD}Task:{RESET}     {progress.get('task', '?')}",
        f"  {BOLD}Status:{RESET}   {status_color}{status}{RESET}  "
        f"(plan v{plan_ver})",
        f"  {BOLD}Updated:{RESET}  {DIM}"
        f"{_format_updated(progress.get('last_updated', '?'))}{RESET}",
    ]


def _render_abort_banner(progress) -> list:
    """Task-level abort banner — fires only for INFRA_FAIL (agent can't
    fix from EDIT). error_source picks ref-broken vs other."""
    if progress.get("baseline_outcome") != "infra_fail":
        return []
    if (progress.get("baseline_error_source") or "") == "ref":
        return [
            "",
            f"  {BOLD}{RED}ABORTED:{RESET}  {RED}REF BROKEN{RESET}"
            "  reference.py is invalid.",
            f"           {DIM}Fix the source --ref file and re-run "
            f"/autoresearch from scratch.{RESET}",
        ]
    return [
        "",
        f"  {BOLD}{YELLOW}ABORTED:{RESET}  "
        f"{YELLOW}EVAL PIPELINE BROKEN{RESET}  no per-shape data "
        "produced.",
        f"           {DIM}Check worker / device / eval.timeout, then "
        f"retry baseline.py.{RESET}",
    ]


def _render_seed_line(progress, baseline) -> str:
    """Seed (initial kernel timing). Task-level INFRA_FAIL aborts
    surface in the banner above; here they collapse to "not measured"."""
    seed = progress.get("seed_metric")
    outcome = progress.get("baseline_outcome")
    if seed is not None:
        if seed != baseline:
            return (f"  {BOLD}Seed:{RESET}     {seed}  "
                    f"{DIM}(initial kernel){RESET}")
        return ""  # redundant with Baseline line (seed_fallback path)
    if outcome == "kernel_fail":
        return (f"  {BOLD}Seed:{RESET}     {RED}FAILED{RESET}  "
                f"{DIM}(kernel verify or profile failed; "
                f"timing dropped){RESET}")
    if outcome in STUCK_BASELINE_OUTCOMES:
        return (f"  {BOLD}Seed:{RESET}     "
                f"{DIM}— (not measured; see ABORTED above){RESET}")
    return (f"  {BOLD}Seed:{RESET}     "
            f"{DIM}— (no timing recorded){RESET}")


def _render_metrics_block(progress) -> list:
    rounds = progress.get("eval_rounds", 0)
    max_rounds = progress.get("max_rounds", 20)
    best = progress.get("best_metric")
    baseline = progress.get("baseline_metric")
    best_commit = progress.get("best_commit", "?")
    failures = progress.get("consecutive_failures", 0)

    frac = rounds / max_rounds if max_rounds > 0 else 0
    budget_color = (GREEN if frac < 0.5
                    else (YELLOW if frac < 0.8 else RED))
    fail_color = (RED if failures >= 3
                  else (YELLOW if failures > 0 else GREEN))
    improv_str = _improvement_text(best, baseline,
                                   progress.get("baseline_source"))

    out = [
        "",
        f"  {BOLD}Budget:{RESET}   {budget_color}{bar(frac)} "
        f"{rounds}/{max_rounds}{RESET}",
    ]
    baseline_tags = {
        "ref": f"{DIM}(PyTorch reference){RESET}",
        "seed_fallback":
            f"{YELLOW}(fallback: seed — ref not measured by worker){RESET}",
    }
    if baseline is None:
        out.append(f"  {BOLD}Baseline:{RESET} {DIM}— (not measured){RESET}")
    else:
        tag = baseline_tags.get(progress.get("baseline_source"),
                                f"{DIM}(source unknown){RESET}")
        out.append(f"  {BOLD}Baseline:{RESET} {baseline}  {tag}")
    seed_line = _render_seed_line(progress, baseline)
    if seed_line:
        out.append(seed_line)
    out.append(f"  {BOLD}Best:{RESET}     {GREEN}{best}{RESET}  ({improv_str})")
    out.append(f"  {BOLD}Commit:{RESET}   {best_commit}")
    trigger = (f"  {RED}⚠ DIAGNOSIS WILL TRIGGER{RESET}"
               if failures >= 3 else "")
    out.append(f"  {BOLD}Failures:{RESET} {fail_color}{failures}{RESET} "
               f"consecutive{trigger}")
    return out


def _decision_cell(decision: str) -> str:
    if decision == "KEEP":
        return f"{GREEN}  KEEP  {RESET}"
    if decision == "DISCARD":
        return f"{YELLOW}DISCARD {RESET}"
    if decision == "FAIL":
        return f"{RED}  FAIL  {RESET}"
    if decision == "SEED":
        return f"{CYAN}  SEED  {RESET}"
    # Older history.jsonl files may carry deprecated decisions (e.g.
    # REACTIVATE). Fall through to dim renderer rather than colour-code.
    return f"{DIM}{decision:^8}{RESET}"


def _metric_cell(metrics: dict) -> str:
    """Display value of the primary metric. Tries latency_us / score
    first, then falls back to the first value in the dict."""
    for k in ("latency_us", "score"):
        if k in metrics and metrics[k] is not None:
            v = metrics[k]
            return f"{v:.1f}" if isinstance(v, float) else str(v)
    if not metrics:
        return "—"
    first_val = next(iter(metrics.values()), None)
    if first_val is None:
        return "—"
    return (f"{first_val:.1f}" if isinstance(first_val, float)
            else str(first_val))


def _compute_history_window(history_window, n_total: int) -> int:
    if history_window is not None:
        return history_window
    # Auto: reserve ~25 lines for header+plan, use rest for history
    try:
        term_h = os.get_terminal_size().lines
    except Exception:
        term_h = 40
    return max(5, term_h - 28)


def _render_history_section(history_all, history_offset, history_window,
                            hist_desc_avail, divider_width) -> list:
    n_total = len(history_all)
    history_window = _compute_history_window(history_window, n_total)
    history_offset = max(0, min(history_offset,
                                max(0, n_total - history_window)))
    end = n_total - history_offset
    start = max(0, end - history_window)
    history = history_all[start:end]

    scroll_info = ""
    if n_total > history_window:
        scroll_info = (f" [{start+1}-{end} of {n_total}, "
                       "↑↓ PgUp/PgDn Home/End q=quit]")

    out = [
        "",
        f"  {BOLD}History{RESET}{DIM}{scroll_info}{RESET}",
        f"  {BOLD}{'─' * divider_width}{RESET}",
        f"  {BOLD}  #  │ Decision │ Metric        │ Description{RESET}",
        f"  {BOLD}{'─' * divider_width}{RESET}",
    ]
    for rec in history:
        rnd = rec.get("round")
        rnd = "?" if rnd is None else str(rnd)
        decision = rec.get("decision", "?")
        raw_desc = rec.get("description", "")
        pid = rec.get("plan_item")
        # Prefix description with plan_item id when present so every row
        # traces back to plan.md.
        desc = _fit(f"{pid}: {raw_desc}" if pid else raw_desc,
                    hist_desc_avail)
        metric_val = _metric_cell(rec.get("metrics", {}))
        out.append(f"  {rnd:>3}  │ {_decision_cell(decision)} │ "
                   f"{metric_val:>13} │ {desc}")
    out.append(f"  {BOLD}{'─' * divider_width}{RESET}")
    return out


def _plan_status_cell(is_active: bool, tag: str, desc: str) -> tuple:
    """(status_str, desc_str) for one plan item row. tag carries the
    leading bracket content like "KEEP, metric=..."."""
    if is_active:
        return f"{CYAN}> ACTIVE {RESET}", f"{CYAN}{desc}{RESET}"
    outcome = ""
    for kw in ("KEEP", "DISCARD", "FAIL"):
        if tag.startswith(kw):
            outcome = kw
            break
    if outcome == "KEEP":
        return f"{GREEN}  KEEP   {RESET}", f"{DIM}{desc}{RESET}"
    if outcome == "DISCARD":
        return f"{YELLOW} DISCARD {RESET}", f"{DIM}{desc}{RESET}"
    if outcome == "FAIL":
        return f"{RED}  FAIL   {RESET}", f"{DIM}{desc}{RESET}"
    return " pending ", desc  # 9 visible chars, matches other statuses


def _render_plan_section(plan_text, plan_mtime, plan_desc_avail,
                         divider_width) -> list:
    plan_age = ""
    if plan_mtime:
        age_sec = time.time() - plan_mtime
        if age_sec < 60:
            plan_age = f"{DIM}(updated {int(age_sec)}s ago){RESET}"
        else:
            plan_age = f"{DIM}(updated {int(age_sec/60)}m ago){RESET}"
    out = [
        "",
        f"  {BOLD}Current Plan:{RESET} {plan_age}",
        f"  {BOLD}{'─' * divider_width}{RESET}",
        f"  {BOLD}  #   │ Status    │ Description{RESET}",
        f"  {BOLD}{'─' * divider_width}{RESET}",
    ]
    # phase_machine.parse_plan_text is the single source of truth shared
    # with hook validators.
    for item in _pm.parse_plan_text(plan_text):
        desc = _fit(item["description"], plan_desc_avail)
        status_str, desc_str = _plan_status_cell(item["active"],
                                                 item["tag"], desc)
        out.append(f"  {item['id']:>4}  │ {status_str}│ {desc_str}")
    out.append(f"  {BOLD}{'─' * divider_width}{RESET}")
    return out


def render(task_dir, history_offset=0, history_window=None):
    """Render dashboard.

    history_offset: how many rounds to skip from the END (0 = latest).
    history_window: how many rounds to show (None = auto on terminal h).
    """
    progress = load_json(_pm.progress_path(task_dir))
    history_all = load_jsonl(_pm.history_path(task_dir))
    plan_text, plan_mtime = load_plan(_pm.plan_path(task_dir))
    hist_desc_avail, plan_desc_avail, divider_width = _compute_layout()

    lines = [
        f"{BOLD}{CYAN}╔════════════════════════════════════════════════════"
        f"══════════╗{RESET}",
        f"{BOLD}{CYAN}║          AUTORESEARCH DASHBOARD                    "
        f"         ║{RESET}",
        f"{BOLD}{CYAN}╚════════════════════════════════════════════════════"
        f"══════════╝{RESET}",
    ]
    if progress is None:
        lines.append(
            f"\n  {RED}No progress.json found at "
            f"{_pm.progress_path(task_dir)}{RESET}")
        lines.append("  Run /autoresearch --ref ... --op-name ... first.")
        return "\n".join(lines)

    lines.extend(_render_task_header(progress, task_dir))
    lines.extend(_render_abort_banner(progress))
    lines.extend(_render_metrics_block(progress))
    lines.extend(_render_history_section(history_all, history_offset,
                                         history_window, hist_desc_avail,
                                         divider_width))
    lines.extend(_render_plan_section(plan_text, plan_mtime,
                                      plan_desc_avail, divider_width))
    lines.append("")
    lines.append(f"  {DIM}Press Ctrl+C to stop watching{RESET}")
    return "\n".join(lines)


def _auto_detect_task_dir() -> str:
    """Auto-detect task_dir via phase_machine.find_active_task_dir — the
    single shared rule used by resume / dashboard / batch (was three
    slightly different rules; see find_active_task_dir docstring)."""
    return _pm.find_active_task_dir() or ""


_KEY_OFFSET_DELTA = {
    "UP":   ("relative", +1),
    "DOWN": ("relative", -1),
    "PGUP": ("relative", +10),
    "PGDN": ("relative", -10),
    "HOME": ("absolute", 999999),
    "END":  ("absolute", 0),
}


def _apply_key(key, history_offset: int) -> tuple:
    """Return (new_offset, should_quit, needs_render) for a single key.
    Unrecognized keys leave state unchanged."""
    if key in ("QUIT", "ESC"):
        return history_offset, True, False
    spec = _KEY_OFFSET_DELTA.get(key)
    if spec is None:
        return history_offset, False, False
    kind, value = spec
    if kind == "absolute":
        return value, False, True
    return max(0, history_offset + value), False, True


def _enable_windows_ansi() -> None:
    """Force UTF-8 output + enable ANSI escapes on Windows 10+."""
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def _resolve_task_dir(arg) -> str:
    if arg:
        return os.path.abspath(arg)
    td = _auto_detect_task_dir()
    if not td:
        print("No task found. Pass a task_dir or start /autoresearch first.")
        sys.exit(1)
    print(f"Auto-detected: {td}", file=sys.stderr)
    return td


def _watch_loop(task_dir: str, watch: int) -> None:
    def clear_screen():
        # Use native clear (reliably clears both visible screen + scrollback)
        os.system("cls" if sys.platform == "win32" else "clear")

    history_offset = 0
    last_render = 0

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
            if interactive:
                try:
                    key = read_key_nonblocking()
                except Exception:
                    key = None
                history_offset, should_quit, key_render = _apply_key(
                    key, history_offset)
                if should_quit:
                    break
                needs_render = needs_render or key_render
            if now - last_render >= watch:
                needs_render = True
            if needs_render:
                clear_screen()
                print(render(task_dir, history_offset=history_offset),
                      flush=True)
                last_render = now
            time.sleep(0.1 if interactive else max(0.5, watch / 2))
    except KeyboardInterrupt:
        pass
    finally:
        if interactive:
            try:
                restore_keyboard()
            except Exception:
                pass
        print(f"\n{DIM}Dashboard stopped.{RESET}")


def main():
    parser = argparse.ArgumentParser(
        description=("AutoResearch live dashboard. Auto-detects task if "
                     "no path given."),
    )
    parser.add_argument("task_dir", nargs="?", default=None,
                        help=("Path to task directory "
                              "(auto-detected if omitted)"))
    parser.add_argument("--watch", type=int, nargs="?", const=5, default=5,
                        help=("Refresh interval in seconds (default: 5, "
                              "use 0 for one-shot)"))
    args = parser.parse_args()

    task_dir = _resolve_task_dir(args.task_dir)
    if sys.platform == "win32":
        _enable_windows_ansi()

    if args.watch and args.watch > 0:
        _watch_loop(task_dir, args.watch)
    else:
        print(render(task_dir))


if __name__ == "__main__":
    main()
