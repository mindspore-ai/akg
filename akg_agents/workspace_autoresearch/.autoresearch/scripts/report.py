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

"""
FINISH-phase report generator — produces .ar_state/report.md with summary
tables and an inline SVG optimization curve.

Stdlib only — no matplotlib, no numpy. The SVG is embedded directly so the
report is a self-contained Markdown file (renders in VS Code / GitHub).

Usage:
    python report.py <task_dir>          # write .ar_state/report.md
    python report.py <task_dir> --print  # dump to stdout (debug)
"""


# pylint: disable=missing-function-docstring,wrong-import-position
import argparse
import os
import sys
from html import escape as _h
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task_config import load_task_config
from phase_machine import history_path, load_progress
from utils.json_io import load_jsonl


REPORT_FILE = "report.md"


def report_path(task_dir: str) -> str:
    return os.path.join(task_dir, ".ar_state", REPORT_FILE)


def _load_history(task_dir: str) -> list[dict]:
    return load_jsonl(history_path(task_dir))


def _escape_md_cell(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _fmt_num(v: float) -> str:
    av = abs(v)
    if av >= 1000:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


class _Series:
    """Per-decision-bucket history view used by the SVG renderer."""
    def __init__(self):
        self.rounds_keep: list = []
        self.vals_keep: list = []
        self.rounds_discard: list = []
        self.vals_discard: list = []
        self.rounds_fail: list = []
        self.best_rounds: list = []
        self.best_vals: list = []


def _collect_series(history: list, primary: str,
                    lower_is_better: bool) -> _Series:
    """Bucket history into keep / discard / fail series plus the
    monotonic best-so-far curve."""
    s = _Series()
    current_best: Optional[float] = None
    for rec in history:
        r = rec.get("round")
        if r is None:
            continue
        decision = rec.get("decision", "")
        val = rec.get("metrics", {}).get(primary)
        if decision == "FAIL":
            s.rounds_fail.append(r)
            continue
        if val is None:
            continue
        if decision in ("KEEP", "SEED"):
            s.rounds_keep.append(r)
            s.vals_keep.append(val)
            if current_best is None:
                current_best = val
            elif lower_is_better:
                current_best = min(current_best, val)
            else:
                current_best = max(current_best, val)
            s.best_rounds.append(r)
            s.best_vals.append(current_best)
        elif decision == "DISCARD":
            s.rounds_discard.append(r)
            s.vals_discard.append(val)
            if current_best is not None:
                s.best_rounds.append(r)
                s.best_vals.append(current_best)
    return s


def _compute_y_range(all_vals: list,
                     ref_val: Optional[float]) -> tuple:
    """Cover all measured values + ref line, padded slightly. The old
    "clamp around ref" approach clipped the seed kernel off the top
    (seed is typically much slower than ref) and the early best-so-far
    line went with it."""
    if not all_vals:
        return 0.0, 1.0
    pool = list(all_vals)
    if ref_val is not None:
        pool.append(ref_val)
    ymin, ymax = min(pool), max(pool)
    span = ymax - ymin if ymax > ymin else max(abs(ymax) * 0.1, 1.0)
    ymin -= span * 0.05
    ymax += span * 0.10
    if ymax <= ymin:
        ymax = ymin + 1.0
    return ymin, ymax


def _svg_header(W: int, H: int, task_name: str, primary: str,
                lower_is_better: bool) -> list:
    direction = "lower is better" if lower_is_better else "higher is better"
    title = _h(f"{task_name} — {primary} ({direction})")
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">',
        f'<text x="{W/2:.1f}" y="22" text-anchor="middle" '
        f'font-size="13" font-weight="bold">{title}</text>',
    ]


def _svg_axes_and_grid(W: int, H: int, ML: int, MT: int, PW: int, PH: int,
                       xmin: float, xmax: float, ymin: float, ymax: float,
                       primary: str, sx, sy) -> list:
    """Plot rectangle, horizontal grid + Y tick labels, X ticks, axis labels."""
    out = [
        f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" '
        'fill="white" stroke="#888" stroke-width="0.5"/>',
    ]
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        y = sy(v)
        out.append(
            f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" '
            'stroke="#e8e8e8" stroke-dasharray="2,2"/>'
        )
        out.append(
            f'<text x="{ML-6}" y="{y+3:.1f}" text-anchor="end">'
            f'{_fmt_num(v)}</text>'
        )
    span = max(1, int(xmax - xmin))
    n_xticks = min(span + 1, 11)
    step = max(1, int(round(span / max(1, n_xticks - 1))))
    rr = int(xmin)
    while rr <= int(xmax):
        x = sx(rr)
        out.append(
            f'<line x1="{x:.1f}" y1="{MT+PH}" x2="{x:.1f}" '
            f'y2="{MT+PH+4}" stroke="#444"/>'
        )
        out.append(
            f'<text x="{x:.1f}" y="{MT+PH+18}" '
            f'text-anchor="middle">R{rr}</text>'
        )
        rr += step
    out.append(
        f'<text x="{ML+PW/2:.1f}" y="{H-12}" text-anchor="middle" '
        'font-size="12">Round</text>'
    )
    cy = MT + PH / 2
    out.append(
        f'<text x="16" y="{cy:.1f}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 16,{cy:.1f})">{_h(primary)}</text>'
    )
    return out


def _svg_best_line(best_rounds: list, best_vals: list, sx, sy) -> list:
    if not (best_rounds and best_vals):
        return []
    d_parts = []
    for i, (br, bv) in enumerate(zip(best_rounds, best_vals)):
        x, y = sx(br), sy(bv)
        if i == 0:
            d_parts.append(f"M {x:.1f} {y:.1f}")
        else:
            prev_y = sy(best_vals[i - 1])
            d_parts.append(f"L {x:.1f} {prev_y:.1f} L {x:.1f} {y:.1f}")
    return [
        f'<path d="{" ".join(d_parts)}" fill="none" stroke="#1f77b4" '
        'stroke-width="2" opacity="0.85"/>'
    ]


def _svg_data_points(s: _Series, sx, sy, MT: int) -> list:
    out = []
    for dr, dv in zip(s.rounds_discard, s.vals_discard):
        out.append(
            f'<circle cx="{sx(dr):.1f}" cy="{sy(dv):.1f}" r="4.5" '
            'fill="salmon" stroke="red" stroke-width="0.5" opacity="0.7"/>'
        )
    for kr, kv in zip(s.rounds_keep, s.vals_keep):
        out.append(
            f'<circle cx="{sx(kr):.1f}" cy="{sy(kv):.1f}" r="5.5" '
            'fill="#2ca02c" stroke="darkgreen" stroke-width="0.6"/>'
        )
    fail_y_px = MT + 10
    for fr in s.rounds_fail:
        x = sx(fr)
        out.append(
            f'<path d="M {x-4:.1f} {fail_y_px-4} L {x+4:.1f} {fail_y_px+4} '
            f'M {x+4:.1f} {fail_y_px-4} L {x-4:.1f} {fail_y_px+4}" '
            'stroke="black" stroke-width="1.5"/>'
        )
    return out


def _svg_speedup_labels(s: _Series, ref_val: Optional[float],
                        all_vals: list, sx, sy) -> list:
    """Speedup-x labels next to KEEP points. Skips seed/R0 — its speedup
    is by definition 1.0x and would clutter the chart."""
    if ref_val is None or not s.rounds_keep:
        return []
    annotations = [(kr, kv) for kr, kv in zip(s.rounds_keep, s.vals_keep)
                   if kr != 0 and kv > 0]
    if not annotations:
        return []
    y_span = max(all_vals) - min(all_vals) if len(all_vals) > 1 else 1
    min_gap = y_span * 0.06
    filtered = [annotations[-1]]
    for kr, kv in reversed(annotations[:-1]):
        if abs(kv - filtered[-1][1]) >= min_gap:
            filtered.append((kr, kv))
    filtered.reverse()
    out = []
    for kr, kv in filtered:
        speedup = ref_val / kv
        out.append(
            f'<text x="{sx(kr):.1f}" y="{sy(kv)-9:.1f}" '
            'text-anchor="middle" font-size="9" '
            f'fill="darkgreen">{speedup:.1f}x</text>'
        )
    return out


def _svg_legend_items(s: _Series, ref_val: Optional[float],
                      ref_label: str) -> list:
    items = []
    if s.vals_keep:
        items.append(("circle", "#2ca02c", "darkgreen",
                      f"keep ({len(s.rounds_keep)})"))
    if s.vals_discard:
        items.append(("circle", "salmon", "red",
                      f"discard ({len(s.rounds_discard)})"))
    if s.rounds_fail:
        items.append(("x", "black", "black",
                      f"fail ({len(s.rounds_fail)})"))
    if s.best_rounds:
        items.append(("line", "#1f77b4", "#1f77b4", "best so far"))
    if ref_val is not None:
        items.append(("dashline", "#ff8c00", "#ff8c00",
                      f"{ref_label} ({ref_val:.1f})"))
    return items


def _svg_legend(items: list, legend_x: int, legend_y: int) -> list:
    out = []
    for i, (kind, fill, stroke, label) in enumerate(items):
        ly = legend_y + i * 18
        if kind == "circle":
            out.append(
                f'<circle cx="{legend_x+6}" cy="{ly:.1f}" r="4" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="0.5"/>'
            )
        elif kind == "x":
            out.append(
                f'<path d="M {legend_x+1} {ly-3:.1f} '
                f'L {legend_x+11} {ly+3:.1f} '
                f'M {legend_x+11} {ly-3:.1f} '
                f'L {legend_x+1} {ly+3:.1f}" '
                f'stroke="{stroke}" stroke-width="1.5"/>'
            )
        elif kind == "line":
            out.append(
                f'<line x1="{legend_x}" y1="{ly:.1f}" '
                f'x2="{legend_x+12}" y2="{ly:.1f}" '
                f'stroke="{stroke}" stroke-width="2"/>'
            )
        elif kind == "dashline":
            out.append(
                f'<line x1="{legend_x}" y1="{ly:.1f}" '
                f'x2="{legend_x+12}" y2="{ly:.1f}" '
                f'stroke="{stroke}" stroke-width="1.5" '
                'stroke-dasharray="4,2"/>'
            )
        out.append(
            f'<text x="{legend_x+18}" y="{ly+3:.1f}" '
            f'font-size="10">{_h(label)}</text>'
        )
    return out


def _generate_svg(history: list[dict], primary: str, lower_is_better: bool,
                  ref_val: Optional[float], ref_label: str,
                  task_name: str) -> str:
    """Render the optimization curve as inline SVG. Returns "" if no data."""
    s = _collect_series(history, primary, lower_is_better)
    all_vals = s.vals_keep + s.vals_discard
    if not all_vals and not s.rounds_fail:
        return ""

    W, H = 900, 420
    ML, MR, MT, MB = 70, 170, 40, 60
    PW, PH = W - ML - MR, H - MT - MB

    all_rounds = s.rounds_keep + s.rounds_discard + s.rounds_fail
    xmin, xmax = 0, (max(all_rounds) if all_rounds else 1)
    if xmax <= xmin:
        xmax = xmin + 1
    ymin, ymax = _compute_y_range(all_vals, ref_val)

    def sx(x: float) -> float:
        return ML + (x - xmin) / (xmax - xmin) * PW

    def sy(v: float) -> float:
        return MT + (1 - (v - ymin) / (ymax - ymin)) * PH

    p: list[str] = []
    p.extend(_svg_header(W, H, task_name, primary, lower_is_better))
    p.extend(_svg_axes_and_grid(W, H, ML, MT, PW, PH,
                                xmin, xmax, ymin, ymax, primary, sx, sy))
    if ref_val is not None and ymin <= ref_val <= ymax:
        ry = sy(ref_val)
        p.append(
            f'<line x1="{ML}" y1="{ry:.1f}" x2="{ML+PW}" y2="{ry:.1f}" '
            'stroke="#ff8c00" stroke-width="1.5" '
            'stroke-dasharray="6,3" opacity="0.75"/>'
        )
    p.extend(_svg_best_line(s.best_rounds, s.best_vals, sx, sy))
    p.extend(_svg_data_points(s, sx, sy, MT))
    p.extend(_svg_speedup_labels(s, ref_val, all_vals, sx, sy))
    p.extend(_svg_legend(
        _svg_legend_items(s, ref_val, ref_label),
        ML + PW + 14, MT + 14,
    ))
    p.append("</svg>")
    return "\n".join(p)


def _resolve_baseline(progress: dict) -> tuple:
    """(ref_val, ref_label). baseline_source distinguishes PyTorch ref
    from seed-fallback so the report labels the line honestly."""
    raw_ref = progress.get("baseline_metric")
    ref_val = (raw_ref if isinstance(raw_ref, (int, float))
               and raw_ref > 0 else None)
    src = progress.get("baseline_source", "")
    if src == "ref":
        ref_label = "PyTorch ref"
    elif src == "seed_fallback":
        ref_label = "seed baseline"
    else:
        ref_label = "baseline"
    return ref_val, ref_label


def _find_best_round(history: list, primary: str, best_val) -> Optional[int]:
    """Earliest KEEP/SEED round whose metric equals best_val. Improvements
    are monotonic in best-so-far so the first match is the introducing
    round."""
    if not isinstance(best_val, (int, float)):
        return None
    for rec in history:
        if rec.get("decision") not in ("KEEP", "SEED"):
            continue
        v = rec.get("metrics", {}).get(primary)
        if isinstance(v, (int, float)) and abs(v - best_val) < 1e-9:
            return rec.get("round")
    return None


def _improvement_str(seed_val, best_val, lower_is_better: bool) -> str:
    if not (isinstance(seed_val, (int, float))
            and isinstance(best_val, (int, float)) and seed_val):
        return "N/A"
    if lower_is_better:
        pct = (seed_val - best_val) / seed_val * 100
        return f"{pct:.1f}% reduction"
    pct = (best_val - seed_val) / seed_val * 100
    return f"{pct:.1f}% increase"


def _collect_improvements(history: list, primary: str,
                          lower_is_better: bool) -> list:
    """Monotonic best-so-far steps among KEEP/SEED rounds."""
    improvements: list = []
    prev_best: Optional[float] = None
    for rec in history:
        if rec.get("decision") not in ("KEEP", "SEED"):
            continue
        v = rec.get("metrics", {}).get(primary)
        if v is None:
            continue
        if prev_best is not None and v != prev_best:
            delta = prev_best - v if lower_is_better else v - prev_best
            if delta > 0:
                improvements.append({
                    "round": rec.get("round"),
                    "desc": rec.get("description", ""),
                    "from": prev_best,
                    "to": v,
                    "delta": delta,
                })
        if (prev_best is None
                or (lower_is_better and v < prev_best)
                or (not lower_is_better and v > prev_best)):
            prev_best = v
    return improvements


def _first_case_descs(history: list) -> list:
    """per_shape_descs from the first history record that carries them
    (the SEED round, populated by
    task_config.eval_assemble.assemble_eval_result)."""
    for rec in history:
        d = (rec.get("metrics", {}) or {}).get("per_shape_descs")
        if isinstance(d, list) and d:
            return d
    return []


def _render_overview(task_name: str, total_rounds: int, n_keep: int,
                     n_discard: int, n_fail: int, primary: str,
                     lower_is_better: bool, ref_val, ref_label: str,
                     seed_val, best_val, best_round,
                     improvement_str: str,
                     speedup_str: Optional[str]) -> list:
    direction_zh = "越低越好" if lower_is_better else "越高越好"
    lines = [
        f"# {task_name} — 优化报告",
        "",
        "## 总览",
        "",
        "| 项目 | 值 |",
        "|------|---|",
        f"| 任务 | {_escape_md_cell(task_name)} |",
        f"| 总轮次 | {total_rounds} |",
        f"| 接受 / 失败 / 丢弃 | {n_keep} / {n_fail} / {n_discard} |",
        f"| 主指标 | {primary} ({direction_zh}) |",
    ]
    if ref_val is not None:
        lines.append(f"| **{ref_label}** | **{ref_val:.2f}** |")
    if seed_val is not None:
        lines.append(f"| Seed kernel | {seed_val} |")
    lines.append(f"| **最优结果** | **{best_val} (Round {best_round})** |")
    lines.append(f"| 总改进 (vs seed) | {improvement_str} |")
    if speedup_str:
        lines.append(f"| **最优加速比 (vs {ref_label})** | **{speedup_str}** |")
    lines.append("")
    return lines


def _render_shapes(case_descs: list) -> list:
    if len(case_descs) <= 1:
        return []
    lines = [f"## 测试形状 ({len(case_descs)})", ""]
    for i, d in enumerate(case_descs):
        lines.append(f"{i}. {d}")
    lines.append("")
    return lines


def _render_improvements_table(improvements: list, primary: str) -> list:
    if not improvements:
        return []
    lines = [
        "## Key Improvements",
        "",
        f"| Round | Description | {primary} | Improvement |",
        f"|-------|-------------|{'---' * 4}|-------------|",
    ]
    for imp in improvements:
        from_v = (f"{imp['from']:.4f}" if isinstance(imp['from'], float)
                  else str(imp['from']))
        to_v = (f"{imp['to']:.4f}" if isinstance(imp['to'], float)
                else str(imp['to']))
        delta_v = (f"{imp['delta']:.4f}" if isinstance(imp['delta'], float)
                   else str(imp['delta']))
        desc = _escape_md_cell(imp['desc'])
        lines.append(
            f"| R{imp['round']} | {desc} | {from_v} → {to_v} | -{delta_v} |"
        )
    lines.append("")
    return lines


def _render_all_rounds(history: list, primary: str) -> list:
    lines = [
        "## All Rounds",
        "",
        f"| Round | Description | Decision | {primary} |",
        f"|-------|-------------|----------|{'---' * 4}|",
    ]
    for rec in history:
        rnd = rec.get("round", "?")
        decision = rec.get("decision", "?")
        val = rec.get("metrics", {}).get(primary, "—")
        if isinstance(val, float):
            val = f"{val:.4f}"
        desc = _escape_md_cell((rec.get("description") or "")[:80])
        lines.append(f"| R{rnd} | {desc} | {decision} | {val} |")
    lines.append("")
    return lines


def render_report(task_dir: str) -> str:
    """Build the full markdown report. Returns "" if no plottable data."""
    config = load_task_config(task_dir)
    if config is None:
        return ""
    history = _load_history(task_dir)
    if not history:
        return ""
    primary = config.primary_metric
    lower_is_better = config.lower_is_better
    progress = load_progress(task_dir) or {}

    ref_val, ref_label = _resolve_baseline(progress)
    n_keep = sum(1 for r in history if r.get("decision") in ("KEEP", "SEED"))
    n_discard = sum(1 for r in history if r.get("decision") == "DISCARD")
    n_fail = sum(1 for r in history if r.get("decision") == "FAIL")
    seed_val = progress.get("seed_metric")
    best_val = progress.get("best_metric")
    best_round = _find_best_round(history, primary, best_val)
    improvement_str = _improvement_str(seed_val, best_val, lower_is_better)
    speedup_str = (f"{ref_val / best_val:.2f}x"
                   if (ref_val is not None
                       and isinstance(best_val, (int, float))
                       and best_val > 0) else None)
    task_name = (progress.get("task")
                 or os.path.basename(os.path.normpath(task_dir)))
    svg = _generate_svg(history, primary, lower_is_better, ref_val,
                        ref_label, task_name)
    improvements = _collect_improvements(history, primary, lower_is_better)

    lines = _render_overview(task_name, len(history), n_keep, n_discard,
                             n_fail, primary, lower_is_better,
                             ref_val, ref_label, seed_val, best_val,
                             best_round, improvement_str, speedup_str)
    lines.extend(_render_shapes(_first_case_descs(history)))
    if svg:
        lines.extend(["## Optimization Curve", "", svg, ""])
    lines.extend(_render_improvements_table(improvements, primary))
    lines.extend(_render_all_rounds(history, primary))

    return "\n".join(lines)


def write_report(task_dir: str) -> Optional[str]:
    """Write the report to .ar_state/report.md. Returns path or None."""
    md = render_report(task_dir)
    if not md:
        return None
    out = report_path(task_dir)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    return out


def main():
    ap = argparse.ArgumentParser(description="Generate FINISH-phase report.md")
    ap.add_argument("task_dir", help="Path to autoresearch task directory")
    ap.add_argument("--print", dest="to_stdout", action="store_true",
                    help="Print report to stdout instead of writing")
    args = ap.parse_args()

    task_dir = os.path.abspath(args.task_dir)
    if args.to_stdout:
        sys.stdout.write(render_report(task_dir))
        return
    p = write_report(task_dir)
    if p:
        print(p)
    else:
        print("(no plottable data — empty history)", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
