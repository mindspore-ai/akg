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

import argparse
import json
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


def _generate_svg(history: list[dict], primary: str, lower_is_better: bool,
                  ref_val: Optional[float], ref_label: str, task_name: str) -> str:
    """Render the optimization curve as inline SVG. Returns "" if no data."""
    rounds_keep, vals_keep = [], []
    rounds_discard, vals_discard = [], []
    rounds_fail, vals_fail = [], []
    best_rounds, best_vals = [], []
    speedup_by_round: dict[int, float] = {}
    current_best: Optional[float] = None

    for rec in history:
        r = rec.get("round")
        if r is None:
            continue
        decision = rec.get("decision", "")
        metrics = rec.get("metrics", {})
        val = metrics.get(primary)
        sp = metrics.get("speedup_vs_ref")
        if isinstance(sp, (int, float)) and sp > 0:
            speedup_by_round[int(r)] = float(sp)

        if decision == "FAIL":
            # correctness-FAIL: kernel ran 50 shapes but verify caught a
            # divergence, so val (latency_us) is present even though
            # decision is FAIL. crash-FAIL has val=None. Track both;
            # render places X at the real Y when val is known, else at
            # the figure top so the round is still visible.
            rounds_fail.append(r)
            vals_fail.append(val)
            continue
        if val is None:
            continue
        if decision in ("KEEP", "SEED"):
            rounds_keep.append(r)
            vals_keep.append(val)
            if current_best is None:
                current_best = val
            elif lower_is_better:
                current_best = min(current_best, val)
            else:
                current_best = max(current_best, val)
            best_rounds.append(r)
            best_vals.append(current_best)
        elif decision == "DISCARD":
            rounds_discard.append(r)
            vals_discard.append(val)
            if current_best is not None:
                best_rounds.append(r)
                best_vals.append(current_best)

    all_vals = vals_keep + vals_discard + [v for v in vals_fail if v is not None]
    if not all_vals and not rounds_fail:
        return ""

    W, H = 900, 420
    ML, MR, MT, MB = 70, 170, 40, 60
    PW, PH = W - ML - MR, H - MT - MB

    all_rounds = rounds_keep + rounds_discard + rounds_fail
    xmin = 0
    xmax = max(all_rounds) if all_rounds else 1
    if xmax <= xmin:
        xmax = xmin + 1

    # Y range covers all measured values + ref line. The old "clamp around
    # ref" approach assumed keeps cluster near ref, but the seed kernel is
    # typically slower than ref by a large margin and was getting clipped
    # off the top, taking the early best-so-far line with it.
    if all_vals:
        pool = list(all_vals)
        if ref_val is not None:
            pool.append(ref_val)
        ymin, ymax = min(pool), max(pool)
        span = ymax - ymin if ymax > ymin else max(abs(ymax) * 0.1, 1.0)
        ymin -= span * 0.05
        ymax += span * 0.10
    else:
        ymin, ymax = 0.0, 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    def sx(x: float) -> float:
        return ML + (x - xmin) / (xmax - xmin) * PW

    def sy(v: float) -> float:
        return MT + (1 - (v - ymin) / (ymax - ymin)) * PH

    p: list[str] = []
    p.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">'
    )

    direction = "lower is better" if lower_is_better else "higher is better"
    title = _h(f"{task_name} — {primary} ({direction})")
    p.append(
        f'<text x="{W/2:.1f}" y="22" text-anchor="middle" '
        f'font-size="13" font-weight="bold">{title}</text>'
    )

    p.append(
        f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" '
        f'fill="white" stroke="#888" stroke-width="0.5"/>'
    )

    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        y = sy(v)
        p.append(
            f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" '
            f'stroke="#e8e8e8" stroke-dasharray="2,2"/>'
        )
        p.append(f'<text x="{ML-6}" y="{y+3:.1f}" text-anchor="end">{_fmt_num(v)}</text>')

    span = max(1, int(xmax - xmin))
    n_xticks = min(span + 1, 11)
    step = max(1, int(round(span / max(1, n_xticks - 1))))
    rr = int(xmin)
    while rr <= int(xmax):
        x = sx(rr)
        p.append(
            f'<line x1="{x:.1f}" y1="{MT+PH}" x2="{x:.1f}" y2="{MT+PH+4}" stroke="#444"/>'
        )
        p.append(f'<text x="{x:.1f}" y="{MT+PH+18}" text-anchor="middle">R{rr}</text>')
        rr += step

    p.append(
        f'<text x="{ML+PW/2:.1f}" y="{H-12}" text-anchor="middle" font-size="12">Round</text>'
    )
    cy = MT + PH / 2
    p.append(
        f'<text x="16" y="{cy:.1f}" text-anchor="middle" font-size="12" '
        f'transform="rotate(-90 16,{cy:.1f})">{_h(primary)}</text>'
    )

    if ref_val is not None and ymin <= ref_val <= ymax:
        ry = sy(ref_val)
        p.append(
            f'<line x1="{ML}" y1="{ry:.1f}" x2="{ML+PW}" y2="{ry:.1f}" '
            f'stroke="#ff8c00" stroke-width="1.5" stroke-dasharray="6,3" opacity="0.75"/>'
        )

    if best_rounds and best_vals:
        d_parts = []
        for i, (br, bv) in enumerate(zip(best_rounds, best_vals)):
            x = sx(br)
            y = sy(bv)
            if i == 0:
                d_parts.append(f"M {x:.1f} {y:.1f}")
            else:
                prev_y = sy(best_vals[i - 1])
                d_parts.append(f"L {x:.1f} {prev_y:.1f} L {x:.1f} {y:.1f}")
        p.append(
            f'<path d="{" ".join(d_parts)}" fill="none" stroke="#1f77b4" '
            f'stroke-width="2" opacity="0.85"/>'
        )

    for dr, dv in zip(rounds_discard, vals_discard):
        p.append(
            f'<circle cx="{sx(dr):.1f}" cy="{sy(dv):.1f}" r="4.5" '
            f'fill="salmon" stroke="red" stroke-width="0.5" opacity="0.7"/>'
        )

    for kr, kv in zip(rounds_keep, vals_keep):
        p.append(
            f'<circle cx="{sx(kr):.1f}" cy="{sy(kv):.1f}" r="5.5" '
            f'fill="#2ca02c" stroke="darkgreen" stroke-width="0.6"/>'
        )

    if ref_val is not None and rounds_keep:
        # Skip seed/R0 — speedup at the seed is by definition baseline-relative
        # and would clutter the chart; users see the seed dot itself.
        annotations = [(kr, kv) for kr, kv in zip(rounds_keep, vals_keep)
                       if kr != 0 and kv > 0]
        if annotations:
            y_span = max(all_vals) - min(all_vals) if len(all_vals) > 1 else 1
            min_gap = y_span * 0.06
            filtered = [annotations[-1]]
            for kr, kv in reversed(annotations[:-1]):
                if abs(kv - filtered[-1][1]) >= min_gap:
                    filtered.append((kr, kv))
            filtered.reverse()
            for kr, kv in filtered:
                speedup = speedup_by_round.get(int(kr))
                if speedup is None:
                    continue
                p.append(
                    f'<text x="{sx(kr):.1f}" y="{sy(kv)-9:.1f}" text-anchor="middle" '
                    f'font-size="9" fill="darkgreen">{speedup:.1f}x</text>'
                )

    # crash-FAIL (val=None) → fixed marker at the figure top so the
    # round number is still visible. correctness-FAIL with a real
    # latency → marker at sy(val) like KEEP/DISCARD, so the chart
    # honours the data instead of hiding it.
    fail_y_px_default = MT + 10
    for fr, fv in zip(rounds_fail, vals_fail):
        x = sx(fr)
        y = sy(fv) if fv is not None else fail_y_px_default
        p.append(
            f'<path d="M {x-4:.1f} {y-4:.1f} L {x+4:.1f} {y+4:.1f} '
            f'M {x+4:.1f} {y-4:.1f} L {x-4:.1f} {y+4:.1f}" '
            f'stroke="black" stroke-width="1.5"/>'
        )

    legend_x = ML + PW + 14
    legend_y = MT + 14
    items = []
    if vals_keep:
        items.append(("circle", "#2ca02c", "darkgreen", f"keep ({len(rounds_keep)})"))
    if vals_discard:
        items.append(("circle", "salmon", "red", f"discard ({len(rounds_discard)})"))
    if rounds_fail:
        items.append(("x", "black", "black", f"fail ({len(rounds_fail)})"))
    if best_rounds:
        items.append(("line", "#1f77b4", "#1f77b4", "best so far"))
    if ref_val is not None:
        items.append(("dashline", "#ff8c00", "#ff8c00", f"{ref_label} ({ref_val:.1f})"))

    for i, (kind, fill, stroke, label) in enumerate(items):
        ly = legend_y + i * 18
        if kind == "circle":
            p.append(
                f'<circle cx="{legend_x+6}" cy="{ly:.1f}" r="4" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="0.5"/>'
            )
        elif kind == "x":
            p.append(
                f'<path d="M {legend_x+1} {ly-3:.1f} L {legend_x+11} {ly+3:.1f} '
                f'M {legend_x+11} {ly-3:.1f} L {legend_x+1} {ly+3:.1f}" '
                f'stroke="{stroke}" stroke-width="1.5"/>'
            )
        elif kind == "line":
            p.append(
                f'<line x1="{legend_x}" y1="{ly:.1f}" x2="{legend_x+12}" y2="{ly:.1f}" '
                f'stroke="{stroke}" stroke-width="2"/>'
            )
        elif kind == "dashline":
            p.append(
                f'<line x1="{legend_x}" y1="{ly:.1f}" x2="{legend_x+12}" y2="{ly:.1f}" '
                f'stroke="{stroke}" stroke-width="1.5" stroke-dasharray="4,2"/>'
            )
        p.append(
            f'<text x="{legend_x+18}" y="{ly+3:.1f}" font-size="10">{_h(label)}</text>'
        )

    p.append("</svg>")
    return "\n".join(p)


def render_report(task_dir: str) -> str:
    """Build the full markdown report. Returns "" if no plottable data."""
    config = load_task_config(task_dir)
    if config is None:
        return ""
    primary = config.primary_metric
    lower_is_better = config.lower_is_better

    history = _load_history(task_dir)
    if not history:
        return ""

    progress = load_progress(task_dir) or {}

    # Baseline anchors speedup. baseline_source distinguishes PyTorch ref
    # from seed-fallback so the report labels the line honestly.
    raw_ref = progress.get("baseline_metric")
    ref_val: Optional[float] = (raw_ref if isinstance(raw_ref, (int, float))
                                and raw_ref > 0 else None)
    baseline_source = progress.get("baseline_source", "")
    ref_label = "PyTorch ref" if baseline_source == "ref" else "baseline"

    total_rounds = len(history)
    n_keep = sum(1 for r in history if r.get("decision") in ("KEEP", "SEED"))
    n_discard = sum(1 for r in history if r.get("decision") == "DISCARD")
    n_fail = sum(1 for r in history if r.get("decision") == "FAIL")

    seed_val = progress.get("seed_metric")
    best_val = progress.get("best_metric")

    # Find which round produced best_val (prefer earliest match — improvements
    # are strictly monotonic in best-so-far, so first match is the introducing
    # round).
    best_round: Optional[int] = None
    for rec in history:
        if rec.get("decision") in ("KEEP", "SEED"):
            v = rec.get("metrics", {}).get(primary)
            if isinstance(v, (int, float)) and isinstance(best_val, (int, float)):
                if abs(v - best_val) < 1e-9:
                    best_round = rec.get("round")
                    break

    improvement_str = "N/A"
    if (isinstance(seed_val, (int, float)) and isinstance(best_val, (int, float))
            and seed_val):
        if lower_is_better:
            pct = (seed_val - best_val) / seed_val * 100
            improvement_str = f"{pct:.1f}% reduction"
        else:
            pct = (best_val - seed_val) / seed_val * 100
            improvement_str = f"{pct:.1f}% increase"

    speedup_str = None
    best_speedup = progress.get("best_speedup")
    if isinstance(best_speedup, (int, float)) and best_speedup > 0:
        speedup_str = f"{best_speedup:.2f}x"

    task_name = progress.get("task") or os.path.basename(os.path.normpath(task_dir))
    svg = _generate_svg(history, primary, lower_is_better, ref_val, ref_label, task_name)

    # Key improvements: monotonic best-so-far steps among KEEP/SEED rounds.
    improvements = []
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

    # Multi-shape: list the tested shapes once at the top so readers know
    # what coverage the metric numbers represent. Single-shape tasks skip
    # the section entirely.
    case_descs: list = []
    for rec in history:
        d = (rec.get("metrics", {}) or {}).get("per_shape_descs")
        if isinstance(d, list) and d:
            case_descs = d
            break
    if len(case_descs) > 1:
        lines.append(f"## 测试形状 ({len(case_descs)})")
        lines.append("")
        for i, d in enumerate(case_descs):
            lines.append(f"{i}. {d}")
        lines.append("")

    if svg:
        lines.extend(["## Optimization Curve", "", svg, ""])

    if improvements:
        lines.extend([
            "## Key Improvements",
            "",
            f"| Round | Description | {primary} | Improvement |",
            f"|-------|-------------|{'---' * 4}|-------------|",
        ])
        for imp in improvements:
            from_v = (f"{imp['from']:.4f}" if isinstance(imp['from'], float)
                      else str(imp['from']))
            to_v = (f"{imp['to']:.4f}" if isinstance(imp['to'], float)
                    else str(imp['to']))
            delta_v = (f"{imp['delta']:.4f}" if isinstance(imp['delta'], float)
                       else str(imp['delta']))
            desc = _escape_md_cell(imp['desc'])
            lines.append(f"| R{imp['round']} | {desc} | {from_v} → {to_v} | -{delta_v} |")
        lines.append("")

    lines.extend([
        "## All Rounds",
        "",
        f"| Round | Description | Decision | {primary} |",
        f"|-------|-------------|----------|{'---' * 4}|",
    ])
    for rec in history:
        rnd = rec.get("round", "?")
        decision = rec.get("decision", "?")
        val = rec.get("metrics", {}).get(primary, "—")
        if isinstance(val, float):
            val = f"{val:.4f}"
        desc = _escape_md_cell((rec.get("description") or "")[:80])
        lines.append(f"| R{rnd} | {desc} | {decision} | {val} |")
    lines.append("")

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
