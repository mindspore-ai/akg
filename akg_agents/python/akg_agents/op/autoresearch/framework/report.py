"""
实验报告生成器 — 从 log.jsonl 生成带内嵌 SVG 图表的 Markdown 报告

输出: report.md (含内嵌 SVG 优化曲线, 无外部依赖)
"""


# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring
import datetime as _dt
import os
from html import escape as _h
from typing import Optional

from .config import TaskConfig
from .logger import RoundLogger, _escape_md_cell


def _find_ref_latency(history: list[dict]) -> Optional[float]:
    """Find ref_latency_us from baseline round. Returns None if not found."""
    for rec in history:
        metrics = rec.get("metrics", {})
        ref_val = metrics.get("ref_latency_us") or metrics.get("_ref_latency_raw")
        if isinstance(ref_val, (int, float)) and ref_val > 0:
            return float(ref_val)
    return None


def generate_report(task_dir: str, config: TaskConfig,
                    output_dir: Optional[str] = None) -> str:
    """
    生成实验报告 (Markdown + 内嵌 SVG), 返回 report.md 路径.

    Args:
        task_dir: 任务目录 (log.jsonl 所在位置).
        config: 任务配置.
        output_dir: 报告输出目录. 默认 None 表示写入 task_dir;
                    传入其他路径可避免污染 task_dir (如 --report 在 main 分支上运行时).

    包含:
      1. 主指标随轮次变化曲线 (区分 keep/discard/fail)
      2. 最优值演进线 (best so far, step-post)
      3. PyTorch 参考基线水平线 + 加速比标注
      4. 文字摘要 + 全部轮次表格
    """
    out = output_dir or task_dir
    history = RoundLogger(task_dir, config).load_history()
    report_md = os.path.join(out, "report.md")
    svg = _generate_svg_chart(history, config) if history else ""
    _generate_markdown(history, config, report_md, svg)
    print(f"[report] Report saved: {os.path.relpath(report_md)}")
    return report_md


def _fmt_num(v: float) -> str:
    """Y-axis tick label formatter: pick precision by magnitude."""
    av = abs(v)
    if av >= 1000:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


# ---------------------------------------------------------------------------
# SVG chart
# ---------------------------------------------------------------------------

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


def _collect_series(history: list, metric_key: str,
                    lower_is_better: bool) -> _Series:
    """Bucket history into keep / discard / fail series plus the
    monotonic best-so-far curve."""
    s = _Series()
    current_best: Optional[float] = None
    for rec in history:
        r = rec["round"]
        val = rec.get("metrics", {}).get(metric_key)
        if not rec.get("correctness") or rec.get("constraint_violations"):
            s.rounds_fail.append(r)
            continue
        if val is None:
            continue
        if rec["accepted"]:
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
        else:
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


def _svg_header(W: int, H: int, task_name: str, metric_key: str,
                lower_is_better: bool) -> list:
    direction = "lower is better" if lower_is_better else "higher is better"
    title = _h(f"{task_name} — {metric_key} ({direction})")
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif" font-size="11">',
        f'<text x="{W/2:.1f}" y="22" text-anchor="middle" '
        f'font-size="13" font-weight="bold">{title}</text>',
    ]


def _svg_axes_and_grid(W: int, H: int, ML: int, MT: int, PW: int, PH: int,
                       xmin: float, xmax: float, ymin: float, ymax: float,
                       metric_key: str, sx, sy) -> list:
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
        f'transform="rotate(-90 16,{cy:.1f})">{_h(metric_key)}</text>'
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
    if not (ref_val is not None and ref_val > 0 and s.rounds_keep):
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


def _svg_legend_items(s: _Series, ref_val: Optional[float]) -> list:
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
                      f"PyTorch ref ({ref_val:.1f})"))
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


def _generate_svg_chart(history: list[dict], config: TaskConfig) -> str:
    """Render an inline SVG of the optimization curve. Returns "" if nothing to plot."""
    metric_key = config.primary_metric
    lower_is_better = config.lower_is_better
    ref_val = _find_ref_latency(history)
    s = _collect_series(history, metric_key, lower_is_better)
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

    def sx(r: float) -> float:
        return ML + (r - xmin) / (xmax - xmin) * PW

    def sy(v: float) -> float:
        return MT + (1 - (v - ymin) / (ymax - ymin)) * PH

    p: list[str] = []
    p.extend(_svg_header(W, H, config.name, metric_key, lower_is_better))
    p.extend(_svg_axes_and_grid(W, H, ML, MT, PW, PH,
                                xmin, xmax, ymin, ymax, metric_key, sx, sy))
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
        _svg_legend_items(s, ref_val),
        ML + PW + 14, MT + 14,
    ))
    p.append("</svg>")
    return "\n".join(p)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _baseline_and_best(history: list, metric_key: str,
                       lower_is_better: bool) -> tuple:
    """Walk history once for (baseline_val, best_val, best_round). Baseline
    is the first KEEP's metric (R0 / seed); best is the lower/higher
    KEEP depending on direction."""
    baseline_val = None
    best_val = None
    best_round = None
    for rec in history:
        val = rec.get("metrics", {}).get(metric_key)
        if val is None or not rec["accepted"]:
            continue
        if baseline_val is None:
            baseline_val = val
        if (best_val is None
                or (lower_is_better and val < best_val)
                or (not lower_is_better and val > best_val)):
            best_val = val
            best_round = rec["round"]
    return baseline_val, best_val, best_round


def _improvement_str(baseline_val, best_val, lower_is_better: bool) -> str:
    if not (baseline_val is not None and best_val is not None
            and baseline_val != 0):
        return "N/A"
    if lower_is_better:
        pct = (baseline_val - best_val) / baseline_val * 100
        return f"{pct:.1f}% reduction"
    pct = (best_val - baseline_val) / baseline_val * 100
    return f"{pct:.1f}% increase"


def _compute_wall_seconds(history: list) -> float:
    """Real wall-clock from the first/last timestamp. Per-round
    duration_sec only covers eval — not LLM calls or turn-to-turn idle
    — so it underestimates. Falls back to sum-of-durations when the
    timestamps are missing/malformed."""
    try:
        ts_first = history[0].get("timestamp", "") if history else ""
        ts_last = history[-1].get("timestamp", "") if history else ""
        if ts_first and ts_last:
            t0 = _dt.datetime.strptime(ts_first, "%Y-%m-%d %H:%M:%S")
            t1 = _dt.datetime.strptime(ts_last, "%Y-%m-%d %H:%M:%S")
            wall = (t1 - t0).total_seconds()
            if wall > 0:
                return wall
    except Exception:
        pass
    return sum(rec.get("duration_sec", 0) for rec in history)


def _format_wall_time(secs: float) -> str:
    if secs >= 3600:
        return f"{secs / 3600:.1f}h"
    if secs >= 60:
        return f"{secs / 60:.1f}min"
    return f"{secs:.0f}s"


def _count_status_buckets(history: list) -> tuple:
    total = len(history)
    n_keep = sum(1 for r in history if r["accepted"])
    n_fail = sum(
        1 for r in history
        if not r["accepted"]
        and (not r.get("correctness") or r.get("constraint_violations"))
    )
    return n_keep, n_fail, total - n_keep - n_fail, total


def _collect_improvements(history: list, metric_key: str,
                          lower_is_better: bool) -> list:
    """Monotonic best-so-far steps among KEEP rounds."""
    improvements: list = []
    prev_best: Optional[float] = None
    for rec in history:
        if not rec["accepted"]:
            continue
        val = rec.get("metrics", {}).get(metric_key)
        if val is None:
            continue
        if prev_best is not None and val != prev_best:
            delta = prev_best - val if lower_is_better else val - prev_best
            if delta > 0:
                improvements.append({
                    "round": rec["round"],
                    "desc": rec["description"],
                    "from": prev_best,
                    "to": val,
                    "delta": delta,
                })
        if (prev_best is None
                or (lower_is_better and val < prev_best)
                or (not lower_is_better and val > prev_best)):
            prev_best = val
    return improvements


def _render_overview(config: TaskConfig, total_rounds: int, n_keep: int,
                     n_fail: int, n_discard: int, wall_time_str: str,
                     ref_baseline_val, baseline_val, best_val, best_round,
                     improvement_str: str,
                     best_speedup: Optional[float]) -> list:
    metric_key = config.primary_metric
    direction_zh = "越低越好" if config.lower_is_better else "越高越好"
    lines = [
        f"# {config.name} — 优化报告",
        "",
        "## 总览",
        "",
        "| 项目 | 值 |",
        "|------|---|",
        f"| 任务 | {_escape_md_cell(config.description)} |",
        f"| 总轮次 | {total_rounds} |",
        f"| 接受 / 失败 / 丢弃 | {n_keep} / {n_fail} / {n_discard} |",
        f"| 主指标 | {metric_key} ({direction_zh}) |",
        f"| 优化总用时 | {wall_time_str} |",
    ]
    if ref_baseline_val is not None:
        unit = metric_key.split('_')[-1]
        lines.append(
            "| **PyTorch 参考基线** | "
            f"**{ref_baseline_val:.2f} {unit}** |"
        )
    lines.extend([
        f"| Kernel Baseline (R0) | {baseline_val} |",
        f"| **最优结果** | **{best_val} (Round {best_round})** |",
        f"| 总改进 (vs Kernel Baseline) | {improvement_str} |",
    ])
    if best_speedup is not None:
        lines.append(
            f"| **最优加速比 (vs PyTorch)** | **{best_speedup:.2f}x** |"
        )
    lines.append("")
    return lines


def _render_improvements_table(improvements: list, metric_key: str) -> list:
    if not improvements:
        return []
    lines = [
        "## Key Improvements",
        "",
        f"| Round | Description | {metric_key} | Improvement |",
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


def _round_status(rec: dict) -> str:
    if rec["accepted"]:
        return "keep"
    if not rec.get("correctness") or rec.get("constraint_violations"):
        return "fail"
    return "discard"


def _render_all_rounds(history: list, metric_key: str) -> list:
    lines = [
        "## All Rounds",
        "",
        f"| Round | Description | Correctness | {metric_key} | Status |",
        f"|-------|-------------|-------------|{'---' * 4}|--------|",
    ]
    for rec in history:
        status = _round_status(rec)
        correct = "PASS" if rec.get("correctness") else "FAIL"
        val = rec.get("metrics", {}).get(metric_key, "N/A")
        if isinstance(val, float):
            val = f"{val:.4f}"
        desc = _escape_md_cell(rec["description"][:50])
        lines.append(
            f"| R{rec['round']} | {desc} | {correct} | {val} | {status} |"
        )
    lines.append("")
    return lines


def _generate_markdown(history: list[dict], config: TaskConfig,
                       output_path: str, svg_chart: str):
    """生成 Markdown 文字报告 (svg_chart 为空字符串时跳过曲线区段)"""
    metric_key = config.primary_metric
    n_keep, n_fail, n_discard, total_rounds = _count_status_buckets(history)
    baseline_val, best_val, best_round = _baseline_and_best(
        history, metric_key, config.lower_is_better)
    ref_baseline_val = _find_ref_latency(history)
    improvement_str = _improvement_str(baseline_val, best_val,
                                       config.lower_is_better)
    wall_time_str = _format_wall_time(_compute_wall_seconds(history))
    best_speedup = (ref_baseline_val / best_val
                    if (ref_baseline_val is not None
                        and best_val is not None and best_val > 0)
                    else None)
    improvements = _collect_improvements(history, metric_key,
                                         config.lower_is_better)

    lines = _render_overview(config, total_rounds, n_keep, n_fail,
                             n_discard, wall_time_str, ref_baseline_val,
                             baseline_val, best_val, best_round,
                             improvement_str, best_speedup)
    if svg_chart:
        lines.extend(["## Optimization Curve", "", svg_chart, ""])
    lines.extend(_render_improvements_table(improvements, metric_key))
    lines.extend(_render_all_rounds(history, metric_key))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
