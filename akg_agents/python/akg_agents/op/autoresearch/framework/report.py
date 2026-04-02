"""
实验报告生成器 — 从 log.jsonl 生成带图表的报告

输出:
  - report.png: 优化过程图表 (主指标 + ref baseline)
  - report.md: Markdown 文本摘要

依赖: matplotlib (如果不可用则只生成文本报告)
"""

import os
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
    生成实验报告, 返回报告文件路径.

    Args:
        task_dir: 任务目录 (log.jsonl 所在位置).
        config: 任务配置.
        output_dir: 报告输出目录. 默认 None 表示写入 task_dir;
                    传入其他路径可避免污染 task_dir (如 --report 在 main 分支上运行时).

    包含:
      1. 主指标随轮次变化曲线 (区分 keep/discard/fail)
      2. 最优值演进线
      3. 文字摘要
    """
    out = output_dir or task_dir
    history = RoundLogger(task_dir, config).load_history()
    if not history:
        return _text_only_report(task_dir, config, history, reason="no history data",
                                 output_dir=out)

    report_png = os.path.join(out, "report.png")
    report_md = os.path.join(out, "report.md")

    # 尝试画图
    try:
        _generate_plots(history, config, report_png)
        has_plot = True
    except Exception as e:
        print(f"[report] WARNING: plot generation failed ({e}), text-only report")
        has_plot = False

    # 生成文字报告
    _generate_markdown(history, config, report_md, has_plot)

    if has_plot:
        print(f"[report] Report saved: {os.path.relpath(report_png)}")
    print(f"[report] Report saved: {os.path.relpath(report_md)}")
    return report_md


def _clamp_axis(ax, values: list[float], ref: float, lower_is_better: bool,
                margin: float = 0.1):
    """Clamp axis range around *ref* so outliers don't compress useful data.

    For lower-is-better: y ∈ [min(values)*0.9, ref * 1.1]
    For higher-is-better: y ∈ [ref * 0.9, max(values)*1.1]
    """
    if not values:
        return
    if lower_is_better:
        ax.set_ylim(min(values) * (1 - margin), ref * (1 + margin))
    else:
        ax.set_ylim(ref * (1 - margin), max(values) * (1 + margin))


def _generate_plots(history: list[dict], config: TaskConfig, output_path: str):
    """生成优化过程图表"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_key = config.primary_metric
    lower_is_better = config.lower_is_better

    # 分类数据
    rounds_keep, vals_keep = [], []
    rounds_discard, vals_discard = [], []
    rounds_fail = []
    best_rounds, best_vals = [], []
    current_best = None

    ref_val = _find_ref_latency(history)

    for rec in history:
        r = rec["round"]
        metrics = rec.get("metrics", {})
        val = metrics.get(metric_key)

        if not rec.get("correctness") or rec.get("constraint_violations"):
            rounds_fail.append(r)
            continue

        if val is None:
            continue

        if rec["accepted"]:
            rounds_keep.append(r)
            vals_keep.append(val)
            if current_best is None:
                current_best = val
            else:
                if lower_is_better:
                    current_best = min(current_best, val)
                else:
                    current_best = max(current_best, val)
            best_rounds.append(r)
            best_vals.append(current_best)
        else:
            rounds_discard.append(r)
            vals_discard.append(val)
            if current_best is not None:
                best_rounds.append(r)
                best_vals.append(current_best)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=120)
    if vals_keep:
        ax.scatter(rounds_keep, vals_keep, c="green", s=60, zorder=5,
                   label=f"keep ({len(rounds_keep)})", edgecolors="darkgreen", linewidths=0.5)
    if vals_discard:
        ax.scatter(rounds_discard, vals_discard, c="salmon", s=40, zorder=4,
                   label=f"discard ({len(rounds_discard)})", alpha=0.7,
                   edgecolors="red", linewidths=0.5)
    if rounds_fail:
        fail_y = max(vals_keep + vals_discard) * 1.1 if (vals_keep + vals_discard) else 1
        ax.scatter(rounds_fail, [fail_y] * len(rounds_fail), c="black", s=30,
                   marker="x", zorder=3, label=f"fail ({len(rounds_fail)})")
    if best_rounds and best_vals:
        ax.step(best_rounds, best_vals, where="post", color="blue", linewidth=2,
                alpha=0.8, label="best so far")

    # PyTorch ref baseline
    if ref_val is not None:
        ax.axhline(y=ref_val, color="orange", linestyle="--", alpha=0.7,
                   linewidth=1.5, label=f"PyTorch ref ({ref_val:.1f})")
        _clamp_axis(ax, vals_keep + vals_discard, ref_val, lower_is_better)

    ax.set_xlabel("Round")
    ax.set_ylabel(metric_key)
    direction = "lower is better" if lower_is_better else "higher is better"
    ax.set_title(f"{config.name} — {metric_key} ({direction})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _generate_markdown(history: list[dict], config: TaskConfig,
                       output_path: str, has_plot: bool):
    """生成 Markdown 文字报告"""
    metric_key = config.primary_metric
    total_rounds = len(history)

    # 统计
    n_keep = sum(1 for r in history if r["accepted"])
    n_fail = sum(1 for r in history
                 if not r["accepted"] and (not r.get("correctness") or r.get("constraint_violations")))
    n_discard = total_rounds - n_keep - n_fail

    # 找 baseline 和 best
    baseline_val = None
    best_val = None
    best_round = None
    for rec in history:
        val = rec.get("metrics", {}).get(metric_key)
        if val is None:
            continue
        if rec["accepted"]:
            if baseline_val is None:
                baseline_val = val
            if best_val is None or (config.lower_is_better and val < best_val) or \
               (not config.lower_is_better and val > best_val):
                best_val = val
                best_round = rec["round"]

    # PyTorch ref baseline (measured once at round 0)
    ref_baseline_val = _find_ref_latency(history)

    # 计算总改进
    improvement_str = "N/A"
    if baseline_val is not None and best_val is not None and baseline_val != 0:
        if config.lower_is_better:
            pct = (baseline_val - best_val) / baseline_val * 100
            improvement_str = f"{pct:.1f}% reduction"
        else:
            pct = (best_val - baseline_val) / baseline_val * 100
            improvement_str = f"{pct:.1f}% increase"

    # 总用时: 用首尾轮的 timestamp 差值计算真实 wall clock,
    # 而非各轮 duration_sec 之和 (后者只包含 eval 时间,
    # 不包含 LLM 调用、plan 提交等 turn 间的等待).
    import datetime as _dt
    total_wall_sec = None
    try:
        ts_first = history[0].get("timestamp", "")
        ts_last = history[-1].get("timestamp", "")
        if ts_first and ts_last:
            t0 = _dt.datetime.strptime(ts_first, "%Y-%m-%d %H:%M:%S")
            t1 = _dt.datetime.strptime(ts_last, "%Y-%m-%d %H:%M:%S")
            total_wall_sec = (t1 - t0).total_seconds()
    except Exception:
        pass
    if total_wall_sec is None or total_wall_sec <= 0:
        # Fallback: sum of per-round eval durations (underestimates).
        total_wall_sec = sum(rec.get("duration_sec", 0) for rec in history)
    if total_wall_sec >= 3600:
        wall_time_str = f"{total_wall_sec / 3600:.1f}h"
    elif total_wall_sec >= 60:
        wall_time_str = f"{total_wall_sec / 60:.1f}min"
    else:
        wall_time_str = f"{total_wall_sec:.0f}s"

    # speedup: computed from ref baseline and best latency
    best_speedup = None
    if ref_baseline_val is not None and best_val is not None and best_val > 0:
        best_speedup = ref_baseline_val / best_val

    # 收集所有 keep 轮的关键改进
    improvements = []
    prev_best = None
    for rec in history:
        if not rec["accepted"]:
            continue
        val = rec.get("metrics", {}).get(metric_key)
        if val is None:
            continue
        if prev_best is not None and val != prev_best:
            delta = prev_best - val if config.lower_is_better else val - prev_best
            if delta > 0:
                improvements.append({
                    "round": rec["round"],
                    "desc": rec["description"],
                    "from": prev_best,
                    "to": val,
                    "delta": delta,
                })
        prev_best = val if (prev_best is None or
                            (config.lower_is_better and val < prev_best) or
                            (not config.lower_is_better and val > prev_best)) else prev_best

    # 写报告
    lines = [
        f"# {config.name} — 优化报告",
        "",
        "## 总览",
        "",
        f"| 项目 | 值 |",
        f"|------|---|",
        f"| 任务 | {_escape_md_cell(config.description)} |",
        f"| 总轮次 | {total_rounds} |",
        f"| 接受 / 失败 / 丢弃 | {n_keep} / {n_fail} / {n_discard} |",
        f"| 主指标 | {metric_key} ({'越低越好' if config.lower_is_better else '越高越好'}) |",
        f"| 优化总用时 | {wall_time_str} |",
    ]
    if ref_baseline_val is not None:
        lines.append(f"| **PyTorch 参考基线** | **{ref_baseline_val:.2f} {metric_key.split('_')[-1]}** |")
    lines.extend([
        f"| Kernel Baseline (R0) | {baseline_val} |",
        f"| **最优结果** | **{best_val} (Round {best_round})** |",
        f"| 总改进 (vs Kernel Baseline) | {improvement_str} |",
    ])
    if best_speedup is not None:
        lines.append(f"| **最优加速比 (vs PyTorch)** | **{best_speedup:.2f}x** |")
    lines.append("")

    if has_plot:
        lines.extend([
            "## Optimization Curve",
            "",
            "![optimization curve](report.png)",
            "",
        ])

    if improvements:
        lines.extend([
            "## Key Improvements",
            "",
            f"| Round | Description | {metric_key} | Improvement |",
            f"|-------|-------------|{'---' * 4}|-------------|",
        ])
        for imp in improvements:
            from_v = f"{imp['from']:.4f}" if isinstance(imp['from'], float) else str(imp['from'])
            to_v = f"{imp['to']:.4f}" if isinstance(imp['to'], float) else str(imp['to'])
            delta_v = f"{imp['delta']:.4f}" if isinstance(imp['delta'], float) else str(imp['delta'])
            desc = _escape_md_cell(imp['desc'])
            lines.append(f"| R{imp['round']} | {desc} | {from_v} → {to_v} | -{delta_v} |")
        lines.append("")

    # 全部轮次一览
    lines.extend([
        "## All Rounds",
        "",
        f"| Round | Description | Correctness | {metric_key} | Status |",
        f"|-------|-------------|-------------|{'---' * 4}|--------|",
    ])
    for rec in history:
        if rec["accepted"]:
            status = "keep"
        elif not rec.get("correctness") or rec.get("constraint_violations"):
            status = "fail"
        else:
            status = "discard"
        correct = "PASS" if rec.get("correctness") else "FAIL"
        val = rec.get("metrics", {}).get(metric_key, "N/A")
        if isinstance(val, float):
            val = f"{val:.4f}"
        desc = _escape_md_cell(rec["description"][:50])
        lines.append(f"| R{rec['round']} | {desc} | {correct} | {val} | {status} |")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _text_only_report(task_dir: str, config: TaskConfig,
                      history: list[dict], reason: str,
                      output_dir: Optional[str] = None) -> str:
    """纯文本 fallback"""
    out = output_dir or task_dir
    report_md = os.path.join(out, "report.md")
    _generate_markdown(history, config, report_md, has_plot=False)
    print(f"[report] Text-only report ({reason}): {report_md}")
    return report_md
