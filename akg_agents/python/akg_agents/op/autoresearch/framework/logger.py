"""
结构化日志 — 双格式记录每轮实验结果

输出:
  - log.jsonl: 机器可读, 一行一个 JSON 对象
  - perf_log.md: 人类可读, Markdown 表格
"""

import json
import os
import re
import time
from typing import Optional

from .config import EvalResult, RoundRecord, TaskConfig


def _extract_error_reason(err: str, limit: int) -> str:
    """Extract the meaningful error from raw stderr.

    Raw eval output often starts with timestamps, PIDs, and device info
    like '2026-04-06-18:06:19 (PID:3957939, Device:5) ERR99999 ...'.
    We skip that noise and look for the actual exception.
    """
    # Try to find the last exception line (most specific)
    # e.g. "RuntimeError: xxx" or "ValueError: xxx"
    match = re.search(r'(\w+Error|\w+Exception):\s*.+', err)
    if match:
        return match.group(0)[:limit]
    # Try ERR* code line, skip timestamp/PID prefix
    match = re.search(r'(ERR\d+\s+.+)', err)
    if match:
        return match.group(1)[:limit]
    # Fallback: skip leading timestamp/PID lines, take first non-empty
    for line in err.splitlines():
        line = line.strip()
        if line and not re.match(r'^\d{4}-\d{2}-\d{2}', line) and line != 'Traceback (most recent call last):':
            return line[:limit]
    return err[:limit]


def _escape_md_cell(text: str) -> str:
    """Escape text for use inside a markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


class RoundLogger:
    """管理实验日志的读写"""

    def __init__(self, task_dir: str, config: TaskConfig):
        self.task_dir = task_dir
        self.config = config
        self.jsonl_path = os.path.join(task_dir, "log.jsonl")
        self.md_path = os.path.join(task_dir, "perf_log.md")
        # _init_md is called lazily on first log_round() to avoid writing
        # perf_log.md on construction (which would pollute main when
        # --status/--report/--eval-only skip the branch switch).
        self._md_initialized = os.path.exists(self.md_path)

    def _ensure_md(self):
        """初始化 Markdown 日志文件 (如果不存在). Called lazily."""
        if self._md_initialized:
            return
        self._md_initialized = True
        meta_lines = ""
        if self.config.metadata:
            meta_lines = "\n".join(f"{k}: {v}" for k, v in self.config.metadata.items())
            meta_lines = "\n" + meta_lines

        header = f"""# {self.config.name} — 优化记录

任务: {self.config.description}{meta_lines}
主指标: {self.config.primary_metric} ({'越低越好' if self.config.lower_is_better else '越高越好'})

## 优化记录

| Round | 描述 | 正确性 | {self.config.primary_metric} | 状态 | commit |
|-------|------|--------|{'---' * 3}|------|--------|
"""
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(header)

    def log_round(self, record: RoundRecord):
        """记录一轮实验结果到 JSONL 和 Markdown"""
        self._ensure_md()
        # JSONL
        entry = {
            "round": record.round_num,
            "description": record.description,
            "correctness": record.result.correctness,
            "accepted": record.accepted,
            "commit": record.commit_hash,
            "duration_sec": record.duration_sec,
            "error": record.result.error,
            "constraint_violations": record.constraint_violations,
            "metrics": record.result.metrics,
            "raw_output": (record.result.raw_output or "")[:self.config.agent.log_raw_output_truncate],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Markdown
        if record.accepted:
            status = "keep"
        elif not record.result.correctness or record.constraint_violations:
            status = "fail"
        else:
            status = "discard"
        correct_str = "PASS" if record.result.correctness else "FAIL"
        primary_val = record.result.metrics.get(self.config.primary_metric, "N/A")
        if isinstance(primary_val, float):
            primary_val = f"{primary_val:.4f}"
        commit_str = record.commit_hash[:7] if record.commit_hash else "-"

        desc = _escape_md_cell(record.description)
        row = f"| R{record.round_num} | {desc} | {correct_str} | {primary_val} | {status} | {commit_str} |\n"
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(row)

        # Auto-write ranking.md for hint subagent to read on demand
        try:
            ranking = self.get_performance_ranking()
            if ranking:
                ranking_path = os.path.join(self.task_dir, "ranking.md")
                with open(ranking_path, "w", encoding="utf-8") as f:
                    f.write(ranking)
        except Exception:
            pass

    def load_history(self) -> list[dict]:
        """读取全部历史记录"""
        if not os.path.exists(self.jsonl_path):
            return []
        records = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def get_best(self) -> Optional[dict]:
        """获取历史最优记录"""
        history = self.load_history()
        best = None
        for record in history:
            if not record.get("correctness"):
                continue
            if not record.get("accepted"):
                continue
            primary = record.get("metrics", {}).get(self.config.primary_metric)
            if primary is None:
                continue
            if best is None:
                best = record
            else:
                best_val = best["metrics"][self.config.primary_metric]
                if self.config.lower_is_better:
                    if primary < best_val:
                        best = record
                else:
                    if primary > best_val:
                        best = record
        return best

    def get_history_summary(self, last_n: int | None = None) -> str:
        if last_n is None:
            last_n = self.config.agent.history_summary_last_n
        """生成最近 N 轮的摘要文本, 用于 agent prompt"""
        history = self.load_history()
        if not history:
            return "No previous experiments."

        best = self.get_best()
        lines = []

        if best:
            lines.append(f"Best so far: R{best['round']} — {best['metrics'].get(self.config.primary_metric, 'N/A')}")
            lines.append("")

        lines.append(f"Recent {min(last_n, len(history))} rounds:")
        for record in history[-last_n:]:
            if record["accepted"]:
                status = "keep"
            elif not record.get("correctness") or record.get("constraint_violations"):
                status = "fail"
            else:
                status = "discard"
            primary = record.get("metrics", {}).get(self.config.primary_metric, "N/A")
            if isinstance(primary, float):
                primary = f"{primary:.4f}"
            suffix = ""
            cv = record.get("constraint_violations")
            if cv:
                suffix = f" [CONSTRAINT: {'; '.join(cv)}]"
            lines.append(f"  R{record['round']}: [{status}] {primary} — {record['description']}{suffix}")

        return "\n".join(lines)

    def get_performance_ranking(self, *, max_correct: int = 0,
                                max_failed: int = 0) -> str:
        """按主指标排序所有正确结果 + 失败记录, 供 agent 识别最优策略和避开死胡同.

        Args:
            max_correct: 保留 top N 条正确结果 (0 = 不限).
            max_failed: 保留最近 N 条失败记录 (0 = 不限).
        """
        history = self.load_history()
        if not history:
            return ""

        metric = self.config.primary_metric
        correct = []
        failed = []
        for r in history:
            if (r.get("correctness")
                    and not r.get("constraint_violations")
                    and r.get("metrics", {}).get(metric) is not None):
                correct.append(r)
            else:
                failed.append(r)

        lines = []
        desc_limit = self.config.agent.ranking_description_truncate

        # Reference value for speedup display
        ref_val = None
        if correct:
            baseline = [r for r in correct if r["round"] == 0]
            if baseline:
                ref_val = baseline[0]["metrics"].get(metric)

        if correct:
            correct.sort(
                key=lambda r: r["metrics"][metric],
                reverse=not self.config.lower_is_better,
            )
            lines.append(
                f"## Performance Ranking ({len(correct)} results, "
                f"{metric}, {'lower=better' if self.config.lower_is_better else 'higher=better'})"
            )
            lines.append(
                "NOTE: benchmark has variance — small differences may be noise. "
                "Strategies close to the best are worth combining or retrying."
            )
            # Build best-at-time map to explain discards
            lib = self.config.lower_is_better
            best_at_time = {}  # round -> best metric value when evaluated
            running_best = None
            for r in history:
                rn = r["round"]
                m = r.get("metrics", {}).get(metric)
                if r.get("correctness") and not r.get("constraint_violations") and m is not None:
                    if r["accepted"]:
                        running_best = m
                    best_at_time[rn] = running_best

            show_correct = correct[:max_correct] if max_correct else correct
            omitted = len(correct) - len(show_correct)
            for i, r in enumerate(show_correct, 1):
                val = r["metrics"][metric]
                speedup = ""
                if ref_val and ref_val > 0 and val > 0:
                    speedup = f" ({ref_val / val:.2f}x vs baseline)"
                if r["accepted"]:
                    tag = "KEEP"
                else:
                    bt = best_at_time.get(r["round"])
                    if bt is not None:
                        tag = f"discarded, best was {bt:.2f}"
                    else:
                        tag = "discarded"
                lines.append(
                    f"  {i}. R{r['round']}: {val:.2f}{speedup} [{tag}] — {r['description'][:desc_limit]}"
                )
            if omitted:
                lines.append(f"  ... ({omitted} lower-ranked results omitted)")

        if failed:
            err_limit = self.config.agent.ranking_error_truncate
            show_failed = failed[-max_failed:] if max_failed else failed
            omitted_f = len(failed) - len(show_failed)
            lines.append(f"\n## Failed Attempts ({len(failed)} — do NOT repeat)")
            if omitted_f:
                lines.append(f"  ... ({omitted_f} older failures omitted)")
            for r in show_failed:
                err = r.get("error")
                cv = r.get("constraint_violations")
                if err:
                    tag = _extract_error_reason(err, err_limit)
                elif not r.get("correctness"):
                    tag = "correctness mismatch"
                elif cv:
                    tag = "constraint: " + "; ".join(cv)
                else:
                    tag = "no benchmark result"
                lines.append(
                    f"  - R{r['round']}: {r['description'][:desc_limit]} [{tag}]"
                )

        return "\n".join(lines)

    def next_round_num(self) -> int:
        """返回下一轮的编号"""
        history = self.load_history()
        if not history:
            return 0
        return max(r["round"] for r in history) + 1
