# Copyright 2025 Huawei Technologies Co., Ltd
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

from __future__ import annotations

from typing import Any, Dict, List

from rich import box
from rich.console import Console
from rich.table import Table

from ai_kernel_generator.cli.cli.constants import DisplayStyle
from textual import log
from ..utils.i18n import t


class SummaryRenderer:
    def __init__(self, console: Console) -> None:
        self.console = console

    def display(
        self,
        *,
        result: dict,
        llm_records: List[Dict[str, Any]],
        node_timings: List[Dict[str, Any]],
        performance_history: List[Dict[str, Any]],
    ) -> None:
        self.console.print("\n")

        table = Table(title=t("summary.title"), box=box.DOUBLE, show_header=True)
        table.add_column(t("summary.col.item"), style=DisplayStyle.CYAN, width=20)
        table.add_column(t("summary.col.value"), style=DisplayStyle.YELLOW, width=50)

        table.add_row(t("summary.row.op_name"), result.get("op_name", "N/A"))

        verify_pass = result.get("verification_result")
        verify_text = (
            f"[{DisplayStyle.BOLD_GREEN}]PASS[/{DisplayStyle.BOLD_GREEN}]"
            if verify_pass
            else f"[{DisplayStyle.BOLD_RED}]FAIL[/{DisplayStyle.BOLD_RED}]"
        )
        table.add_row(t("summary.row.verify"), verify_text)

        # 错误信息（如果有）
        err = result.get("error")
        if isinstance(err, str) and err.strip():
            table.add_row(t("summary.row.error"), err.strip())

        try:
            if performance_history:
                table.add_row(
                    t("summary.row.perf_rounds"), str(len(performance_history))
                )
                for perf in performance_history:
                    round_idx = perf["round"]
                    gen_time = perf["gen_time"]
                    base_time = perf["base_time"]
                    speedup = perf["speedup"]

                    speedup_color = (
                        DisplayStyle.BOLD_GREEN
                        if speedup > 1.0
                        else DisplayStyle.YELLOW
                    )
                    perf_str = (
                        f"{t('presenter.perf.baseline')}: {base_time:.2f}µs | "
                        f"{t('presenter.perf.optimized')}: {gen_time:.2f}µs | "
                        f"{t('presenter.perf.speedup')}: [{speedup_color}]{speedup:.2f}x[/{speedup_color}]"
                    )
                    table.add_row(
                        t("summary.row.round_perf", round=round_idx), perf_str
                    )
            else:
                perf_result = result.get("metadata", {}).get("perf_result", {})
                if perf_result:
                    gen_time = perf_result.get("gen_time", 0.0)
                    base_time = perf_result.get("base_time", 0.0)
                    speedup = perf_result.get("speedup", 0.0)

                    if base_time > 0:
                        table.add_row(
                            t("summary.row.baseline_time"), f"{base_time:.2f} µs"
                        )
                    if gen_time > 0:
                        table.add_row(
                            t("summary.row.optimized_time"), f"{gen_time:.2f} µs"
                        )
                    if speedup > 0:
                        speedup_color = (
                            DisplayStyle.BOLD_GREEN
                            if speedup > 1.0
                            else DisplayStyle.YELLOW
                        )
                        table.add_row(
                            t("summary.row.speedup"),
                            f"[{speedup_color}]{speedup:.2f}x[/{speedup_color}]",
                        )
        except Exception as e:
            log.warning("[Summary] render perf section failed; skip", exc_info=e)

        table.add_row(
            t("summary.row.total_time"),
            f"{result.get('total_time', 0):.2f} {t('summary.unit.seconds')}",
        )

        meta = result.get("metadata") or {}
        if isinstance(meta, dict):
            log_dir = meta.get("log_dir")
            task_desc_path = meta.get("task_desc_path")
            kernel_code_path = meta.get("kernel_code_path")
            if log_dir:
                table.add_row(t("summary.row.log_dir"), str(log_dir))
            if task_desc_path:
                table.add_row(t("summary.row.task_desc_path"), str(task_desc_path))
            if kernel_code_path:
                table.add_row(t("summary.row.kernel_code_path"), str(kernel_code_path))

        self.console.print(table)

        # ========= Evolve 汇总（多轮 + 多并发）=========
        try:
            evolve = meta.get("evolve") if isinstance(meta, dict) else None
            if isinstance(evolve, dict):
                tasks = (
                    evolve.get("tasks") if isinstance(evolve.get("tasks"), dict) else {}
                )
                watch_tid = str(evolve.get("watch_task_id") or "").strip()
                round_snaps = (
                    evolve.get("round_snapshots")
                    if isinstance(evolve.get("round_snapshots"), dict)
                    else {}
                )

                # 轮次汇总
                if round_snaps:
                    self.console.print("\n")
                    rt = Table(
                        title=t("summary.evolve.round_summary"),
                        box=box.ROUNDED,
                        show_header=True,
                    )
                    rt.add_column("Round", style=DisplayStyle.CYAN, justify="right")
                    rt.add_column(
                        "Done/Total", style=DisplayStyle.YELLOW, justify="right"
                    )
                    rt.add_column("OK", style=DisplayStyle.BOLD_GREEN, justify="right")
                    rt.add_column("Fail", style=DisplayStyle.BOLD_RED, justify="right")
                    for r in sorted(round_snaps.keys()):
                        snap = round_snaps.get(r) or {}
                        rt.add_row(
                            str(r),
                            f"{snap.get('done', '-')}/{snap.get('total', '-')}",
                            str(snap.get("ok", "-")),
                            str(snap.get("fail", "-")),
                        )
                    self.console.print(rt)

                # 任务明细汇总
                if tasks:
                    self.console.print("\n")
                    tt = Table(
                        title=t("summary.evolve.task_detail"),
                        box=box.SIMPLE_HEAVY,
                        show_header=True,
                    )
                    tt.add_column("task_id", style=DisplayStyle.CYAN)
                    tt.add_column("status", style=DisplayStyle.YELLOW)
                    tt.add_column("verify", style=DisplayStyle.YELLOW)
                    tt.add_column(
                        t("summary.evolve.error_brief"), style=DisplayStyle.DIM
                    )
                    for tid in sorted(tasks.keys()):
                        info = tasks.get(tid) or {}
                        st = str(info.get("status") or "")
                        vr = info.get("verifier_result", None)
                        if vr is True:
                            vtxt = f"[{DisplayStyle.BOLD_GREEN}]PASS[/{DisplayStyle.BOLD_GREEN}]"
                        elif vr is False:
                            vtxt = f"[{DisplayStyle.BOLD_RED}]FAIL[/{DisplayStyle.BOLD_RED}]"
                        else:
                            vtxt = "-"
                        err = str(
                            info.get("verifier_error") or info.get("error") or ""
                        ).strip()
                        if len(err) > 120:
                            err = err[:120] + "..."
                        # 标注当前观察目标
                        tid_disp = tid
                        if watch_tid and tid == watch_tid:
                            tid_disp = f"[{DisplayStyle.BOLD_CYAN}]{tid}[/{DisplayStyle.BOLD_CYAN}]"
                        tt.add_row(tid_disp, st or "-", vtxt, err or "-")
                    self.console.print(tt)
        except Exception as e:
            log.warning("[Summary] render evolve section failed; skip", exc_info=e)

        token_table = Table(
            title=t("summary.tokens.title"), box=box.ROUNDED, show_header=True
        )
        token_table.add_column(t("summary.tokens.col.agent"), style=DisplayStyle.CYAN)
        token_table.add_column(
            t("summary.tokens.col.input"), style=DisplayStyle.CYAN, justify="right"
        )
        token_table.add_column("Reasoning", style=DisplayStyle.MAGENTA, justify="right")
        token_table.add_column(
            t("summary.tokens.col.output"), style=DisplayStyle.YELLOW, justify="right"
        )
        token_table.add_column(
            t("summary.tokens.col.total"), style=DisplayStyle.GREEN, justify="right"
        )
        token_table.add_column(
            t("summary.tokens.col.time"), style=DisplayStyle.YELLOW, justify="right"
        )

        def _fmt_token(value: Any) -> str:
            return str(value) if value is not None else "-"

        def _calc_display_total(p: Any, o: Any, raw_total: Any) -> Any:
            # 以 LLM API 返回的 total_tokens 为准；不要用 p+r+o 推导（reasoning 通常是 output 的子集）
            if raw_total is not None:
                return raw_total
            if any(v is not None for v in (p, o)):
                return (p or 0) + (o or 0)
            return None

        if llm_records:
            total_prompt = total_reasoning = total_output = 0
            total_total = 0
            has_any_total = False

            for rec in llm_records:
                p = rec.get("prompt_tokens")
                r = rec.get("reasoning_tokens")
                o = rec.get("output_tokens")
                raw_t = rec.get("raw_total_tokens")
                d = rec.get("duration")

                display_total = _calc_display_total(p, o, raw_t)

                total_prompt += p or 0
                total_reasoning += r or 0
                total_output += o or 0
                if raw_t is not None:
                    total_total += raw_t
                    has_any_total = True

                token_table.add_row(
                    rec.get("agent", "-"),
                    _fmt_token(p),
                    _fmt_token(r),
                    _fmt_token(o),
                    _fmt_token(display_total),
                    f"{d:.2f}" if d is not None else "-",
                )

            token_table.add_row(
                f"[{DisplayStyle.BOLD}]{t('summary.tokens.total_row')}[/{DisplayStyle.BOLD}]",
                _fmt_token(total_prompt if total_prompt else None),
                _fmt_token(total_reasoning if total_reasoning else None),
                _fmt_token(total_output if total_output else None),
                _fmt_token(total_total if has_any_total else None),
                "-",
            )
        else:
            token_table.add_row("-", "-", "-", "-", "-", "-")

        self.console.print(token_table)

        if node_timings:
            self.console.print("\n")
            time_table = Table(title=t("summary.node_timings.title"), box=box.ROUNDED)
            time_table.add_column(
                t("summary.node_timings.col.node"), style=DisplayStyle.CYAN
            )
            time_table.add_column(
                t("summary.node_timings.col.duration"),
                style=DisplayStyle.YELLOW,
                justify="right",
            )
            time_table.add_column(
                t("summary.node_timings.col.percentage"),
                style=DisplayStyle.GREEN,
                justify="right",
            )

            total_time = result.get("total_time", 0)
            for timing in node_timings:
                node = timing.get("node", "")
                duration = timing.get("duration", 0)
                percentage = (duration / total_time * 100) if total_time > 0 else 0

                time_table.add_row(node, f"{duration:.2f}s", f"{percentage:.1f}%")

            self.console.print(time_table)
