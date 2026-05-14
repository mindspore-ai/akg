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

from __future__ import annotations

import time
from typing import Any, Mapping

from akg_agents.op.dynamic_tune.cases.case import _CaseSpec


def _md_escape(s: str) -> str:
    """把 ``|`` 转义掉, 不让它把 markdown table 列拆错; 其它字符不动."""
    return str(s).replace("|", "\\|")


def _fmt_seconds(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f} s"
    return "n/a"


def _fmt_us(value: Any, *, width: int = 10) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):>{width}.2f}"
    return "n/a"


def _render_markdown_report(
    *,
    case_spec: _CaseSpec,
    summary: Mapping[str, Any],
) -> str:
    """把 summary.json 翻译成一个**自包含**的 case 报告 markdown.

    设计原则:
    - 单 case 一份, 直接给人看, 不依赖外部 dashboard.
    - 数据全部来自 summary, 无需回查 NPU; 这样 ``run.py`` 在不同机器上跑出来的 summary
      也能离线再渲染.
    - 重要结论 (verify pass/fail, end-to-end speedup, selector 选了哪个 config) 顶部就有,
      细节表 (matrix, per-shape 数字) 放后面给 deep dive.
    """
    lines: list[str] = []
    name = case_spec.name
    op_name = summary.get("op_name", name)
    device = summary.get("device", "?")
    impl_path = summary.get("impl_path", "?")
    artifact_path = summary.get("artifact_path", "?")
    report_ts = time.strftime("%Y-%m-%d %H:%M:%S")

    timings = summary.get("timings") or {}
    profile_block = summary.get("profile") or {}
    matrix_block = summary.get("autotune_matrix") or {}
    selector_block = matrix_block.get("selector") or {}

    verify_passed = bool(summary.get("verify_passed", False))
    end_to_end_speedup = profile_block.get("speedup")
    selector_kind = selector_block.get("kind", "?")
    decision_counts = selector_block.get("decision_counts") or {}
    if decision_counts:
        winning_cfg = max(decision_counts.items(), key=lambda kv: kv[1])[0]
        winning_str = f"{winning_cfg} ({decision_counts[winning_cfg]}/{sum(decision_counts.values())} shapes)"
    else:
        winning_str = "n/a"

    # ---------------- header ---------------- #
    lines.append(f"# {name} 报告")
    lines.append("")
    lines.append(f"- generated_at: {report_ts}")
    lines.append(f"- device: `{device}`")
    lines.append(f"- op_name: `{op_name}`")
    lines.append(f"- impl_path: `{impl_path}`")
    lines.append(f"- case_dir: `{case_spec.case_dir}`")
    lines.append(f"- artifact: `{artifact_path}`")
    if summary.get("manifest_dir"):
        lines.append(f"- manifest_dir: `{summary['manifest_dir']}`")
    lines.append("")

    # ---------------- TL;DR ---------------- #
    lines.append("## TL;DR")
    lines.append("")
    lines.append(
        f"- **verify**: {'PASS' if verify_passed else 'FAIL'}"
    )
    if isinstance(end_to_end_speedup, (int, float)):
        lines.append(f"- **end-to-end speedup vs PyTorch ref**: {end_to_end_speedup:.2f}×")
    else:
        lines.append("- end-to-end speedup vs PyTorch ref: n/a")
    lines.append(f"- **selector**: kind=`{selector_kind}`, top decision = {winning_str}")
    geomean_regret = selector_block.get("geomean_regret_ratio")
    if isinstance(geomean_regret, (int, float)):
        lines.append(
            f"- selector geomean regret_ratio: {geomean_regret:+.4f}"
            + (" (与逐 shape argmin 完全一致)" if abs(geomean_regret) < 1e-6 else "")
        )
    lines.append("")

    # ---------------- timings ---------------- #
    lines.append("## 阶段耗时")
    lines.append("")
    lines.append("| 阶段 | 时间 |")
    lines.append("|------|------|")
    for label, key in [
        ("autotune", "autotune_seconds"),
        ("verify", "verify_seconds"),
        ("profile", "profile_seconds"),
        ("**total**", "total_seconds"),
    ]:
        lines.append(f"| {label} | {_fmt_seconds(timings.get(key))} |")
    lines.append("")

    # ---------------- profile ---------------- #
    if profile_block:
        lines.append("## Profile (impl vs PyTorch reference)")
        lines.append("")
        method = profile_block.get("method", "akg_kernel_verifier")
        gen_us = profile_block.get("gen_time_us")
        base_us = profile_block.get("base_time_us")
        speedup = profile_block.get("speedup")
        path_used_impl = profile_block.get("path_used_impl")
        path_used_base = profile_block.get("path_used_base")
        meta_bits = [f"method=`{method}`"]
        if path_used_impl:
            meta_bits.append(f"impl_path=`{path_used_impl}`")
        if path_used_base:
            meta_bits.append(f"base_path=`{path_used_base}`")
        lines.append(", ".join(meta_bits))
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|----|")
        lines.append(f"| impl 平均 (μs) | {_fmt_us(gen_us)} |")
        lines.append(f"| base 平均 (μs) | {_fmt_us(base_us)} |")
        if isinstance(speedup, (int, float)):
            lines.append(f"| speedup | **{speedup:.2f}×** |")
        lines.append("")

        per_impl = profile_block.get("per_shape_impl_us") or []
        per_base = profile_block.get("per_shape_base_us") or []
        per_shapes = profile_block.get("shapes") or []
        if per_impl and per_base and per_shapes and len(per_impl) == len(per_base) == len(per_shapes):
            lines.append("### 每个 shape 的 profile 结果")
            lines.append("")
            lines.append("| shape | impl (μs) | base (μs) | speedup |")
            lines.append("|-------|-----------|-----------|---------|")
            for shape, impl_us, base_us_i in zip(per_shapes, per_impl, per_base):
                if isinstance(impl_us, (int, float)) and isinstance(base_us_i, (int, float)) and impl_us > 0:
                    s_speedup = base_us_i / impl_us
                    speedup_cell = f"{s_speedup:.2f}×"
                else:
                    speedup_cell = "n/a"
                lines.append(
                    f"| {_md_escape(tuple(shape))} | "
                    f"{_fmt_us(impl_us)} | {_fmt_us(base_us_i)} | {speedup_cell} |"
                )
            lines.append("")

    # ---------------- autotune matrix ---------------- #
    if matrix_block:
        lines.append("## Autotune Runtime Matrix")
        lines.append("")
        path_used = matrix_block.get("path_used", "?")
        n_shapes = matrix_block.get("n_shapes", "?")
        n_configs = matrix_block.get("n_configs", "?")
        lines.append(
            f"path=`{path_used}`, n_shapes={n_shapes}, n_configs={n_configs} (μs, lower is better)"
        )
        lines.append("")
        cfg_labels = [str(c) for c in matrix_block.get("configs") or []]
        shape_labels = [tuple(s) for s in matrix_block.get("shapes") or []]
        latencies = matrix_block.get("latencies_us") or []
        per_shape_best = matrix_block.get("per_shape_best_us") or []
        per_shape_best_cfg = matrix_block.get("per_shape_best_config") or []
        if cfg_labels and shape_labels and latencies:
            header = "| shape | " + " | ".join(_md_escape(c) for c in cfg_labels)
            header += " | best (μs) | best cfg |"
            sep = "|" + "|".join(["-------"] * (len(cfg_labels) + 3)) + "|"
            lines.append(header)
            lines.append(sep)
            for row_idx, shape in enumerate(shape_labels):
                row_lats = latencies[row_idx]
                best_col = (
                    int(min(range(len(row_lats)), key=lambda c: row_lats[c]))
                    if row_lats
                    else None
                )
                cells = []
                for col_idx, lat in enumerate(row_lats):
                    cell = _fmt_us(lat).strip()
                    if col_idx == best_col:
                        cell = f"**{cell}**"
                    cells.append(cell)
                lines.append(
                    "| "
                    + _md_escape(str(shape))
                    + " | "
                    + " | ".join(cells)
                    + f" | {_fmt_us(per_shape_best[row_idx]).strip() if row_idx < len(per_shape_best) else 'n/a'}"
                    + f" | {_md_escape(per_shape_best_cfg[row_idx]) if row_idx < len(per_shape_best_cfg) else 'n/a'} |"
                )
            lines.append("")

        # ---------------- per-config geomean latency ---------------- #
        per_cfg_us = matrix_block.get("per_config_geomean_us") or []
        best_overall = matrix_block.get("best_config_overall")
        selector_block = matrix_block.get("selector") or {}
        selector_geomean = selector_block.get("selector_geomean_us")
        if cfg_labels and per_cfg_us:
            lines.append("### 每个 config 的 geomean 延迟")
            lines.append("")
            lines.append("| config | geomean μs |")
            lines.append("|--------|------------|")
            for idx, label in enumerate(cfg_labels):
                mean_us = per_cfg_us[idx] if idx < len(per_cfg_us) else None
                marker = " ⭐ best_fixed" if label == best_overall else ""
                lines.append(
                    f"| {_md_escape(label)} | {_fmt_us(mean_us).strip()}{marker} |"
                )
            lines.append("")
            if selector_geomean is not None:
                best_fixed_us = min(per_cfg_us)
                tune_effect = best_fixed_us / selector_geomean if selector_geomean > 0 else 0
                lines.append(f"- **selector geomean**: {_fmt_us(selector_geomean).strip()}")
                lines.append(f"- **tune_effect** (best_fixed / selector): {tune_effect:.4f} (>1 = per-shape tuning beats fixed)")
                lines.append("")

    # ---------------- selector decisions ---------------- #
    per_shape_decisions = selector_block.get("per_shape") or []
    if per_shape_decisions:
        lines.append("## Selector 决策")
        lines.append("")
        lines.append(
            f"selector kind=`{selector_kind}`. "
            "regret_ratio = (selected/best - 1)，0 表示在该 shape 上选到了最优 config。"
        )
        lines.append("")
        lines.append("| shape | selected | selected μs | best μs | regret |")
        lines.append("|-------|----------|-------------|---------|--------|")
        for item in per_shape_decisions:
            shape = tuple(item.get("shape") or ())
            cfg = item.get("config")
            sel_us = item.get("latency_us")
            best_us = item.get("best_us")
            regret_ratio = item.get("regret_ratio")
            if cfg is None:
                lines.append(
                    f"| {_md_escape(str(shape))} | "
                    f"<error> | n/a | n/a | {_md_escape(item.get('error', ''))} |"
                )
                continue
            if item.get("matches_best"):
                regret_cell = "= best"
            elif isinstance(regret_ratio, (int, float)):
                regret_cell = f"{regret_ratio:+.2%}"
            else:
                regret_cell = "n/a"
            lines.append(
                f"| {_md_escape(str(shape))} | {_md_escape(cfg)} | "
                f"{_fmt_us(sel_us).strip()} | {_fmt_us(best_us).strip()} | {regret_cell} |"
            )
        lines.append("")
        if decision_counts:
            counts_str = ", ".join(
                f"`{cfg}`={n}"
                for cfg, n in sorted(decision_counts.items(), key=lambda kv: -kv[1])
            )
            lines.append(f"**decision_counts**: {counts_str}")
            lines.append("")

    # ---------------- verifier log ---------------- #
    verifier_log = summary.get("verifier_log_excerpt")
    if verifier_log:
        lines.append("## Verifier 日志摘录")
        lines.append("")
        lines.append("```")
        lines.append(str(verifier_log).strip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)
