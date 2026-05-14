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

"""Diagnostics for tune outcomes."""

from __future__ import annotations

from typing import Any, Mapping


def config_label(config: Any) -> str:
    """Config -> compact readable label for logs and summaries."""

    params = getattr(config, "params", None)
    if params is None:
        return str(config)
    if len(params) == 0:
        return getattr(config, "config_id", "<config>")
    return ",".join(f"{name}={int(value)}" for name, value in params)


def summarize_matrix(matrix: Any) -> dict[str, Any]:
    """Flatten a LatencyMatrix into a JSON/report friendly summary."""

    import numpy as np  # type: ignore

    latencies = np.asarray(matrix.latencies_us, dtype=np.float64)
    n_shapes, n_configs = latencies.shape
    per_shape_best = latencies.min(axis=1, keepdims=True)
    per_config_geomean_latency = np.exp(np.log(latencies).mean(axis=0))

    config_labels = [config_label(cfg) for cfg in matrix.configs]
    shape_labels = [tuple(int(v) for v in shape) for shape in matrix.shapes]

    return {
        "shapes": [list(s) for s in shape_labels],
        "configs": config_labels,
        "latencies_us": latencies.tolist(),
        "per_config_geomean_us": per_config_geomean_latency.tolist(),
        "per_shape_best_us": per_shape_best.flatten().tolist(),
        "per_shape_best_config": [
            config_labels[i] for i in latencies.argmin(axis=1).tolist()
        ],
        "best_config_overall": config_labels[int(np.argmin(per_config_geomean_latency))],
        "path_used": getattr(matrix, "path_used", "?"),
        "n_shapes": int(n_shapes),
        "n_configs": int(n_configs),
    }


def summarize_selector(manifest: Any, matrix: Any) -> dict[str, Any]:
    """Run deployed selector against train shapes and summarize its decisions."""

    import numpy as np  # type: ignore

    from akg_agents.op.dynamic_tune.deploy.loader import DeployedSelector

    selector = DeployedSelector(manifest=manifest)
    latencies = np.asarray(matrix.latencies_us, dtype=np.float64)
    per_shape_best = latencies.min(axis=1)
    per_shape: list[dict[str, Any]] = []
    regret_ratios: list[float] = []
    decision_counts: dict[str, int] = {}
    for row_idx, shape in enumerate(matrix.shapes):
        shape_tuple = tuple(int(v) for v in shape)
        try:
            selected_cfg = selector.select_config(shape_tuple)
        except Exception as exc:  # pragma: no cover
            per_shape.append(
                {
                    "shape": list(shape_tuple),
                    "config": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        cfg_label = config_label(selected_cfg)
        decision_counts[cfg_label] = decision_counts.get(cfg_label, 0) + 1
        selected_col = None
        for col_idx, cfg in enumerate(matrix.configs):
            if cfg.config_id == selected_cfg.config_id:
                selected_col = col_idx
                break
        best_us = float(per_shape_best[row_idx])
        if selected_col is None:
            per_shape.append(
                {
                    "shape": list(shape_tuple),
                    "config": cfg_label,
                    "latency_us": None,
                    "best_us": best_us,
                    "regret_us": None,
                    "regret_ratio": None,
                }
            )
            continue
        sel_latency = float(latencies[row_idx, selected_col])
        regret = sel_latency - best_us
        regret_ratio = (sel_latency / best_us) - 1.0 if best_us > 0 else 0.0
        regret_ratios.append(regret_ratio)
        per_shape.append(
            {
                "shape": list(shape_tuple),
                "config": cfg_label,
                "latency_us": sel_latency,
                "best_us": best_us,
                "regret_us": regret,
                "regret_ratio": regret_ratio,
                "matches_best": sel_latency <= best_us + 1e-6,
            }
        )

    geomean_regret = (
        float(np.exp(np.log(1.0 + np.asarray(regret_ratios)).mean()) - 1.0)
        if regret_ratios
        else None
    )
    sel_latencies = [item["latency_us"] for item in per_shape if item.get("latency_us") is not None]
    selector_geomean_us = float(np.exp(np.log(np.asarray(sel_latencies)).mean())) if sel_latencies else None
    return {
        "kind": manifest.selector.kind,
        "per_shape": per_shape,
        "decision_counts": decision_counts,
        "geomean_regret_ratio": geomean_regret,
        "selector_geomean_us": selector_geomean_us,
    }


def print_selector_decisions(
    case_name: str, selector_summary: Mapping[str, Any]
) -> None:
    """Print selector per-shape decisions and aggregate regret."""

    kind = selector_summary.get("kind", "?")
    per_shape = list(selector_summary.get("per_shape") or [])
    if not per_shape:
        print(
            f"[autotune] selector decisions case={case_name} kind={kind}: "
            "(空, 没有 train_shape)"
        )
        return
    print(
        f"[autotune] selector decisions case={case_name} kind={kind} "
        f"(regret_ratio = (selected/best - 1), 0 = 选中本 shape 最优):"
    )
    shape_w = max(8, max(len(str(tuple(item["shape"]))) for item in per_shape))
    cfg_w = max(14, max(len(str(item.get("config") or "")) for item in per_shape))
    for item in per_shape:
        if item.get("config") is None:
            print(
                f"  {str(tuple(item['shape'])).ljust(shape_w)} -> "
                f"<error> {item.get('error', '')}"
            )
            continue
        sel_us = item.get("latency_us")
        best_us = item.get("best_us")
        regret_ratio = item.get("regret_ratio")
        matches = item.get("matches_best")
        mark = " (=best)" if matches else f" (regret={regret_ratio:+.2%})"
        sel_us_str = f"{sel_us:>10.2f}" if sel_us is not None else "       n/a"
        best_us_str = f"{best_us:>10.2f}" if best_us is not None else "       n/a"
        print(
            f"  {str(tuple(item['shape'])).ljust(shape_w)} -> "
            f"{str(item['config']).ljust(cfg_w)} "
            f"selected_us={sel_us_str} best_us={best_us_str}{mark}"
        )
    counts = selector_summary.get("decision_counts") or {}
    if counts:
        counts_str = ", ".join(
            f"{cfg}={n}" for cfg, n in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        print(f"  decision_counts: {counts_str}")
    geomean_regret = selector_summary.get("geomean_regret_ratio")
    if geomean_regret is not None:
        note = (
            "  (selector 与逐 shape argmin 完全一致)"
            if abs(geomean_regret) < 1e-6
            else ""
        )
        print(f"  geomean_regret_ratio={geomean_regret:+.4f}{note}")


def print_matrix_table(
    case_name: str, matrix: Any, summary: Mapping[str, Any]
) -> None:
    """Print a shape x config latency matrix and per-config geomean speedup."""

    config_labels: list[str] = list(summary["configs"])
    shape_labels: list[tuple[int, ...]] = [tuple(s) for s in summary["shapes"]]
    latencies: list[list[float]] = summary["latencies_us"]
    per_shape_best: list[float] = summary["per_shape_best_us"]
    per_shape_best_cfg: list[str] = summary["per_shape_best_config"]
    per_config_geomean_us: list[float] = summary["per_config_geomean_us"]
    path_used = summary.get("path_used", "?")

    shape_col_w = max(8, max((len(str(s)) for s in shape_labels), default=8))
    cfg_col_w = max(14, max((len(label) for label in config_labels), default=14) + 2)

    print(
        f"[autotune] runtime matrix case={case_name} "
        f"path={path_used} (us, lower is better):"
    )
    header = "  " + "shape".ljust(shape_col_w) + " | "
    header += " | ".join(label.center(cfg_col_w) for label in config_labels)
    header += " | " + "best_us".center(12) + " | " + "best_cfg".center(cfg_col_w)
    print(header)
    sep = "  " + "-" * shape_col_w + "-+-"
    sep += "-+-".join("-" * cfg_col_w for _ in config_labels)
    sep += "-+-" + "-" * 12 + "-+-" + "-" * cfg_col_w
    print(sep)
    for row_idx, shape in enumerate(shape_labels):
        row_cells = [
            f"{latencies[row_idx][col_idx]:>{cfg_col_w}.2f}"
            for col_idx in range(len(config_labels))
        ]
        line = (
            "  "
            + str(shape).ljust(shape_col_w)
            + " | "
            + " | ".join(row_cells)
            + f" | {per_shape_best[row_idx]:>12.2f}"
            + f" | {per_shape_best_cfg[row_idx].center(cfg_col_w)}"
        )
        print(line)

    best_fixed_idx = min(range(len(per_config_geomean_us)), key=lambda i: per_config_geomean_us[i])
    best_fixed_label = config_labels[best_fixed_idx]
    best_fixed_geomean_us = per_config_geomean_us[best_fixed_idx]

    print(
        f"[autotune] per-config geomean latency case={case_name} "
        f"(us, lower is better):"
    )
    for cfg_idx, label in enumerate(config_labels):
        mean_us = per_config_geomean_us[cfg_idx]
        marker = "  <-- best_fixed" if cfg_idx == best_fixed_idx else ""
        print(f"  cfg#{cfg_idx} {label}: geomean_us={mean_us:.2f}{marker}")

    selector_summary = summary.get("selector") or {}
    selector_geomean = selector_summary.get("selector_geomean_us")
    if selector_geomean is not None and selector_geomean > 0:
        tune_effect = best_fixed_geomean_us / selector_geomean
        print(
            f"[autotune] key metrics case={case_name}:\n"
            f"  best_fixed={best_fixed_label} geomean_us={best_fixed_geomean_us:.2f}\n"
            f"  selector  geomean_us={selector_geomean:.2f}\n"
            f"  tune_effect (best_fixed / selector) = {tune_effect:.4f}  "
            f"(>1 means per-shape tuning beats fixed config)"
        )


__all__ = [
    "config_label",
    "print_matrix_table",
    "print_selector_decisions",
    "summarize_matrix",
    "summarize_selector",
]
