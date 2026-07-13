"""Compact formatter for pipeline.py's eval result print.

Replaces ``print(f"[PIPELINE] Eval: correctness={c}, metrics={m}")`` —
that line dumped the entire metrics dict (per_shape_descs ~4KB, etc.).

Two functions:
  - ``summary_line(metrics, correctness)``: one-liner with the headline numbers
  - ``per_shape_table(metrics)``: aligned table of per-shape latencies + shapes

Caller prints summary_line first, then per_shape_table (if num_cases > 0).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Shape-desc cleaning
# ---------------------------------------------------------------------------

# Reference scaffolds emit shapes as:
#   "inputs[3]: shape=(1048576,) dtype=torch.float16, shape=None dtype=int, shape=None dtype=bool"
# The first chunk (the data tensor) is the only one carrying real
# shape info — the rest are scalar args (`dim`, `keepdim`-style) padded
# to a uniform "shape=None dtype=X" string by the scaffold. Strip them.
_SHAPE_NONE_RE = re.compile(r",\s*shape=None\s+dtype=\w+")
_TORCH_DTYPE_RE = re.compile(r"\btorch\.")
_INPUTS_PREFIX_RE = re.compile(r"^inputs\[\d+\]:\s*")


def _clean_shape_desc(desc: str) -> str:
    """Strip scaffold boilerplate so the table column stays narrow."""
    if not desc:
        return ""
    s = _INPUTS_PREFIX_RE.sub("", desc)
    s = _SHAPE_NONE_RE.sub("", s)
    s = _TORCH_DTYPE_RE.sub("", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summary_line(metrics: Dict[str, Any], correctness: bool,
                 tag: str = "PIPELINE") -> str:
    """One-liner. Falls back to ``correctness=...`` only when metrics is
    empty (eval crashed before producing numbers)."""
    if not metrics:
        return f"[{tag}] Eval: correctness={correctness} (no metrics)"

    parts = [f"correctness={correctness}"]
    lat = metrics.get("latency_us")
    if isinstance(lat, (int, float)):
        parts.append(f"latency={lat:.2f}us")
    spd = metrics.get("speedup_vs_ref")
    if isinstance(spd, (int, float)):
        parts.append(f"speedup={spd:.2f}x vs ref")
    nc = metrics.get("num_cases")
    if isinstance(nc, int) and nc > 0:
        parts.append(f"num_cases={nc}")
    return f"[{tag}] Eval: " + " | ".join(parts)


def per_shape_table(metrics: Dict[str, Any]) -> str:
    """Aligned per-shape table — the SINGLE per-shape view for both a passing
    round and a multi-case FAIL. Columns adapt to what's populated:

      - KEEP/DISCARD (all pass):  ``#  gen_us  base_us  speedup  shape``
      - multi-case FAIL:          ``#  status  gen_us  base_us  speedup  shape``

    On a FAIL the host folds the per-case verify result into the same
    ``per_shape_*`` arrays (``per_shape_status`` = PASS / failure_kind; gen/base
    are the free verify walls, ``—`` where unmeasured). "" when no per-shape
    rows so the caller can ``if table: print(table)``."""
    status: List[str] = list(metrics.get("per_shape_status") or [])
    gen: List[Any] = list(metrics.get("per_shape_gen_us") or [])
    base: List[Any] = list(metrics.get("per_shape_base_us") or [])
    descs: List[str] = list(metrics.get("per_shape_descs") or [])
    n = max(len(status), len(gen), len(descs))
    if n == 0:
        return ""

    def _num(xs: List[Any], i: int):
        v = xs[i] if i < len(xs) else None
        return v if isinstance(v, (int, float)) else None

    show_status = bool(status)
    show_gen = any(_num(gen, i) is not None for i in range(n))
    show_base = show_gen and any(_num(base, i) is not None for i in range(n))

    header = ["#"]
    if show_status:
        header.append("status")
    if show_gen:
        header.append("gen_us")
    if show_base:
        header += ["base_us", "speedup"]
    header.append("shape")
    rows: List[List[str]] = [header]

    for i in range(n):
        row = [f"#{i}"]
        if show_status:
            row.append(status[i] if i < len(status) else "—")
        g = _num(gen, i)
        if show_gen:
            row.append(f"{g:.2f}" if g is not None else "—")
        if show_base:
            b = _num(base, i)
            row.append(f"{b:.2f}" if b is not None else "—")
            row.append(f"{b / g:.2f}x" if (b and g) else "—")
        row.append(_clean_shape_desc(descs[i]) if i < len(descs) else "")
        rows.append(row)

    return _render_table(rows, indent="  ")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_table(rows: List[List[str]], indent: str = "") -> str:
    """Left-align text columns, right-align numeric columns. First row is
    header, gets a thin separator underneath."""
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    widths = [0] * ncols
    for r in rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(str(cell)))

    def _is_numeric(s: str) -> bool:
        s = s.strip().rstrip("x").rstrip("us").rstrip()
        if not s or s in ("—", "#"):
            return False
        if s.startswith("#") and s[1:].isdigit():
            return True
        try:
            float(s)
            return True
        except ValueError:
            return False

    out: List[str] = []
    for ri, r in enumerate(rows):
        cells = []
        for j in range(ncols):
            val = r[j] if j < len(r) else ""
            # Last column (shape) is always left-aligned, never padded
            # on the right (avoid trailing space).
            if j == ncols - 1:
                cells.append(val)
            else:
                if ri > 0 and _is_numeric(str(val)):
                    cells.append(str(val).rjust(widths[j]))
                else:
                    cells.append(str(val).ljust(widths[j]))
        out.append(indent + "  ".join(cells).rstrip())
        if ri == 0:
            # underline header — same width as columns separated by '  '
            sep_parts = ["-" * widths[j] for j in range(ncols)]
            out.append(indent + "  ".join(sep_parts).rstrip())
    return "\n".join(out)


# Shared eval-round plumbing for engine/baseline.py + engine/pipeline.py.

def write_artifact(path, data: Dict[str, Any]) -> str:
    """Dump ``data`` as an indented JSON artifact file; return its path. Single
    owner of the "write a JSON result file" step shared by batch verify.py
    (verify_results.json) and the FAIL report — neither hand-rolls its own
    write + json.dumps."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False),
                 encoding="utf-8")
    return str(p)


def eval_result_to_dict(result) -> Dict[str, Any]:
    """EvalResult -> the dict record_baseline / record_round consume."""
    d = {
        "outcome": result.outcome.value,
        "correctness": result.correctness,
        "metrics": result.metrics or {},
        "error": result.error,
        "error_source": result.error_source,
    }
    if not result.correctness or result.error:
        # signals already parsed by akg_eval from the full log — don't re-parse
        # the truncated tail; raw_output is already the tail.
        d["failure_signals"] = result.failure_signals or {}
        d["raw_output_tail"] = result.raw_output
        if result.fail_report:
            d["fail_report"] = result.fail_report
    return d


def print_eval_metrics(eval_data: Dict[str, Any], tag: str = "PIPELINE") -> None:
    """Summary line + per-shape table. On a multi-case FAIL the same table shows
    each shape's status + (verify-wall) gen/base/speedup — same path as a pass."""
    metrics = eval_data.get("metrics", {})
    print(summary_line(metrics, eval_data.get("correctness", False), tag), flush=True)
    table = per_shape_table(metrics)
    if table:
        print(table, flush=True)


def print_failure_signals(eval_data: Dict[str, Any], tag: str = "PIPELINE") -> None:
    """On failure: just point to the FAIL report file (full per-case + tracebacks
    + complete log + structured signals) for the agent to open with its file
    reader. The per-shape table is printed by ``print_eval_metrics``; the error
    line / signals / raw log are NOT echoed to stdout — they all live in the
    report. The raw log tail prints inline only as a last resort (no report)."""
    if eval_data.get("correctness", False) and not eval_data.get("error"):
        return
    report = eval_data.get("fail_report")
    if report:
        print(f"[{tag}] Full failure detail (per-shape status + tracebacks + "
              f"complete log + signals) written to: {report}", flush=True)
        print(f"[{tag}] ^ open it with your file-reading tool (Read), not "
              f"bash/cat — it is the full, untruncated record.", flush=True)
    elif eval_data.get("raw_output_tail"):
        print(f"[{tag}] Eval log tail (no report written):", flush=True)
        print(eval_data["raw_output_tail"], flush=True)
    elif eval_data.get("error"):
        print(f"[{tag}] Error: {eval_data['error']}", flush=True)
