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

"""Parse verify/profile subprocess logs into structured failure signals.

The eval subprocesses return multi-KB logs containing MLIR errors, Python
tracebacks, and NPU runtime errors. Diagnosing from a small structured
summary is much faster than from 4KB of raw text, both for
Claude-in-the-loop and for humans reading pipeline output.

This module pattern-matches known failure modes and emits an
`EvalDiagnostic` whose JSON shape is:

    {
        "primary": "ub_overflow",       # most actionable kind, or None
        "signals": [                    # one dict per matched pattern (first hit each)
            {"kind": ..., "<fields>": ..., "excerpt": ..., "hint": ...},
            ...
        ],
        "python_error": "RuntimeError: ...",  # last exception line, or None
    }

Add new patterns to `PATTERNS` below. Keep them ordered by specificity
(most specific first) because `finditer` stops at the first match per kind.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, List, Optional


@dataclass
class EvalDiagnostic:
    """Structured failure summary produced by `extract_failure_signals`.

    Single source of truth for the failure_signals schema that flows
    eval subprocess stderr -> failure_extractor -> baseline.py / pipeline.py
    -> record_round -> history.jsonl -> guidance prompt. Every consumer
    either takes an
    instance directly or reads its `to_dict()` form back off disk; the
    field set is owned here.

    `signals` stays a list of dicts because each pattern brings its own
    fields (ub_overflow has requested_bits / available_bits, npu_oom has
    tried_gib, ...). Promoting Signal to its own dataclass would require
    a discriminated union per kind and adds nothing for current consumers,
    which inspect `s["kind"]` plus the kind-specific keys.
    """
    primary: Optional[str] = None
    signals: List[dict] = field(default_factory=list)
    python_error: Optional[str] = None

    # ---- dict-compat read API ------------------------------------------
    # Existing readers in record_round / guidance use `diag.get("X", default)`
    # because the value off disk is a plain dict. Keeping `.get` here means
    # in-memory EvalDiagnostic objects pass the same accessors without a
    # rewrite at every read site.
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "EvalDiagnostic":
        if not data:
            return cls()
        return cls(
            primary=data.get("primary"),
            signals=list(data.get("signals") or []),
            python_error=data.get("python_error"),
        )

    @property
    def is_empty(self) -> bool:
        return not self.signals and not self.python_error


# Each pattern: (kind, compiled regex, extractor(match) -> dict, hint)
PATTERNS: list[tuple[str, re.Pattern, Callable[[re.Match], dict], str]] = [
    # --- Ascend MLIR compile: UB (unified buffer) overflow ---
    (
        "ub_overflow",
        re.compile(
            r"error: ub overflow, requires (\d+) bits while (\d+) bits available",
        ),
        lambda m: {
            "requested_bits": int(m.group(1)),
            "available_bits": int(m.group(2)),
            "requested_kb": int(m.group(1)) // 8192,
            "available_kb": int(m.group(2)) // 8192,
        },
        "Triton tile exceeds Ascend UB capacity. Shrink BLOCK_* constexprs, "
        "split large broadcast-multiply tiles, or convert tl.static_range -> "
        "range (static_range duplicates intermediate buffers per iteration).",
    ),
    # --- Ascend runtime: vector core trap (almost always misaligned load) ---
    (
        "aivec_exception",
        re.compile(
            r"aivec error, core id is (\d+), error code = (\d+)",
        ),
        lambda m: {"core_id": int(m.group(1)), "error_code": int(m.group(2))},
        "NPU vector core trap - usually a tl.load whose per-row stride isn't "
        "256-B aligned (i.e. stride-in-elements not divisible by 64 for fp32), "
        "or an out-of-bounds gather. Audit every tl.load address expression.",
    ),
    # --- Generic kernel task error (often paired with aivec_exception above) ---
    (
        "kernel_task_error",
        re.compile(
            r"Kernel task happen error, retCode=(0x[0-9a-fA-F]+)",
        ),
        lambda m: {"ret_code": m.group(1)},
        "Kernel-side runtime error. If also aivec_exception above, treat as "
        "the same root cause (misaligned or OOB load).",
    ),
    # --- NPU OOM ---
    (
        "npu_oom",
        re.compile(
            r"NPU out of memory\. Tried to allocate ([\d.]+) GiB.*?"
            r"([\d.]+) GiB total capacity.*?"
            r"([\d.]+) GiB already allocated",
            re.DOTALL,
        ),
        lambda m: {
            "tried_gib": float(m.group(1)),
            "total_gib": float(m.group(2)),
            "allocated_gib": float(m.group(3)),
        },
        "Too much HBM requested. Avoid full-tensor .contiguous() on unfold/ "
        "im2col views; process in chunks; cast to fp32 only where needed.",
    ),
    # --- ACL stream sync (downstream effect; real cause is usually above) ---
    (
        "acl_stream_error",
        re.compile(r"ACL stream synchronize failed, error code:(\d+)"),
        lambda m: {"error_code": int(m.group(1))},
        "ACL runtime error - usually a downstream effect of an earlier kernel "
        "crash (check aivec_exception / kernel_task_error above for the cause).",
    ),
    # --- Import / symbol missing ---
    (
        "import_error",
        re.compile(r"cannot import name '(\w+)'"),
        lambda m: {"missing_symbol": m.group(1)},
        "Kernel module must export `ModelNew` (name is fixed by the verify "
        "pipeline). Check class definition and file syntax.",
    ),
    # --- Grid size limit ---
    (
        "grid_too_large",
        re.compile(r"grid should be less than (\d+)"),
        lambda m: {"limit": int(m.group(1))},
        "Launch grid exceeds Ascend limit. Flatten multi-axis grids to 1-D, "
        "or move an axis inside the kernel as a serial loop.",
    ),
    # --- Triton MLIR compile failure (catch-all, runs after more specific UB) ---
    (
        "mlir_compile_error",
        re.compile(r"MLIRCompilationError"),
        lambda m: {},
        "Triton compilation failed - check earlier signals in this list for "
        "the specific cause (UB overflow, unsupported op, etc.).",
    ),
    # --- tbe module missing: ASC runtime environment issue ---
    (
        "tbe_module_missing",
        re.compile(
            r"ModuleNotFoundError: No module named 'tbe'",
        ),
        lambda m: {},
        "infra_fail: CANN environment not properly configured - ASCEND_HOME_PATH "
        "is unset or TBE module path is missing. This is an environment issue, "
        "not a kernel bug. Re-source set_env.sh or fix ASCEND_HOME_PATH.",
    ),
    # --- catlass / ascendc: numerical precision failure (AssertionError format) ---
    (
        "precision_fail",
        re.compile(
            r"AssertionError:\s*.*?"
            r"(?:outlier=(\d+)\s*/\s*cap=(\d+)|(\d+)\s*.*?hard_fail).*?"
            r"rtol=([\d.eE+-]+)\s*atol=([\d.eE+-]+).*?"
            r"mere=([\d.eE+-]+)\s*mare=([\d.eE+-]+)",
            re.DOTALL,
        ),
        lambda m: {
            "outlier_count": int(m.group(1) or m.group(3) or 0),
            "outlier_cap": int(m.group(2) or 0),
            "is_hard_fail": m.group(3) is not None,
            "rtol": float(m.group(4)),
            "atol": float(m.group(5)),
            "mean_rel_err": float(m.group(6)),
            "max_rel_err": float(m.group(7)),
        },
        "Numerical precision failure - output values exceed tolerance. "
        "Common causes: fp16 precision loss in intermediate computation; "
        "wrong reduce/accumulate order; missing cast to fp32 before "
        "summation; incorrect scale factor in epilogue.",
    ),
    # --- catlass / ascendc: worst mismatch location detail ---
    (
        "precision_fail_location",
        re.compile(
            r"位置\[[^\]]+\]:\s*"
            r"ref=([\d.eE+-]+)\s*impl=([\d.eE+-]+)\s*"
            r"abs_diff=([\d.eE+-]+)\s*"
            r"strict_tol=([\d.eE+-]+)",
        ),
        lambda m: {
            "ref_val": float(m.group(1)),
            "impl_val": float(m.group(2)),
            "abs_diff": float(m.group(3)),
            "strict_tol": float(m.group(4)),
        },
        "Worst mismatch location - shows the single most off element "
        "and how far it exceeds strict_tol. If abs_diff is only slightly "
        "above strict_tol, consider widening atol or switching to fp32 "
        "for the critical computation path.",
    ),
]

# Grab the last Python traceback line (the actual exception), not the full
# stack. Accepts both bare class names (`ValueError: ...`) and qualified ones
# (`triton.compiler.errors.CompilationError: ...`); without the dotted prefix
# Triton's wrapped errors fell through and r4-style FAILs surfaced no
# python_error at all.
_PYTHON_EXCEPTION_RE = re.compile(
    r"^(?:\w+\.)*[A-Z]\w*(?:Error|Exception|Warning):\s+.+$",
    re.MULTILINE,
)


def extract_failure_signals(raw_output: str,
                            max_excerpt: int = 200) -> EvalDiagnostic:
    """Scan combined eval log for known failure patterns.

    `raw_output` is the `log` field from eval_runner's verify/profile
    response (or a concatenation of both). Returns an EvalDiagnostic;
    callers that need the JSON form for stdout / history.jsonl call
    `.to_dict()` on the result.
    """
    if not raw_output:
        return EvalDiagnostic()

    signals: list[dict[str, Any]] = []
    seen_kinds: set[str] = set()

    for kind, regex, extractor, hint in PATTERNS:
        m = regex.search(raw_output)
        if not m or kind in seen_kinds:
            continue
        seen_kinds.add(kind)
        data = extractor(m)
        start = max(0, m.start() - 30)
        end = min(len(raw_output), m.end() + 30)
        excerpt = raw_output[start:end].replace("\n", " ").strip()[:max_excerpt]
        signals.append({"kind": kind, **data, "excerpt": excerpt, "hint": hint})

    exceptions = _PYTHON_EXCEPTION_RE.findall(raw_output)
    python_error = exceptions[-1][:240] if exceptions else None

    return EvalDiagnostic(
        primary=(signals[0]["kind"] if signals else None),
        signals=signals,
        python_error=python_error,
    )


def format_for_stdout(sig: dict[str, Any]) -> str:
    """Render a signals dict for pipeline.py's human-readable output.

    Empty string if nothing useful was extracted - callers can skip printing.
    """
    if not sig.get("signals") and not sig.get("python_error"):
        return ""
    lines = ["[PIPELINE] Eval failure signals:"]
    for s in sig["signals"]:
        kind = s["kind"]
        params = ", ".join(
            f"{k}={v}"
            for k, v in s.items()
            if k not in ("kind", "excerpt", "hint") and v is not None
        )
        header = f"  - {kind}"
        if params:
            header += f"  [{params}]"
        lines.append(header)
        if s.get("excerpt"):
            lines.append(f"      excerpt: {s['excerpt']}")
        if s.get("hint"):
            lines.append(f"      hint:    {s['hint']}")
    if sig.get("python_error"):
        lines.append(f"  - python_error: {sig['python_error']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys

    data = sys.stdin.read()
    result = extract_failure_signals(data)
    json.dump(result.to_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")
