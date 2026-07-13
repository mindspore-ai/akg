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
        "signals": [                    # one dict per matched pattern (last hit each)
            {"kind": ..., "<fields>": ..., "excerpt": ..., "hint": ...},
            ...
        ],
        "python_error": "RuntimeError: ...",  # last exception line, or None
        "tail_excerpt": "...",               # important lines from log tail
    }

Add new patterns to `PATTERNS` below. Keep them ordered by specificity
(most specific first) because the first emitted signal becomes `primary`.
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
    tail_excerpt: Optional[str] = None

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
            tail_excerpt=data.get("tail_excerpt"),
        )

    @property
    def is_empty(self) -> bool:
        return not self.signals and not self.python_error and not self.tail_excerpt


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
    # --- AscendC device compile: point at the .asc line, not the wrapper error ---
    (
        "ascendc_compile_error",
        re.compile(
            r"([^\n:]+\.asc):(\d+)(?::(\d+))?:\s*"
            r"(fatal error|error):\s*([^\n]+)",
            re.IGNORECASE,
        ),
        lambda m: {
            "file": m.group(1).strip(),
            "line": int(m.group(2)),
            "column": int(m.group(3)) if m.group(3) else None,
            "level": m.group(4).lower(),
            "message": m.group(5).strip(),
        },
        "AscendC device compile failed at the .asc source line. Fix the "
        "reported API/type/syntax issue before chasing wrapper or CMake errors.",
    ),
    # --- Host extension compile/link errors ---
    (
        "host_compile_error",
        re.compile(
            r"([^\n:]+\.(?:cc|cpp|cxx|c|h|hpp)):(\d+)(?::(\d+))?:\s*"
            r"(fatal error|error):\s*([^\n]+)",
            re.IGNORECASE,
        ),
        lambda m: {
            "file": m.group(1).strip(),
            "line": int(m.group(2)),
            "column": int(m.group(3)) if m.group(3) else None,
            "level": m.group(4).lower(),
            "message": m.group(5).strip(),
        },
        "Host-side extension compile failed. Check includes, generated "
        "operator registration code, and the torch_npu/CANN header environment.",
    ),
    (
        "link_undefined_symbol",
        re.compile(
            r"undefined symbol:\s*([^\s,]+)|undefined reference to [`']([^`'\n]+)[`']",
            re.IGNORECASE,
        ),
        lambda m: {
            "symbol": next((g.strip() for g in m.groups() if g), None)
        },
        "The extension built or loaded with an unresolved symbol. Align the "
        "registered torch.ops name, exported C++ symbol, and linked CANN/"
        "torch_npu libraries.",
    ),
    (
        "cmake_error",
        re.compile(r"CMake Error(?: at)?[^\n]*(?:\n\s{2,}[^\n]*)?", re.IGNORECASE),
        lambda m: {"message": m.group(0).strip()},
        "CMake configure/generate failed. Read the CMake Error block near the "
        "tail; it usually names the missing package, path, compiler, or target.",
    ),
    (
        "ascendc_arch_mismatch",
        re.compile(
            r"(?:unsupported|invalid|not support(?:ed)?).*?"
            r"(?:--npu-arch|NPU_ARCH|soc|arch|Ascend|910)|"
            r"(?:--npu-arch|NPU_ARCH|soc|arch).*?"
            r"(?:unsupported|invalid|not support(?:ed)?)",
            re.IGNORECASE,
        ),
        lambda m: {},
        "AscendC build selected an unsupported/mismatched NPU arch. Check "
        "task/config arch, worker device type, and the direct-invoke arch map.",
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
    (
        "vector_core_timeout",
        re.compile(
            r"Vector core execution timed out|vector core timeout|"
            r"runtime result\s*=\s*507034|error code is\s*507034",
            re.IGNORECASE,
        ),
        lambda m: {"error_code": 507034},
        "Ascend vector core timed out. Treat the current device context as "
        "suspect: stop the owning eval process, verify on a different healthy "
        "device, and reset or quarantine the timed-out device before resuming.",
    ),
    (
        "ascendc_core_exception",
        re.compile(
            r"vector core exception|aicore exception|aiv(?:ec)?(?:\s+core)? "
            r"exception|CCU instruction address check error|"
            r"MTE instruction address check error",
            re.IGNORECASE,
        ),
        lambda m: {"message": m.group(0).strip()},
        "AscendC runtime core trap. Audit GM offsets, DataCopy lengths, "
        "tail masks, UB aliases, blockIdx tiling, and any scalar address "
        "calculation used before the failing launch.",
    ),
    (
        "acl_error_code",
        re.compile(
            r"\bACL error\s*(\d{5,6})\b|"
            r"\b(?:aclrt\w+|rt\w+|acl[a-zA-Z_]+)\b.*?"
            r"(?:failed|error).*?(?:error code[:= ]*|retCode=)?(\d{5,6})",
            re.IGNORECASE,
        ),
        lambda m: {
            "error_code": int(next(g for g in m.groups() if g))
        },
        "CANN/ACL runtime error. If paired with a vector/AICore exception, "
        "treat the kernel crash as root cause and inspect plog/device logs.",
    ),
    (
        "err_code_line",
        re.compile(r"\b((?:ERR|EZ)\d{4,8})\b[:\s,\-]*(.{0,180})"),
        lambda m: {
            "code": m.group(1),
            "message": m.group(2).strip() or None,
        },
        "CANN emitted an ERR/EZ diagnostic near failure. Use the nearby tail "
        "lines as the actionable message; earlier repeated warnings are often noise.",
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
    (
        "segfault_abort",
        re.compile(r"\b(Segmentation fault|SIGSEGV|core dumped|Aborted)\b", re.IGNORECASE),
        lambda m: {"signal": m.group(1)},
        "Process crashed outside normal Python error handling. Check extension "
        "load/registration code, invalid pointers passed to CANN APIs, and "
        "whether a previous device error corrupted the process.",
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
    # --- AscendC direct-invoke: missing/invalid target arch ---
    (
        "ascendc_arch_config",
        re.compile(
            r"ascendc (?:requires config\['arch'\]|cannot derive "
            r"direct-invoke --npu-arch from ([^\n]+))",
        ),
        lambda m: {"arch": (m.group(1) or "").strip() or None},
        "AscendC direct-invoke needs a supported Ascend arch so the adapter "
        "can choose the CANN --npu-arch token. Check task/config arch and "
        "worker device selection.",
    ),
    # --- AscendC direct-invoke: CANN environment missing ---
    (
        "ascendc_env_missing",
        re.compile(
            r"ASCEND_HOME_PATH is not set|ASCEND_HOME_PATH.*(?:missing|unset)",
            re.IGNORECASE,
        ),
        lambda m: {},
        "infra_fail: CANN environment is not sourced. Source set_env.sh/env.sh "
        "so ASCEND_HOME_PATH, compiler paths, and torch_npu libraries are visible.",
    ),
    (
        "torch_npu_import_failed",
        re.compile(
            r"(?:ImportError|ModuleNotFoundError):.*(?:torch_npu|No module named 'torch_npu')|"
            r"DLL load failed while importing torch_npu",
            re.IGNORECASE,
        ),
        lambda m: {},
        "infra_fail: torch_npu is not importable in the eval process. Activate "
        "the NPU Python environment before running WA/worker eval.",
    ),
    # --- AscendC-CATLASS: project/env/build failures ---
    (
        "catlass_arch_config",
        re.compile(
            r"ascendc_catlass requires config\['arch'\]|"
            r"Unsupported arch for ascendc_catlass:\s*([^\.\n]+)",
        ),
        lambda m: {"arch": (m.group(1) or "").strip() or None},
        "ascendc_catlass needs a supported Ascend arch so the adapter can "
        "choose the CATLASS CMake arch token. Check task/config arch and worker "
        "device selection.",
    ),
    (
        "catlass_root_missing",
        re.compile(r"CATLASS_ROOT is not set", re.IGNORECASE),
        lambda m: {},
        "infra_fail: CATLASS_ROOT is missing. Set task.yaml catlass.root, "
        "config catlass_root, CATLASS_ROOT, or install thirdparty/catlass.",
    ),
    (
        "catlass_project_missing",
        re.compile(
            r"catlass_op_src not found or not a directory.*?Got:\s*([^\n]+)|"
            r"catlass_op directory not found:\s*([^\n]+)|"
            r"ascendc_catlass kernel handoff must be a catlass_op directory; got ([^\n]+)|"
            r"ascendc_catlass kernel handoff is a directory; expected sibling .*? at ([^\n]+)",
            re.DOTALL,
        ),
        lambda m: {
            "path": next(
                (g.strip() for g in m.groups() if g and g.strip()),
                None,
            )
        },
        "ascendc_catlass expects a catlass_op project tree plus the sibling "
        "kernel.py wrapper. Fix task layout or catlass.op_dir/catlass_op_src.",
    ),
    (
        "catlass_cmake_build_failed",
        re.compile(r"catlass cmake build failed"),
        lambda m: {},
        "CATLASS build failed. Inspect the CMake/make log for missing CANN, "
        "CATLASS_ROOT, torch_npu headers/libs, arch mismatch, or .asc compile errors.",
    ),
    (
        "catlass_load_library_failed",
        re.compile(
            r"(?:OSError|RuntimeError):\s+.*(?:libcatlass\.so|catlass).*?"
            r"(?:load_library|cannot open shared object file|DLL load failed|undefined symbol)",
            re.IGNORECASE | re.DOTALL,
        ),
        lambda m: {},
        "torch.ops.load_library failed for the CATLASS extension. Check that "
        "libcatlass.so was built for the active Python/torch_npu/CANN environment.",
    ),
    (
        "catlass_torch_op_missing",
        re.compile(
            r"_OpNamespace.*['\"]catlass['\"].*object has no attribute '([^']+)'|"
            r"torch\.ops\.catlass\.(\w+).*?(?:not exist|no attribute)",
            re.IGNORECASE | re.DOTALL,
        ),
        lambda m: {"op": (m.group(1) or m.group(2) or "").strip() or None},
        "CATLASS extension loaded but did not register the torch.ops.catlass "
        "symbol that kernel.py calls. Align TORCH_LIBRARY/op name with ModelNew.forward().",
    ),
    # --- AscendC direct-invoke: project tree missing or malformed ---
    (
        "ascendc_project_missing",
        re.compile(
            r"ascendc project directory not found:\s*([^\n]+)|"
            r"ascendc_op_src not found or not a directory.*?Got:\s*([^\n]+)",
            re.DOTALL,
        ),
        lambda m: {"path": (m.group(1) or m.group(2) or "").strip()},
        "AscendC direct-invoke expects --kernel to point at the ascendc_op "
        "project directory with sibling kernel.py. Fix the task layout or "
        "task.yaml editable file paths.",
    ),
    (
        "ascendc_cmakelists_missing",
        re.compile(r"ascendc CMakeLists\.txt not found:\s*([^\n]+)"),
        lambda m: {"path": m.group(1).strip()},
        "The ascendc_op project is missing CMakeLists.txt. Restore the direct-"
        "invoke project tree instead of editing only kernel.py.",
    ),
    # --- AscendC direct-invoke: cmake/build failures ---
    (
        "ascendc_cmake_configure_failed",
        re.compile(r"ascendc cmake configure failed"),
        lambda m: {},
        "CMake configure failed. Inspect the configure log for missing CANN, "
        "torch_npu, include paths, compiler, or invalid CMake target wiring.",
    ),
    (
        "ascendc_cmake_missing",
        re.compile(
            r"FileNotFoundError:.*(?:'cmake'|\"cmake\"|cmake)",
            re.IGNORECASE,
        ),
        lambda m: {},
        "infra_fail: cmake executable was not found. Install CMake or activate "
        "the environment that provides it before running AscendC direct-invoke eval.",
    ),
    (
        "ascendc_cmake_build_failed",
        re.compile(r"ascendc cmake build failed"),
        lambda m: {},
        "AscendC build failed. Inspect the build log for .asc compile errors, "
        "npu-arch mismatch, missing headers, or host/device pass issues.",
    ),
    (
        "ascendc_no_shared_library",
        re.compile(r"ascendc build finished without a shared library in ([^\n]+)"),
        lambda m: {"build_dir": m.group(1).strip()},
        "CMake completed but produced no .so under build/. Ensure the PyTorch "
        "extension target is SHARED and not only an executable/run.sh target.",
    ),
    # --- AscendC direct-invoke: Python wrapper / torch extension load ---
    (
        "ascendc_extension_missing",
        re.compile(r"no AscendC extension found under ([^\n]+)"),
        lambda m: {"build_dir": m.group(1).strip()},
        "kernel.py could not find the built .so. Make _load() search build/ "
        "recursively or align the CMake library output name/path.",
    ),
    (
        "ascendc_load_library_failed",
        re.compile(
            r"(?:OSError|RuntimeError):\s+.*(?:load_library|cannot open shared "
            r"object file|DLL load failed|undefined symbol)",
            re.IGNORECASE,
        ),
        lambda m: {},
        "torch.ops.load_library failed. Check that the selected .so links "
        "torch_npu/CANN libs and that LD_LIBRARY_PATH/PATH includes them.",
    ),
    (
        "ascendc_torch_op_missing",
        re.compile(
            r"_OpNamespace.*object has no attribute '([^']+)'|"
            r"torch\.ops\.(\w+)\.(\w+).*?(?:not exist|no attribute)",
            re.IGNORECASE | re.DOTALL,
        ),
        lambda m: {
            "op": (m.group(1) or m.group(3) or "").strip() or None,
            "namespace": (m.group(2) or "").strip() or None,
        },
        "The extension loaded but did not register the torch.ops symbol that "
        "kernel.py calls. Align TORCH_LIBRARY namespace/op name with ModelNew.forward().",
    ),
    # --- Common DSL front-end/env failures ---
    (
        "dsl_unsupported_framework",
        re.compile(
            r"(?P<dsl>TileLang Ascend|TileLang CUDA|ascendc direct-invoke) "
            r"currently (?:only supports framework='(?P<expected1>[^']+)'|"
            r"supports (?P<expected2>\w+) only), "
            r"got (?:framework=)?'(?P<actual>[^']+)'",
        ),
        lambda m: {
            "dsl": m.group("dsl"),
            "expected": m.group("expected1") or m.group("expected2"),
            "actual": m.group("actual"),
        },
        "The selected DSL adapter was invoked with an unsupported framework. "
        "Fix task.yaml framework/dsl instead of debugging kernel code.",
    ),
    (
        "dsl_module_missing",
        re.compile(
            r"ModuleNotFoundError: No module named '"
            r"((?:triton|tilelang|pypto|swft)(?:\.[^']*)?)'",
        ),
        lambda m: {"module": m.group(1)},
        "infra_fail: the selected DSL runtime package is not importable in "
        "the eval process. Activate the matching environment or install the DSL.",
    ),
    (
        "python_syntax_error",
        re.compile(r"SyntaxError:\s*([^\n]+)"),
        lambda m: {"message": m.group(1).strip()},
        "Generated kernel/wrapper has invalid Python syntax. Fix syntax before "
        "reasoning about DSL/compiler behavior.",
    ),
    # --- Grid size limit ---
    (
        "grid_too_large",
        re.compile(r"grid should be less than (\d+)"),
        lambda m: {"limit": int(m.group(1))},
        "Launch grid exceeds Ascend limit. Flatten multi-axis grids to 1-D, "
        "or move an axis inside the kernel as a serial loop.",
    ),
    # --- Triton compile-time assertion / front-end compile failure ---
    (
        "triton_compile_failure",
        re.compile(
            r"triton\.compiler\.errors\."
            r"(CompileTimeAssertionFailure|CompilationError|CompilationAssertionFailure)",
        ),
        lambda m: {"error_type": m.group(1)},
        "Triton JIT compilation failed before launch. Inspect the reported "
        "source location for failing tl.static_assert, invalid constexpr/meta "
        "assumption, or unsupported frontend construct.",
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
    # --- CANN-Bench precision: our assert_outputs line (strongest signal) ---
    (
        "precision_fail",
        re.compile(
            r"\[cannbench_precision\][^\n]*?mere=([\d.eE+-]+)\s+mare=([\d.eE+-]+)"
            r"\s+threshold=([\d.eE+-]+)\s+passed=False"
        ),
        lambda m: {
            "mean_rel_err": float(m.group(1)),
            "max_rel_err": float(m.group(2)),
            "threshold": float(m.group(3)),
        },
        "Numerical precision failure (CANN-Bench MERE/MARE): output exceeds "
        "the per-dtype threshold. A correctness miss, not a crash — check "
        "accumulation precision (fp32), reduce order, scale/epsilon, casts.",
    ),
    # --- CANN-Bench precision: native compare floating-mismatch assertion ---
    (
        "precision_fail",
        re.compile(
            r"compare:\s*floating mismatch:.*?relative_error=([\d.eE+-]+)",
            re.DOTALL,
        ),
        lambda m: {"max_rel_err": float(m.group(1))},
        "Numerical precision failure: kernel output diverges from the "
        "reference (relative_error shown). A correctness miss, not a crash.",
    ),
    # --- catlass / ascendc: numerical precision failure (AssertionError format) ---
    (
        "precision_fail",
        re.compile(
            r"AssertionError:\s*.*?"
            r"(?:outlier=(\d+)\s*/\s*cap=(\d+)|"
            r"(?:\u5b58\u5728\s*)?(\d+)\s*"
            r"(?:(?:\u4e2a\u5143\u7d20|elements?)[^\n\r]*?hard_fail|hard_fail)).*?"
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
            r"(?:\u4f4d\u7f6e|位置|浣嶇疆)\[[^\]]+\]:\s*"
            r"ref=([\d.eE+-]+)\s*impl=([\d.eE+-]+)\s*"
            r"abs_diff=([\d.eE+-]+)\s*"
            r"(strict|relaxed)_tol=([\d.eE+-]+)",
        ),
        lambda m: {
            "ref_val": float(m.group(1)),
            "impl_val": float(m.group(2)),
            "abs_diff": float(m.group(3)),
            "tolerance_kind": m.group(4),
            "tolerance": float(m.group(5)),
            "strict_tol": float(m.group(5)) if m.group(4) == "strict" else None,
            "relaxed_tol": float(m.group(5)) if m.group(4) == "relaxed" else None,
        },
        "Worst mismatch location - shows the single most off element "
        "and how far it exceeds strict_tol. If abs_diff is only slightly "
        "above strict_tol, consider widening atol or switching to fp32 "
        "for the critical computation path.",
    ),
    (
        "precision_mismatch_stats",
        re.compile(
            r"(?:max_abs(?:_diff)?|max_rel(?:_err)?|mean_rel(?:_err)?|"
            r"mere|mare|rtol|atol)\s*[=:]\s*[\d.eE+-]+[^\n]*",
            re.IGNORECASE,
        ),
        lambda m: {"line": m.group(0).strip()},
        "Numerical mismatch summary. Use the worst-case shape/index and "
        "decide whether the fix is algorithmic, accumulation precision, "
        "dtype conversion, or tolerance drift.",
    ),
]

# Grab the last Python traceback line (the actual exception), not the full
# stack. Accepts both bare class names (`ValueError: ...`) and qualified ones
# (`triton.compiler.errors.CompilationError: ...`); without the dotted prefix
# Triton's wrapped errors fell through and r4-style FAILs surfaced no
# python_error at all.
_PYTHON_EXCEPTION_RE = re.compile(
    r"^(?:\w+\.)*[A-Z]\w*(?:Error|Exception|Warning|Failure):\s+.+$",
    re.MULTILINE,
)

_TAIL_IMPORTANT_RE = re.compile(
    r"\b(?:error|exception|traceback|failed|failure|fatal|assertion)\b|"
    r"\b(?:ERR|EZ)\d{4,8}\b|"
    r"ACL error|aclrt|rtStream|vector core|aicore|aivec|aiv|kernel task|CCU|"
    r"\.asc:\d+|undefined symbol|undefined reference|No such file|"
    r"Segmentation fault|SIGSEGV|core dumped|"
    r"max_abs|abs_diff|strict_tol|mean_rel|rtol|atol",
    re.IGNORECASE,
)

_TAIL_NOISE_RE = re.compile(
    r"^\s*(?:"
    r"warning:|cmake warning|--\s|"
    r"\[INFO\]|\[WARNING\]|INFO:|"
    r"W\d{4,}|note:|In file included from|"
    r"ninja:|Scanning dependencies|\[\d+/\d+\]"
    r")",
    re.IGNORECASE,
)


def _last_match(regex: re.Pattern, text: str) -> Optional[re.Match]:
    """Return the last regex match in text.

    Build/eval logs often start with hundreds of harmless warnings and end
    with the only actionable error. Pattern matching from the tail prevents an
    early stale warning from becoming the structured root cause.
    """
    last: Optional[re.Match] = None
    for candidate in regex.finditer(text):
        last = candidate
    return last


def _line_excerpt(text: str,
                  start: int,
                  end: int,
                  *,
                  context_lines: int = 2,
                  max_chars: int = 500) -> str:
    """Return nearby log lines around a match, trimmed for prompt output."""
    lines = text.splitlines()
    if not lines:
        return text[max(0, start - 30):min(len(text), end + 30)].strip()

    start_line = text[:start].count("\n")
    end_line = text[:end].count("\n")
    lo = max(0, start_line - context_lines)
    hi = min(len(lines), end_line + context_lines + 1)
    excerpt = "\n".join(line.rstrip() for line in lines[lo:hi] if line.strip())
    excerpt = excerpt.strip()
    if len(excerpt) <= max_chars:
        return excerpt
    matched = text[start:end].replace("\n", " ").strip()
    if matched:
        return matched[:max_chars]
    return excerpt[-max_chars:].lstrip()


def _interesting_tail(text: str,
                      *,
                      max_lines: int = 12,
                      max_chars: int = 1200) -> Optional[str]:
    """Keep high-signal lines from the end of a noisy compile/runtime log."""
    if not text:
        return None

    picked: list[str] = []
    saw_important = False
    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        important = bool(_TAIL_IMPORTANT_RE.search(line))
        if important:
            picked.append(line)
            saw_important = True
        elif saw_important and not _TAIL_NOISE_RE.search(line):
            picked.append(line)
        elif not saw_important and not _TAIL_NOISE_RE.search(line):
            # Last non-warning line after the error is often the subprocess
            # wrapper's summary. Keep a little of it as context.
            picked.append(line)
        if len(picked) >= max_lines:
            break

    if not picked:
        return None
    if not saw_important and not any(_TAIL_IMPORTANT_RE.search(l) for l in picked):
        # Avoid making a diagnostic out of pure progress/build chatter.
        return None

    picked.reverse()
    excerpt = "\n".join(picked).strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[-max_chars:].lstrip()
    return excerpt or None


# Signals that are usually a downstream effect or too generic to be the root
# cause — they must never win `primary` over a specific root-cause signal
# (e.g. a precision miss or a real kernel trap). ERR99999 / acl-stream sync
# fire *after* the real crash; kernel_task_error/mlir defer to a specific
# cause; precision_mismatch_stats is the bare mere/mare line (precision_fail
# is the real one); segfault is often downstream of a device error.
_NOISE_KINDS = frozenset({
    "err_code_line",
    "acl_error_code",
    "acl_stream_error",
    "kernel_task_error",
    "mlir_compile_error",
    "precision_mismatch_stats",
    "segfault_abort",
})


def _prioritize_signals(signals: list[dict[str, Any]],
                        python_error: Optional[str] = None
                        ) -> list[dict[str, Any]]:
    """Root-cause signals first, downstream/generic noise last (stable within
    each group). So `primary` is the actionable cause — a precision miss or a
    real trap — not a downstream ERR99999 / acl-stream / bare mere-mare line.
    Root order stays PATTERNS order (compile > trap > precision), which is the
    right precedence when several co-occur."""
    if len(signals) < 2:
        return signals
    root = [s for s in signals if s.get("kind") not in _NOISE_KINDS]
    noise = [s for s in signals if s.get("kind") in _NOISE_KINDS]
    return root + noise


def extract_failure_signals(raw_output: str,
                            max_excerpt: int = 500) -> EvalDiagnostic:
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
        m = _last_match(regex, raw_output)
        if not m or kind in seen_kinds:
            continue
        seen_kinds.add(kind)
        data = extractor(m)
        excerpt = _line_excerpt(
            raw_output,
            m.start(),
            m.end(),
            max_chars=max_excerpt,
        )
        signals.append({"kind": kind, **data, "excerpt": excerpt, "hint": hint})

    exceptions = _PYTHON_EXCEPTION_RE.findall(raw_output)
    python_error = exceptions[-1][:240] if exceptions else None
    signals = _prioritize_signals(signals, python_error)
    tail_excerpt = _interesting_tail(raw_output)

    return EvalDiagnostic(
        primary=(signals[0]["kind"] if signals else None),
        signals=signals,
        python_error=python_error,
        tail_excerpt=tail_excerpt,
    )


def _signal_params(s: dict[str, Any]) -> str:
    """`k=v, k=v` for a signal's kind-specific fields (excl. kind/excerpt/hint)."""
    return ", ".join(f"{k}={v}" for k, v in s.items()
                     if k not in ("kind", "excerpt", "hint") and v is not None)


def summarize_one_line(sig: dict[str, Any]) -> str:
    """One-line distillation of failure_signals for a table cell / short note:
    the primary signal's ``kind [params]``, else the python_error, else ""."""
    signals = sig.get("signals") or []
    if signals:
        params = _signal_params(signals[0])
        return signals[0]["kind"] + (f" [{params}]" if params else "")
    return sig.get("python_error") or ""


def format_for_stdout(sig: dict[str, Any]) -> str:
    """Render a signals dict for pipeline.py's human-readable output.

    Empty string if nothing useful was extracted - callers can skip printing.
    """
    # Only the actionable distillation goes to stdout: kind + params + hint.
    # No log excerpts (per-signal or tail) — those repeat the Error line / each
    # other; the full untruncated log is in the FAIL report file.
    signals = sig.get("signals") or []
    python_error = sig.get("python_error")
    if not signals and not python_error:
        return ""
    lines = ["[PIPELINE] Eval failure signals:"]
    for s in signals:
        params = _signal_params(s)
        header = f"  - {s['kind']}"
        if params:
            header += f"  [{params}]"
        lines.append(header)
        if s.get("hint"):
            lines.append(f"      hint:    {s['hint']}")
    if python_error:
        lines.append(f"  - python_error: {python_error}")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    import sys

    data = sys.stdin.read()
    result = extract_failure_signals(data)
    json.dump(result.to_dict(), sys.stdout, indent=2)
    sys.stdout.write("\n")
