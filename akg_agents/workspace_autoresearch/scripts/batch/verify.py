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

"""Pre-flight verification for batch directories.

Tier 1 (default, no hardware): compile + import + required-symbol check
on ref.py (Model + get_inputs/get_input_groups + get_init_inputs) and the
kernel's importable wrapper (ModelNew). Kernel.py additionally runs the
DSL-aware static check via ``akg_agents.op.utils.code_checker.CodeChecker``
(syntax / py_compile / import / per-DSL anti-cheat / autotune); each DSL
contributes its own ``_<dsl>ComplianceCheck`` subclass - so the same
tier-1 path covers triton (@triton.jit + forward must launch it), pypto
(@pypto.jit), catlass (forward must call torch.ops.<ns>.*), etc.
For directory-backed AscendC, tier-1 also scans editable project files
(``.cpp/.h/.asc/CMakeLists.txt``) for CANN-Bench-style fallback/D2H
anti-cheat before Tier-2 touches any worker.

Tier 2 (--full): FORMAL verify-only pass. It materializes a temporary
task directory and calls ``utils.akg_eval.eval_kernel(...,
verify_only=True)``, so it reuses the same ``KernelVerifier`` +
``DSLAdapter`` chain as batch eval / worker correctness, minus profiling.

For multi-file DSLs (ascendc / ascendc_catlass), ``case["kernel"]`` is a directory
that gets passed to ``/autoresearch --kernel``, while
``case["kernel_module"]`` is the sibling ``kernel.py`` (or
``<op>_kernel.py``) that tier-1 imports and tier-2 verifies.

Each op runs in its own subprocess. Results: <batch_dir>/verify_results.json.

Usage:
    python scripts/batch/verify.py <batch_dir>             # Tier 1
    python scripts/batch/verify.py <batch_dir> --full      # Tier 1 + Tier 2
    python scripts/batch/verify.py <batch_dir> --full --worker-url 127.0.0.1:9111
    python scripts/batch/verify.py <batch_dir> --only op1,op2
    python scripts/batch/verify.py <batch_dir> --full --only op --case-ids 3,7
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest as mf
# Reach up one level for the shared precision module - single source of
# truth so verify.py and autoresearch's per-round eval can't drift.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.input_groups import num_cases_from_ref  # noqa: E402
from utils.hw_detect import derive_arch, probe_hint  # noqa: E402
from utils.settings import (  # noqa: E402
    batch_tier1_timeout, batch_tier2_timeout, target_backend, target_dsl,
)

VERIFY_RESULTS = "verify_results.json"
# Timeouts (seconds) from config.yaml `batch:`. tier2 cold JIT across ~50
# cases on triton-ascend can take minutes for kernels with many constexpr
# specializations (maxpool3d, ...); warm runs land in tens of seconds.
TIER1_TIMEOUT = batch_tier1_timeout()
TIER2_TIMEOUT = batch_tier2_timeout()

# Reference must export Model + get_init_inputs + one of (get_inputs,
# get_input_groups). The "input provider" is checked separately (per
# input_groups.resolve duck-type) since either symbol satisfies it.
REF_REQUIRED = ("Model", "get_init_inputs")
REF_INPUT_PROVIDERS = ("get_inputs", "get_input_groups")
KERNEL_REQUIRED = ("ModelNew",)


# ---------------------------------------------------------------------------
# Subprocess pool (this same file is re-invoked with --tier-runner)
# ---------------------------------------------------------------------------
def _tier1_inspect(path: Path, required: tuple[str, ...]) -> dict:
    """Compile, import, check required attrs are present, and (for
    kernel files only) run the DSL-aware CodeChecker (per-DSL anti-cheat
    + autotune)."""
    out: dict = {"path": str(path), "compile": "skip", "import": "skip",
                 "exports": "skip", "validate": "skip", "missing": [], "msg": ""}
    try:
        # utf-8-sig: PowerShell / Notepad on Windows tends to write source
        # files with a UTF-8 BOM; plain utf-8 leaves U+FEFF in the string and
        # compile() then dies with "invalid non-printable character U+FEFF".
        src = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        out["compile"] = "FAIL"
        out["msg"] = f"read error: {e}"
        return out
    try:
        compile(src, str(path), "exec")
        out["compile"] = "PASS"
    except SyntaxError as e:
        out["compile"] = "FAIL"
        out["msg"] = f"syntax error line {e.lineno}: {e.msg}"
        return out

    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location(
            f"_verify_{path.stem}", str(path)
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not build spec")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out["import"] = "PASS"
    except Exception as e:
        out["import"] = "FAIL"
        out["msg"] = f"{type(e).__name__}: {e}"
        return out

    missing = [name for name in required if not hasattr(mod, name)]
    # Reference also needs an input provider: either get_inputs() or
    # get_input_groups(). Treat absence of both as missing.
    if required is REF_REQUIRED and not any(
            hasattr(mod, n) for n in REF_INPUT_PROVIDERS):
        missing.append(" or ".join(REF_INPUT_PROVIDERS))
    if missing:
        out["exports"] = "FAIL"
        out["missing"] = missing
        out["msg"] = f"missing: {', '.join(missing)}"
        return out
    out["exports"] = "PASS"

    # Step 4: DSL-aware static check (kernel-only). Refs legitimately use
    # torch native; CodeChecker enforces per-DSL anti-cheat - e.g. triton:
    # ModelNew.forward must drive an @triton.jit kernel; catlass: forward
    # must call torch.ops.<ns>.*; non-Python-AST DSLs skip silently.
    if required is KERNEL_REQUIRED:
        from akg_agents.op.utils.code_checker import CodeChecker
        try:
            passed, error_msg, errors = CodeChecker(
                backend=target_backend(), dsl=target_dsl()
            ).check(src, task_info={"file": str(path)})
        except Exception as e:
            out["validate"] = "FAIL"
            out["msg"] = f"checker raised: {type(e).__name__}: {e}"[:160]
            return out
        if passed:
            out["validate"] = "PASS"
        else:
            out["validate"] = "FAIL"
            # Top error's detail + line if any, else first line of error_msg.
            if errors:
                first = errors[0]
                out["msg"] = (
                    f"L{first.get('line', 0)} "
                    f"{first.get('error_type', '?')}: "
                    f"{first.get('detail', '')}"
                )[:160]
            else:
                out["msg"] = (error_msg.splitlines()[0] if error_msg
                              else "regression detected")[:160]
    return out


def _tier1_static_check(path: Path) -> dict:
    """Run CodeChecker on a source file that is not necessarily Python.

    Directory-backed DSLs such as AscendC expose C++/AscendC/CMake files
    in addition to kernel.py. These files must be scanned for fallback
    compute and D2H egress before Tier-2 touches a worker.
    """
    out: dict = {"path": str(path), "compile": "skip", "import": "skip",
                 "exports": "PASS", "validate": "skip", "missing": [], "msg": ""}
    try:
        src = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as e:
        out["validate"] = "FAIL"
        out["msg"] = f"read error: {e}"
        return out

    from akg_agents.op.utils.code_checker import CodeChecker
    try:
        passed, error_msg, errors = CodeChecker(
            backend=target_backend(), dsl=target_dsl()
        ).check(src, task_info={"file": str(path)})
    except Exception as e:
        out["validate"] = "FAIL"
        out["msg"] = f"checker raised: {type(e).__name__}: {e}"[:160]
        return out

    if passed:
        out["validate"] = "PASS"
        return out

    out["validate"] = "FAIL"
    if errors:
        first = errors[0]
        out["msg"] = (
            f"L{first.get('line', 0)} "
            f"{first.get('error_type', '?')}: "
            f"{first.get('detail', '')}"
        )[:160]
    else:
        out["msg"] = (error_msg.splitlines()[0] if error_msg
                      else "regression detected")[:160]
    return out


def _parse_device_ids(devices: str | int | list[int] | None) -> list[int]:
    if devices is None:
        return []
    if isinstance(devices, list):
        return [int(x) for x in devices]
    text = str(devices).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_case_ids(case_ids: str | int | list[int] | None) -> list[int]:
    if case_ids is None:
        return []
    if isinstance(case_ids, list):
        return [int(x) for x in case_ids]
    text = str(case_ids).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _tier2_subprocess_timeout(ref_path: Path,
                              case_ids: list[int] | None = None) -> int:
    """Parent wall-clock cap for the Tier-2 runner. ``_tier2_run`` hands the
    PER-SHAPE ``TIER2_TIMEOUT`` to akg_eval, which expands it by num_cases
    before dispatch; the parent must mirror that expansion (same num_cases
    SSOT, ``input_groups.num_cases_from_ref``) or it kills a multi-shape
    verify mid-run and loses the per-case sidecar/logs. ``+ TIER1_TIMEOUT``
    covers the runner's own import/compile preamble."""
    num_cases = len(case_ids or []) or num_cases_from_ref(ref_path)
    return TIER2_TIMEOUT * num_cases + TIER1_TIMEOUT


def _tier2_run(ref_path: Path, kernel_path: Path, *,
               worker_url: str = "", device_ids: list[int] | None = None,
               case_ids: list[int] | None = None) -> dict:
    """Run the formal KernelVerifier-backed verify path on a temp task dir."""
    out: dict = {"status": "skip", "msg": "", "max_abs_diff": None}

    try:
        import shutil
        import tempfile
        from akg_agents.op.verifier.adapters.factory import get_dsl_adapter
        from scaffold import scaffold_task_dir
        from task_config.loader import load_task_config
        from utils.akg_eval import eval_kernel as formal_eval
        from utils.failure_extractor import summarize_one_line
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"formal verify import failed: {type(e).__name__}: {e}"
        return out

    def _op_name_from_ref(path: Path) -> str:
        stem = path.stem
        return stem[:-4] if stem.endswith("_ref") else stem

    def _ref_code(src_ref: Path, selected_ids: list[int]) -> str:
        src = src_ref.read_text(encoding="utf-8-sig")
        if not selected_ids:
            return src
        selected_literal = ", ".join(str(x) for x in selected_ids)
        patch = f"""

# --- verify.py case-id filter ---
# Applied only to the temporary reference.py used by Tier-2.  Generated
# CANN-Bench references build their multi-shape inputs from the global CASES
# table, so filtering that table is independent of any environment-variable
# convention inside the original reference file.  Prefer explicit case_id;
# fall back to the 1-based CASES ordinal for generic multi-shape refs.
_VERIFY_SELECTED_CASE_IDS = {{{selected_literal}}}
if _VERIFY_SELECTED_CASE_IDS:
    try:
        _VERIFY_FILTERED_CASES = []
        for _case_ordinal, _case in enumerate(CASES, 1):
            try:
                _case_id = int(_case.get("case_id") or _case_ordinal)
            except AttributeError:
                _case_id = _case_ordinal
            if _case_id in _VERIFY_SELECTED_CASE_IDS:
                _VERIFY_FILTERED_CASES.append(_case)
        CASES = _VERIFY_FILTERED_CASES
    except NameError:
        raise RuntimeError("verify.py --case-ids requires reference.py CASES")
    if not CASES:
        raise RuntimeError(
            f"verify.py --case-ids selected no cases: "
            f"{{sorted(_VERIFY_SELECTED_CASE_IDS)}}"
        )
# --- end verify.py case-id filter ---
"""
        return src.rstrip() + patch + "\n"

    op_name = _op_name_from_ref(ref_path)
    arch = None
    backend = target_backend()
    device_ids = list(device_ids or [])
    if not worker_url:
        if not device_ids:
            device_ids = [0]
        arch = derive_arch(device_ids[0], backend=backend)
    if not arch and not worker_url:
        out["status"] = "ERROR"
        out["msg"] = (
            f"could not derive local arch for backend={backend!r} "
            f"({probe_hint(backend)}); pass --worker-url for remote "
            "worker verify"
        )
        return out

    adapter = get_dsl_adapter(target_dsl())
    temp_root = Path(tempfile.mkdtemp(prefix=f"_batch_verify_{op_name}_"))
    try:
        entry_name = adapter.entry_filename_template.format(op_name=op_name)
        editable_files = [entry_name]
        project_src = None
        if adapter.kernel_arg_is_directory:
            project_name = adapter.kernel_project_dir_name
            if not project_name:
                raise RuntimeError(
                    f"{type(adapter).__name__} is directory-backed but has no "
                    "kernel_project_dir_name"
                )
            project_src = kernel_path.parent / project_name
            for rel in adapter.list_kernel_project_files(
                    str(project_src), op_name=op_name):
                if rel not in editable_files:
                    editable_files.append(rel)

        task_dir = Path(scaffold_task_dir(
            ref_code=_ref_code(ref_path, case_ids or []),
            kernel_code=kernel_path.read_text(encoding="utf-8-sig"),
            op_name=op_name,
            arch=arch or "",
            devices=device_ids,
            max_rounds=1,
            eval_timeout=TIER2_TIMEOUT,
            output_dir=str(temp_root),
            editable_filename=entry_name,
            editable_files=editable_files,
            kernel_project_src=(str(project_src) if project_src else None),
            ref_source_path=str(ref_path),
            worker_url=worker_url,
        ))

        config = load_task_config(str(task_dir))
        if config is None:
            raise RuntimeError("load_task_config returned None")
        raw = formal_eval(
            str(task_dir),
            config,
            device_id=device_ids or None,
            worker_url=worker_url or None,
            current_step=0,
            verify_only=True,
        )
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"formal verify setup failed: {type(e).__name__}: {e}"
        return out
    finally:
        if os.environ.get("AR_KEEP_BATCH_VERIFY_TEMP") == "1":
            out["temp_dir"] = str(temp_root)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

    metrics = raw.get("metrics") or {}
    out["max_abs_diff"] = metrics.get("max_abs_diff")
    out["num_cases"] = int(metrics.get("num_cases") or 1)
    per_case = list(raw.get("per_case") or [])
    if per_case:
        out["per_case"] = per_case

    if raw.get("outcome") == "ok":
        out["status"] = "PASS"
        out["msg"] = f"OK (n={out['num_cases']})"
        return out

    sig = raw.get("failure_signals") or {}
    failure_kinds = {
        str(item.get("failure_kind") or "")
        for item in per_case
        if isinstance(item, dict)
    }
    # FAIL = a numerical miss the agent can fix; ERROR = crash/infra. Trust
    # the per-case kernel_miss, else fall back to the distilled primary signal
    # (covers the per_case-less path where only failure_signals survived).
    is_fail = "kernel_miss" in failure_kinds or sig.get("primary") == "precision_fail"
    out["status"] = "FAIL" if is_fail else "ERROR"
    # Note reuses the shared failure_extractor distillation akg_eval already
    # produced (raw["failure_signals"]), not a hand-rolled error slice.
    out["msg"] = (summarize_one_line(sig)
                  or str(raw.get("error") or "formal verify failed"))[:160]
    out["raw_output_tail"] = str(raw.get("raw_output_tail") or "")[-4000:]
    return out


def _run_tier_subprocess() -> int:
    """Subprocess entry point. Writes JSON to a sidecar path on stdout's last line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("1ref", "1kernel", "1source", "2"), required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kernel", default="")
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--worker-url", default="")
    ap.add_argument("--device-ids", default="")
    ap.add_argument("--case-ids", default="")
    args = ap.parse_args(sys.argv[2:])  # skip the --tier-runner sentinel

    ref_path = Path(args.ref)
    kernel_path = Path(args.kernel) if args.kernel else None

    if args.tier == "1ref":
        result = _tier1_inspect(ref_path, REF_REQUIRED)
    elif args.tier == "1kernel":
        result = _tier1_inspect(kernel_path, KERNEL_REQUIRED)
    elif args.tier == "1source":
        result = _tier1_static_check(kernel_path)
    else:  # tier == "2"
        result = _tier2_run(
            ref_path,
            kernel_path,
            worker_url=args.worker_url,
            device_ids=_parse_device_ids(args.device_ids),
            case_ids=_parse_case_ids(args.case_ids),
        )

    Path(args.sidecar).write_text(json.dumps(result), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run_subprocess(*, tier: str, ref: Path, kernel: Path | None,
                    timeout: int, worker_url: str = "",
                    device_ids: list[int] | None = None,
                    case_ids: list[int] | None = None) -> dict:
    sidecar = Path(os.environ.get("TMP", "/tmp")) / f"_verify_{os.getpid()}_{tier}_{ref.stem}.json"
    if sidecar.exists():
        sidecar.unlink()
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--tier-runner",
           "--tier", tier,
           "--ref", str(ref),
           "--sidecar", str(sidecar)]
    if kernel is not None:
        cmd += ["--kernel", str(kernel)]
    if tier == "2":
        if worker_url:
            cmd += ["--worker-url", worker_url]
        ids = ",".join(str(x) for x in (device_ids or []))
        if ids:
            cmd += ["--device-ids", ids]
        selected = ",".join(str(x) for x in (case_ids or []))
        if selected:
            cmd += ["--case-ids", selected]

    env = os.environ.copy()
    # Default the Windows libomp/libiomp5md double-init workaround so users
    # don't see a wall of OMP error #15 on first run. No-op on Linux. Anyone
    # who wants the strict behavior can pre-set KMP_DUPLICATE_LIB_OK=FALSE.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"status": "ERROR", "msg": f"timeout after {timeout}s",
                "elapsed_s": round(time.time() - t0, 2)}

    elapsed = round(time.time() - t0, 2)
    if not sidecar.exists():
        return {"status": "ERROR",
                "msg": f"no result; rc={proc.returncode}",
                "stderr_tail": (proc.stderr or proc.stdout)[-400:],
                "elapsed_s": elapsed}
    try:
        result = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as e:
        return {"status": "ERROR", "msg": f"parse sidecar: {e}",
                "elapsed_s": elapsed}
    finally:
        try:
            sidecar.unlink()
        except OSError:
            pass
    result["elapsed_s"] = elapsed
    return result


def _verify_one(case: dict, full: bool, *, worker_url: str = "",
                device_ids: list[int] | None = None,
                case_ids: list[int] | None = None) -> dict:
    op = case["op_name"]
    ref = Path(case["ref"])
    # Tier-1 + tier-2 always import the Python wrapper. For single-file
    # DSLs kernel_module == kernel; for directory-backed DSLs kernel is
    # the project directory and kernel_module is the sibling kernel.py.
    kernel = Path(case.get("kernel_module") or case["kernel"])

    out: dict = {"op_name": op, "tier1_ref": None, "tier1_kernel": None,
                 "tier1_sources": [], "tier2": None}

    out["tier1_ref"] = _run_subprocess(tier="1ref", ref=ref, kernel=None,
                                       timeout=TIER1_TIMEOUT)
    out["tier1_kernel"] = _run_subprocess(tier="1kernel", ref=ref,
                                          kernel=kernel,
                                          timeout=TIER1_TIMEOUT)

    try:
        from akg_agents.op.verifier.adapters.factory import get_dsl_adapter
        adapter = get_dsl_adapter(target_dsl())
    except Exception:
        adapter = None

    project_path = Path(case["kernel"])
    if adapter is not None and getattr(adapter, "kernel_arg_is_directory", False):
        if project_path.is_dir():
            project_root = project_path
            wrapper_root = project_root.parent
            for rel in adapter.list_kernel_project_files(str(project_root), op_name=op):
                source_path = wrapper_root / rel
                if source_path == kernel:
                    continue
                out["tier1_sources"].append(
                    _run_subprocess(tier="1source", ref=ref,
                                    kernel=source_path,
                                    timeout=TIER1_TIMEOUT)
                )

    def _t1_pass(rec: dict | None) -> bool:
        if not rec:
            return False
        # validate is "skip" for ref-side records (only kernel runs it).
        return (rec.get("exports") == "PASS"
                and rec.get("validate") != "FAIL")
    tier1_ok = (
        _t1_pass(out["tier1_ref"])
        and _t1_pass(out["tier1_kernel"])
        and all(src.get("validate") != "FAIL" for src in out["tier1_sources"])
    )

    if full:
        if tier1_ok:
            out["tier2"] = _run_subprocess(tier="2", ref=ref, kernel=kernel,
                                           timeout=_tier2_subprocess_timeout(
                                               ref, case_ids),
                                           worker_url=worker_url,
                                           device_ids=device_ids,
                                           case_ids=case_ids)
        else:
            out["tier2"] = {"status": "skip",
                            "msg": "tier1 failed; skipping tier2",
                            "elapsed_s": 0}

    return out


_CONTENT_FAIL_FIELDS = ("compile", "import", "exports", "validate")


def _summary_status(record: dict, full: bool) -> str:
    """P/F/E/S single-letter. compile/import/exports/validate failures
    all map to F (matches the per-tier table column); runtime ERROR is E."""
    t1r = record["tier1_ref"]
    t1k = record["tier1_kernel"]
    t1s = record.get("tier1_sources") or []
    t2 = record["tier2"]

    def _bad(t):
        return t and ("FAIL" in (t.get("compile"), t.get("import"),
                                  t.get("exports"), t.get("validate"))
                      or t.get("status") in ("FAIL", "ERROR"))

    def _content_fail(t):
        return t and any(t.get(f) == "FAIL" for f in _CONTENT_FAIL_FIELDS)

    if _bad(t1r):
        return "F" if _content_fail(t1r) else "E"
    if _bad(t1k):
        return "F" if _content_fail(t1k) else "E"
    for src in t1s:
        if _bad(src):
            return "F" if _content_fail(src) else "E"
    if full and t2:
        if t2.get("status") == "PASS":
            return "P"
        if t2.get("status") == "FAIL":
            return "F"
        if t2.get("status") == "ERROR":
            return "E"
        return "S"
    return "P"


def _print_table(results: dict, full: bool) -> None:
    rows: list[tuple[str, str, str, str, str, str]] = []
    for op, rec in results.items():
        t1r = rec["tier1_ref"]
        t1k = rec["tier1_kernel"]
        t1s = rec.get("tier1_sources") or []
        t2 = rec["tier2"]

        col_t1r = "PASS" if t1r and t1r.get("exports") == "PASS" else (
            "FAIL" if t1r and t1r.get("exports") == "FAIL" else (
                "FAIL" if t1r and (t1r.get("compile") == "FAIL"
                                   or t1r.get("import") == "FAIL") else "ERROR"))

        if t1k is not None:
            if t1k.get("validate") == "FAIL":
                col_t1k = "FAIL"
            elif t1k.get("exports") == "PASS":
                col_t1k = "PASS"
            elif t1k.get("exports") == "FAIL":
                col_t1k = "FAIL"
            elif t1k.get("compile") == "FAIL" or t1k.get("import") == "FAIL":
                col_t1k = "FAIL"
            else:
                col_t1k = "ERROR"
        else:
            col_t1k = "-"

        if t1s:
            failed_sources = [src for src in t1s if src.get("validate") == "FAIL"]
            col_t1s = "FAIL" if failed_sources else "PASS"
        else:
            col_t1s = "-"

        if full and t2 is not None:
            col_t2 = t2.get("status", "?")
        else:
            col_t2 = "-"

        # Pick the most informative message
        msg = ""
        for src in (t2, *(t1s or []), t1k, t1r):
            if src and src.get("msg") and src.get("msg") != "OK":
                msg = src["msg"]
                if "FAIL" in (src.get("compile"), src.get("import"),
                              src.get("exports")) or src.get("status") in ("FAIL", "ERROR"):
                    break
        rows.append((op, col_t1r, col_t1k, col_t1s, col_t2,
                     _summary_status(rec, full), msg[:70]))

    op_w = max(8, max(len(r[0]) for r in rows))
    headers = ("op", "t1_ref", "t1_kern", "t1_src", "t2", "ok", "note")
    print(f"  {headers[0]:<{op_w}}  {headers[1]:<6}  {headers[2]:<7}  "
          f"{headers[3]:<6}  {headers[4]:<6}  {headers[5]:<3}  {headers[6]}")
    print(f"  {'-' * op_w}  {'-' * 6}  {'-' * 7}  {'-' * 6}  "
          f"{'-' * 6}  {'-' * 3}  {'-' * 60}")
    for op, t1r, t1k, t1s, t2, ok, msg in rows:
        print(f"  {op:<{op_w}}  {t1r:<6}  {t1k:<7}  {t1s:<6}  {t2:<6}  {ok:<3}  {msg}")


def run_verification(batch_dir: Path, *, full: bool = False,
                     only: str = "", worker_url: str = "",
                     devices: str | int = "",
                     case_ids: str | int | list[int] | None = None) -> int:
    """Run the verification loop programmatically (so prepare.py and other
    scripts can call us without subprocessing). Returns the same exit code
    main() would: 0 if everything passed, 1 if any FAIL/ERROR. All output
    still goes to stdout for the caller to surface."""
    batch_dir = Path(batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")

    try:
        manifest_path = mf.find_manifest(batch_dir)
        manifest_data = mf.load_manifest(manifest_path)
    except mf.ManifestError as e:
        sys.exit(str(e))

    try:
        cases = mf.resolve_cases(batch_dir, manifest_data, "ref-kernel")
    except mf.ManifestError as e:
        sys.exit(f"manifest validation failed: {e}")

    only_set = {s.strip() for s in (only or "").split(",") if s.strip()}
    if only_set:
        cases = [c for c in cases if c["op_name"] in only_set]
        if not cases:
            sys.exit("--only filtered out all ops")

    print(f"verify  batch_dir={batch_dir}  "
          f"tier={'1+2' if full else '1'}  ops={len(cases)}  "
          f"precision: KernelVerifier/CANN-Bench MERE/MARE for AscendC")
    device_ids = _parse_device_ids(devices)
    selected_case_ids = _parse_case_ids(case_ids)
    if full and worker_url:
        dev_desc = ",".join(str(x) for x in device_ids) if device_ids else "worker-declared"
        print(f"  worker_url={worker_url}  devices={dev_desc}")
    if selected_case_ids:
        print(f"  case_ids={','.join(str(x) for x in selected_case_ids)}")
    print()

    results: dict = {}
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        op = case["op_name"]
        sys.stdout.write(f"  [{i:>3}/{len(cases)}] {op} ... ")
        sys.stdout.flush()
        rec = _verify_one(
            case,
            full=full,
            worker_url=worker_url,
            device_ids=device_ids,
            case_ids=selected_case_ids,
        )
        results[op] = rec
        ok = _summary_status(rec, full=full)
        sys.stdout.write(f"{ok}\n")
        sys.stdout.flush()

    from utils.eval_summary import write_artifact
    out_path = write_artifact(batch_dir / VERIFY_RESULTS, {
        "full": full,
        "precision": "cannbench-mere-mare",
        "results": results,
    })

    print()
    _print_table(results, full=full)
    print()

    n_pass = sum(1 for op in results
                 if _summary_status(results[op], full) == "P")
    n_fail = sum(1 for op in results
                 if _summary_status(results[op], full) == "F")
    n_err = sum(1 for op in results
                if _summary_status(results[op], full) == "E")
    print(f"  total={len(results)}  pass={n_pass}  fail={n_fail}  "
          f"error={n_err}  elapsed={time.time()-t0:.1f}s")
    print(f"  results: {out_path}")
    return 0 if (n_fail == 0 and n_err == 0) else 1


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--tier-runner":
        return _run_tier_subprocess()

    ap = argparse.ArgumentParser(description="Pre-flight verify for batch directories.")
    ap.add_argument("batch_dir")
    ap.add_argument("--full", action="store_true",
                    help="also run Tier 2 (formal verify-only path via KernelVerifier); "
                         "needs the same hardware /autoresearch eval would use")
    ap.add_argument("--only", default="",
                    help="comma-separated op names")
    ap.add_argument("--worker-url", default="",
                    help="remote worker URL for --full Tier 2, e.g. 127.0.0.1:9111")
    ap.add_argument("--devices", default="",
                    help="optional device id/list filter for --full Tier 2; "
                         "omitted with --worker-url lets the worker declare/allocate")
    ap.add_argument("--case-ids", default="",
                    help="optional comma-separated CANN-Bench case ids for Tier 2")
    args = ap.parse_args()

    return run_verification(
        Path(args.batch_dir),
        full=args.full, only=args.only,
        worker_url=args.worker_url,
        devices=args.devices,
        case_ids=args.case_ids,
    )


if __name__ == "__main__":
    sys.exit(main())
