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
on ref.py (Model + get_inputs + get_init_inputs) and kernel.py (ModelNew).
Kernel.py also runs the triton-impl validator (`utils.validate_triton_impl`):
must use @triton.jit, forward() must launch it, no host-side for/while
loops, no forbidden torch.X / F.X / @ matmul.

Tier 2 (--full): LOCAL smoke test. Loads both modules on `torch.npu:0`
(or CPU), runs them, allclose via `utils.correctness`. NOT a proxy for
the batch eval path — `batch/run.py` defaults to a remote worker, and
a green --full says only "kernel runs locally".

Each op runs in its own subprocess. Results: <batch_dir>/verify_results.json.

Usage:
    python scripts/batch/verify.py <batch_dir>             # Tier 1
    python scripts/batch/verify.py <batch_dir> --full      # Tier 1 + Tier 2
    python scripts/batch/verify.py <batch_dir> --only op1,op2
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
from utils.correctness import compare_outputs_per_case  # noqa: E402
from utils.input_groups import resolve as _resolve_groups  # noqa: E402
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
    kernel files only) run the triton-impl regression validator."""
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

    # Step 4: regression validator (kernel-only). Refs legitimately use
    # torch native; the validator is about ensuring `ModelNew.forward`
    # actually drives a triton kernel rather than fall back to torch.
    if required is KERNEL_REQUIRED:
        from akg_agents.op.utils.code_checker import CodeChecker
        try:
            passed, error_msg, errors = CodeChecker(
                backend=target_backend(), dsl=target_dsl()
            ).check(src)
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


def _tier2_run(ref_path: Path, kernel_path: Path) -> dict:
    """Run ref + kernel, compare outputs via the shared correctness module
    (autoresearch's eval calls into the same `compare_outputs`)."""
    out: dict = {"status": "skip", "msg": "", "max_abs_diff": None}

    try:
        import torch  # type: ignore
    except ImportError as e:
        out["status"] = "ERROR"
        out["msg"] = f"torch import failed: {e}"
        return out
    try:
        import torch_npu  # type: ignore  # noqa: F401
    except Exception:
        pass  # not on Ascend; fine — kernel will pick its own device

    import importlib.util
    try:
        ref_spec = importlib.util.spec_from_file_location("_v_ref", str(ref_path))
        ref_mod = importlib.util.module_from_spec(ref_spec)  # type: ignore[arg-type]
        ref_spec.loader.exec_module(ref_mod)  # type: ignore[union-attr]
        kernel_spec = importlib.util.spec_from_file_location("_v_kernel", str(kernel_path))
        kernel_mod = importlib.util.module_from_spec(kernel_spec)  # type: ignore[arg-type]
        kernel_spec.loader.exec_module(kernel_mod)  # type: ignore[union-attr]
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"import: {type(e).__name__}: {e}"
        return out

    try:
        init_args = ref_mod.get_init_inputs()
        cases = _resolve_groups(ref_mod)
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"get_inputs/get_input_groups: {type(e).__name__}: {e}"
        return out

    # NPU when available; CPU otherwise. Kernels allocate output buffers
    # via torch.empty_like(x), so CPU input → CPU output buffer → garbage.
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is not None and getattr(npu_mod, "is_available", lambda: False)():
        device = torch.device("npu:0")
    else:
        device = torch.device("cpu")

    try:
        ref = ref_mod.Model(*init_args).to(device).eval()
        new = kernel_mod.ModelNew(*init_args).to(device).eval()
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"construct: {type(e).__name__}: {e}"
        return out

    def _to_list(x):
        if isinstance(x, (tuple, list)):
            return list(x)
        return [x]

    def _to_device(seq):
        return [t.to(device) if hasattr(t, "to") else t for t in seq]

    out_ref_per_case: list = []
    out_new_per_case: list = []
    try:
        with torch.no_grad():
            for case in cases:
                inp = _to_device(list(case))
                out_ref_per_case.append(_to_list(ref(*inp)))
                out_new_per_case.append(_to_list(new(*inp)))
    except Exception as e:
        out["status"] = "ERROR"
        out["msg"] = f"forward: {type(e).__name__}: {e}"
        return out

    cmp = compare_outputs_per_case(out_ref_per_case, out_new_per_case)
    out["max_abs_diff"] = cmp["max_abs_diff"]
    out["num_cases"] = len(cases)
    out["per_case"] = cmp["per_case"]
    if cmp["correctness"]:
        out["status"] = "PASS"
        out["msg"] = f"OK (n={len(cases)})"
    else:
        out["status"] = "FAIL"
        # Surface the first failing diagnostic — full list goes into JSON.
        bad = next((d for d in cmp["diagnostics"] if "OK" not in d), None)
        out["msg"] = bad or "correctness mismatch (no diagnostics)"
        out["diagnostics"] = cmp["diagnostics"]
    return out


def _run_tier_subprocess() -> int:
    """Subprocess entry point. Writes JSON to a sidecar path on stdout's last line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("1ref", "1kernel", "2"), required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--kernel", default="")
    ap.add_argument("--sidecar", required=True)
    args = ap.parse_args(sys.argv[2:])  # skip the --tier-runner sentinel

    ref_path = Path(args.ref)
    kernel_path = Path(args.kernel) if args.kernel else None

    if args.tier == "1ref":
        result = _tier1_inspect(ref_path, REF_REQUIRED)
    elif args.tier == "1kernel":
        result = _tier1_inspect(kernel_path, KERNEL_REQUIRED)
    else:  # tier == "2"
        result = _tier2_run(ref_path, kernel_path)

    Path(args.sidecar).write_text(json.dumps(result), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run_subprocess(*, tier: str, ref: Path, kernel: Path | None,
                    timeout: int) -> dict:
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


def _verify_one(case: dict, full: bool) -> dict:
    op = case["op_name"]
    ref = Path(case["ref"])
    kernel = Path(case["kernel"])

    out: dict = {"op_name": op, "tier1_ref": None, "tier1_kernel": None,
                 "tier2": None}

    out["tier1_ref"] = _run_subprocess(tier="1ref", ref=ref, kernel=None,
                                       timeout=TIER1_TIMEOUT)
    out["tier1_kernel"] = _run_subprocess(tier="1kernel", ref=ref,
                                          kernel=kernel,
                                          timeout=TIER1_TIMEOUT)

    def _t1_pass(rec: dict | None) -> bool:
        if not rec:
            return False
        # validate is "skip" for ref-side records (only kernel runs it).
        return (rec.get("exports") == "PASS"
                and rec.get("validate") != "FAIL")
    tier1_ok = _t1_pass(out["tier1_ref"]) and _t1_pass(out["tier1_kernel"])

    if full:
        if tier1_ok:
            out["tier2"] = _run_subprocess(tier="2", ref=ref, kernel=kernel,
                                           timeout=TIER2_TIMEOUT)
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

        if full and t2 is not None:
            col_t2 = t2.get("status", "?")
        else:
            col_t2 = "-"

        # Pick the most informative message
        msg = ""
        for src in (t2, t1k, t1r):
            if src and src.get("msg") and src.get("msg") != "OK":
                msg = src["msg"]
                if "FAIL" in (src.get("compile"), src.get("import"),
                              src.get("exports")) or src.get("status") in ("FAIL", "ERROR"):
                    break
        rows.append((op, col_t1r, col_t1k, col_t2,
                     _summary_status(rec, full), msg[:70]))

    op_w = max(8, max(len(r[0]) for r in rows))
    headers = ("op", "t1_ref", "t1_kern", "t2", "ok", "note")
    print(f"  {headers[0]:<{op_w}}  {headers[1]:<6}  {headers[2]:<7}  "
          f"{headers[3]:<6}  {headers[4]:<3}  {headers[5]}")
    print(f"  {'-' * op_w}  {'-' * 6}  {'-' * 7}  {'-' * 6}  {'-' * 3}  {'-' * 60}")
    for op, t1r, t1k, t2, ok, msg in rows:
        print(f"  {op:<{op_w}}  {t1r:<6}  {t1k:<7}  {t2:<6}  {ok:<3}  {msg}")


def run_verification(batch_dir: Path, *, full: bool = False,
                     only: str = "") -> int:
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
          f"precision: allclose-style (per-dtype rtol+atol)")
    print()

    results: dict = {}
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        op = case["op_name"]
        sys.stdout.write(f"  [{i:>3}/{len(cases)}] {op} ... ")
        sys.stdout.flush()
        rec = _verify_one(case, full=full)
        results[op] = rec
        ok = _summary_status(rec, full=full)
        sys.stdout.write(f"{ok}\n")
        sys.stdout.flush()

    out_path = batch_dir / VERIFY_RESULTS
    out_path.write_text(json.dumps({
        "full": full,
        "precision": "npu-benchmark-mere-mare",
        "results": results,
    }, indent=2), encoding="utf-8")

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
                    help="also run Tier 2 (execute ref + kernel, compare outputs); "
                         "needs the same hardware /autoresearch eval would use")
    ap.add_argument("--only", default="",
                    help="comma-separated op names")
    args = ap.parse_args()

    return run_verification(
        Path(args.batch_dir),
        full=args.full, only=args.only,
    )


if __name__ == "__main__":
    sys.exit(main())
