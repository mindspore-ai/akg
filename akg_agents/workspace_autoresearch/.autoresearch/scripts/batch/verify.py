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

"""Static pre-flight verification for batch directories.

Compile + import + required-symbol check on ref.py (Model +
get_init_inputs + one of get_inputs / get_input_groups) and kernel.py
(ModelNew). Each op runs in its own subprocess so a missing dependency
in one op doesn't poison the others.

Full numerical correctness goes through akg's KernelVerifier on a
worker — see ``utils.akg_eval.eval_kernel`` — this pre-flight only
catches syntax / import / missing-symbol breakage.

Results are written to <batch_dir>/verify_results.json.

Usage:
    python .autoresearch/scripts/batch/verify.py <batch_dir>
    python .autoresearch/scripts/batch/verify.py <batch_dir> --only op1,op2
"""

# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring,wrong-import-position
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VERIFY_RESULTS = "verify_results.json"
INSPECT_TIMEOUT = 30

# Reference must export Model + get_init_inputs + one of (get_inputs,
# get_input_groups). The "input provider" is checked separately (per
# input_groups.resolve duck-type) since either symbol satisfies it.
REF_REQUIRED = ("Model", "get_init_inputs")
REF_INPUT_PROVIDERS = ("get_inputs", "get_input_groups")
KERNEL_REQUIRED = ("ModelNew",)


# ---------------------------------------------------------------------------
# Subprocess worker (this same file is re-invoked with --inspect-worker)
# ---------------------------------------------------------------------------
def _inspect(path: Path, required: tuple[str, ...]) -> dict:
    """Compile, import, check required attrs are present."""
    out: dict = {"path": str(path), "compile": "skip", "import": "skip",
                 "exports": "skip", "missing": [], "msg": ""}
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
    if required is REF_REQUIRED and not any(
            hasattr(mod, n) for n in REF_INPUT_PROVIDERS):
        missing.append(" or ".join(REF_INPUT_PROVIDERS))
    if missing:
        out["exports"] = "FAIL"
        out["missing"] = missing
        out["msg"] = f"missing: {', '.join(missing)}"
    else:
        out["exports"] = "PASS"
    return out


def _worker_main() -> int:
    """Subprocess entry point. Writes JSON to a sidecar path."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=("ref", "kernel"), required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--sidecar", required=True)
    args = ap.parse_args(sys.argv[2:])  # skip the --inspect-worker sentinel

    required = REF_REQUIRED if args.target == "ref" else KERNEL_REQUIRED
    result = _inspect(Path(args.path), required)
    Path(args.sidecar).write_text(json.dumps(result), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _run_subprocess(*, target: str, path: Path, timeout: int) -> dict:
    sidecar = (Path(os.environ.get("TMP", "/tmp"))
               / f"_verify_{os.getpid()}_{target}_{path.stem}.json")
    if sidecar.exists():
        sidecar.unlink()
    cmd = [sys.executable, str(Path(__file__).resolve()),
           "--inspect-worker",
           "--target", target,
           "--path", str(path),
           "--sidecar", str(sidecar)]

    env = os.environ.copy()
    # Default the Windows libomp/libiomp5md double-init workaround so users
    # don't see a wall of OMP error #15 on first run. No-op on Linux. Anyone
    # who wants the strict behavior can pre-set KMP_DUPLICATE_LIB_OK=FALSE.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env, check=False)
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


def _verify_one(case: dict) -> dict:
    op = case["op_name"]
    ref = Path(case["ref"])
    kernel = Path(case["kernel"])
    return {
        "op_name": op,
        "ref": _run_subprocess(target="ref", path=ref,
                               timeout=INSPECT_TIMEOUT),
        "kernel": _run_subprocess(target="kernel", path=kernel,
                                  timeout=INSPECT_TIMEOUT),
    }


_CONTENT_FAIL_FIELDS = ("compile", "import", "exports")


def _summary_status(record: dict) -> str:
    """P/F/E single-letter. Compile/import/exports failures all map to F
    (matches the per-target table column); runtime ERROR is E."""
    for key in ("ref", "kernel"):
        t = record[key]
        bad = t and ("FAIL" in (t.get("compile"), t.get("import"),
                                t.get("exports"))
                     or t.get("status") in ("FAIL", "ERROR"))
        if not bad:
            continue
        content_fail = any(t.get(f) == "FAIL"
                           for f in _CONTENT_FAIL_FIELDS)
        return "F" if content_fail else "E"
    return "P"


def _status_label(t: dict | None) -> str:
    """Single PASS/FAIL/ERROR/- label for one target record."""
    if t is None:
        return "-"
    if t.get("exports") == "PASS":
        return "PASS"
    if t.get("exports") == "FAIL":
        return "FAIL"
    if t.get("compile") == "FAIL" or t.get("import") == "FAIL":
        return "FAIL"
    return "ERROR"


def _pick_message(kernel: dict | None, ref: dict | None) -> str:
    """Most informative non-OK msg, scanning kernel→ref. Breaks on the
    first record that actually has a FAIL/ERROR status."""
    msg = ""
    for src in (kernel, ref):
        if not (src and src.get("msg") and src.get("msg") != "OK"):
            continue
        msg = src["msg"]
        if "FAIL" in (src.get("compile"), src.get("import"),
                      src.get("exports")) or src.get("status") in ("FAIL",
                                                                   "ERROR"):
            break
    return msg


def _print_table(results: dict) -> None:
    rows: list[tuple[str, str, str, str, str]] = []
    for op, rec in results.items():
        rows.append((
            op,
            _status_label(rec["ref"]),
            _status_label(rec["kernel"]),
            _summary_status(rec),
            _pick_message(rec["kernel"], rec["ref"])[:70],
        ))

    op_w = max(8, max(len(r[0]) for r in rows))
    headers = ("op", "ref", "kernel", "ok", "note")
    print(f"  {headers[0]:<{op_w}}  {headers[1]:<6}  {headers[2]:<7}  "
          f"{headers[3]:<3}  {headers[4]}")
    print(f"  {'-' * op_w}  {'-' * 6}  {'-' * 7}  {'-' * 3}  {'-' * 60}")
    for op, ref, kernel, ok, msg in rows:
        print(f"  {op:<{op_w}}  {ref:<6}  {kernel:<7}  {ok:<3}  {msg}")


def _load_cases(batch_dir: Path, only: str) -> list[dict]:
    """Load + resolve + optional --only filter. Exits on any manifest
    error (run_verification doesn't recover from those)."""
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
    return cases


def _verify_all(cases: list[dict]) -> dict:
    results: dict = {}
    for i, case in enumerate(cases, 1):
        op = case["op_name"]
        sys.stdout.write(f"  [{i:>3}/{len(cases)}] {op} ... ")
        sys.stdout.flush()
        rec = _verify_one(case)
        results[op] = rec
        sys.stdout.write(f"{_summary_status(rec)}\n")
        sys.stdout.flush()
    return results


def _report_results(batch_dir: Path, results: dict, elapsed: float) -> int:
    out_path = batch_dir / VERIFY_RESULTS
    out_path.write_text(json.dumps({"results": results}, indent=2),
                        encoding="utf-8")
    print()
    _print_table(results)
    print()
    n_pass = sum(1 for op in results if _summary_status(results[op]) == "P")
    n_fail = sum(1 for op in results if _summary_status(results[op]) == "F")
    n_err = sum(1 for op in results if _summary_status(results[op]) == "E")
    print(f"  total={len(results)}  pass={n_pass}  fail={n_fail}  "
          f"error={n_err}  elapsed={elapsed:.1f}s")
    print(f"  results: {out_path}")
    return 0 if (n_fail == 0 and n_err == 0) else 1


def run_verification(batch_dir: Path, *, only: str = "") -> int:
    """Run the verification loop programmatically (so prepare.py and other
    scripts can call us without subprocessing). Returns 0 on full pass,
    1 if any FAIL/ERROR."""
    batch_dir = Path(batch_dir).resolve()
    if not batch_dir.is_dir():
        sys.exit(f"batch dir not found: {batch_dir}")
    cases = _load_cases(batch_dir, only)
    print(f"verify  batch_dir={batch_dir}  ops={len(cases)}")
    print()
    t0 = time.time()
    results = _verify_all(cases)
    return _report_results(batch_dir, results, time.time() - t0)


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--inspect-worker":
        return _worker_main()

    ap = argparse.ArgumentParser(
        description="Pre-flight static verify for batch directories.",
    )
    ap.add_argument("batch_dir")
    ap.add_argument("--only", default="",
                    help="comma-separated op names")
    args = ap.parse_args()

    return run_verification(Path(args.batch_dir), only=args.only)


if __name__ == "__main__":
    sys.exit(main())
