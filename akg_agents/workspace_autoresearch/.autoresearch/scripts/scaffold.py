#!/usr/bin/env python3
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

"""
Task directory scaffolder for Claude Code autoresearch.

Zero external dependency. Creates a self-contained task directory with:
  - task.yaml (config)
  - reference.py (correctness baseline; AST-checked via utils.ref_ast.
    validate_ref before scaffold copies it. Runtime correctness is
    validated by --run-baseline whose verify script tags error_source.)
  - kernel.py (editable seed; written from the user's --kernel file)
  - .ar_state/ (progress tracking)
  - .git/ (baseline commit)

Usage:
    # Local eval (direct subprocess, no HTTP). --devices is the device id.
    python .autoresearch/scripts/scaffold.py --ref reference.py --kernel kernel.py \\
        --op-name my_op --dsl triton_ascend --devices 0

    # Remote eval (ship package to a worker; localhost is fine).
    python .autoresearch/scripts/scaffold.py --ref reference.py --kernel kernel.py \\
        --op-name my_op --dsl triton_ascend --worker-url 127.0.0.1:9001

Output (last line of stdout):
    {"task_dir": "/absolute/path/to/task_dir", "status": "ok"}
"""


# pylint: disable=broad-exception-caught,import-outside-toplevel,missing-function-docstring
import argparse
import json
import os
import subprocess
import sys
import time
import uuid

import yaml


# ---------------------------------------------------------------------------
# Reference validation — delegated to the standalone library module so
# phase_machine.validators can call the same rule without importing this
# CLI script. The local re-export keeps callers that imported
# `scaffold.validate_ref` working.
# ---------------------------------------------------------------------------
from utils.ref_ast import validate_ref  # noqa: E402, F401  (re-export)


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

def scaffold_task_dir(
    *,
    ref_code: str,
    kernel_code: str,
    op_name: str,
    desc: str = "",
    dsl: str = "",
    framework: str = "torch",
    backend: str = "",
    arch: str = "",
    devices: list | None = None,
    worker_urls: list | None = None,
    max_rounds: int = 20,
    eval_timeout: int = 600,
    output_dir: str | None = None,
    editable_filename: str = "kernel.py",
    code_checker_enabled: bool = True,
) -> str:
    """Create task directory with all files. Returns absolute path."""
    # Determine base directory
    if output_dir:
        base_dir = output_dir
    else:
        base_dir = os.path.join(os.getcwd(), "ar_tasks")

    dir_name = f"{op_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    task_dir = os.path.join(base_dir, dir_name)
    os.makedirs(task_dir)

    # Write reference.py and the seed kernel.py from the user's files.
    _write(task_dir, "reference.py", ref_code)
    _write(task_dir, editable_filename, kernel_code)

    # Generate task.yaml
    task_yaml = {
        "name": op_name,
        "description": desc or f"Optimize {op_name}",
        "dsl": dsl or None,
        "framework": framework or None,
        "backend": backend or None,
        "arch": arch or None,
        "editable_files": [editable_filename],
        "eval": {
            "timeout": eval_timeout,
        },
        "metric": {
            "primary": "latency_us",
            "lower_is_better": True,
        },
        "agent": {
            "ref_file": "reference.py",
            "max_rounds": max_rounds,
        },
    }
    if devices:
        task_yaml["devices"] = list(devices)

    # Only emit the code_checker block when disabled — default-true tasks
    # stay clean. quick_check.py and phase_machine.validate_kernel honor
    # this field.
    if not code_checker_enabled:
        task_yaml["code_checker"] = {"enabled": False}

    # Add worker config if provided
    if worker_urls:
        task_yaml["worker"] = {"urls": worker_urls}

    yaml_content = yaml.dump(task_yaml, default_flow_style=False, allow_unicode=True)
    _write(task_dir, "task.yaml", yaml_content)

    # Create .ar_state directory
    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)

    # Git init + baseline commit
    _git_init(task_dir)

    return os.path.abspath(task_dir)


def _write(task_dir: str, rel_path: str, content: str):
    full_path = os.path.join(task_dir, rel_path)
    parent = os.path.dirname(full_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)


def _git_init(task_dir: str):
    """Initialize git repo and create baseline commit.

    The actual commit goes through git_utils.commit_in_task — same code
    path hooks use for round commits, so reliability is consistent.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.git_utils import commit_in_task

    subprocess.run(["git", "init"], cwd=task_dir, capture_output=True, check=True)
    ok, info = commit_in_task(task_dir, ["."], "scaffold: baseline")
    if not ok:
        raise RuntimeError(f"scaffold baseline commit failed: {info}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_arg_parser() -> argparse.ArgumentParser:
    """Construct scaffold's argparse, with no side effects.

    Extracted out of main() so parse_args.py can reuse the exact same flag
    spec without duplicating it. Single source of truth for which flags
    /autoresearch accepts and how they're typed/defaulted.
    """
    parser = argparse.ArgumentParser(
        description="Scaffold a task directory for Claude Code autoresearch",
    )
    parser.add_argument("--ref", required=True,
                        help="Path to reference.py (Model/get_inputs format)")
    parser.add_argument("--kernel", required=True,
                        help="Path to seed kernel file")
    parser.add_argument("--op-name", default=None,
                        help="Operator name (required)")
    # DSL = primary pivot. backend is a pure function of DSL; arch is
    # derived from hardware (local: npu-smi on --devices; remote: worker
    # /api/v1/status). Neither needs to be user-facing.
    # Pull the canonical DSL list from the adapter registry so the help
    # string can't drift from what `factory.get_dsl_adapter` actually accepts.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.hw_detect import list_supported_dsls
    parser.add_argument("--dsl", default=None,
                        help=f"DSL name (one of: {', '.join(list_supported_dsls())}). "
                             "Defaults to config.yaml:default_dsl.")
    parser.add_argument("--framework", default="torch",
                        choices=["torch", "mindspore", "numpy"],
                        help="Framework for the reference/kernel code "
                             "(default: torch).")
    parser.add_argument("--devices", default=None,
                        help="Comma-separated device IDs for local eval "
                             "(direct subprocess, no HTTP). Mutually "
                             "exclusive with --worker-url.")
    parser.add_argument("--worker-url", default=None,
                        help="Worker URL(s), comma-separated. Ships the "
                             "package to a worker; localhost is fine. "
                             "Mutually exclusive with --devices.")
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--eval-timeout", type=int, default=600)
    parser.add_argument("--output-dir", default=None,
                        help="Parent directory for the task (default: ./ar_tasks/)")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run baseline eval after scaffolding")
    parser.add_argument("--no-code-checker", action="store_true",
                        help=("Disable the static CodeChecker pipeline "
                              "(syntax / imports / DSL / autotune compliance) "
                              "for this task. Useful when the DSL rules are "
                              "too strict for the chosen kernel style. Writes "
                              "`code_checker: {enabled: false}` into "
                              "task.yaml; flip the field to re-enable later."))
    return parser


def _exit_error(payload: dict, rc: int = 1) -> None:
    """Print a one-line JSON error and exit. Every CLI failure goes
    through this so scaffold's stdout protocol stays single-line."""
    print(json.dumps(payload))
    sys.exit(rc)


def _validate_dsl_and_framework(args) -> None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.settings import default_dsl
    from utils.hw_detect import backend_for_dsl
    from akg_agents.op.verifier.adapters.factory import (
        get_dsl_adapter, get_framework_adapter,
    )
    args.dsl = (args.dsl or default_dsl()).lower()
    try:
        get_dsl_adapter(args.dsl)
    except Exception as e:
        _exit_error({"status": "error",
                     "error": f"unsupported --dsl {args.dsl!r}: {e}"})
    try:
        args.backend = backend_for_dsl(args.dsl)
    except ValueError as e:
        _exit_error({"status": "error", "error": str(e)})
    try:
        get_framework_adapter(args.framework)
    except Exception as e:
        _exit_error({"status": "error",
                     "error": ("unsupported --framework "
                               f"{args.framework!r}: {e}")})


def _resolve_devices(args) -> list:
    from utils.hw_detect import derive_arch
    devices_list = [int(d.strip()) for d in args.devices.split(",")
                    if d.strip()]
    if not devices_list:
        _exit_error({"status": "error",
                     "error": "--devices parsed to an empty list"})
    args.arch = derive_arch(args.backend, devices_list[0])
    if not args.arch:
        _exit_error({"status": "error",
                     "error": ("could not derive arch from "
                               f"{args.backend} device {devices_list[0]} "
                               "(is the SMI tool on PATH?)")})
    return devices_list


def _resolve_worker_urls(args) -> list:
    from utils.hw_detect import fetch_worker_hardware
    worker_urls = [u.strip() for u in args.worker_url.split(",")
                   if u.strip()]
    status = fetch_worker_hardware(worker_urls[0])
    if not status:
        _exit_error({"status": "error",
                     "error": (f"worker {worker_urls[0]} unreachable "
                               "or /api/v1/status failed")})
    worker_backend = str(status.get("backend", "")).lower()
    worker_arch = str(status.get("arch", "")).lower()
    if worker_backend and worker_backend != args.backend:
        _exit_error({"status": "error",
                     "error": (f"worker backend {worker_backend!r} "
                               f"incompatible with --dsl {args.dsl!r} "
                               f"(requires {args.backend!r})")})
    args.arch = worker_arch or None
    if not args.arch:
        _exit_error({"status": "error",
                     "error": ("worker /api/v1/status returned no "
                               f"arch: {status}")})
    return worker_urls


def _resolve_hardware(args) -> tuple:
    """Resolve --devices XOR --worker-url, populating args.arch and
    returning (devices_list, worker_urls)."""
    args.arch = None
    if args.devices and args.worker_url:
        _exit_error({"status": "error",
                     "error": ("--devices and --worker-url are mutually "
                               "exclusive. Pick one (--devices = local "
                               "subprocess; "
                               "--worker-url = HTTP worker).")})
    if args.devices:
        return _resolve_devices(args), []
    if args.worker_url:
        return [], _resolve_worker_urls(args)
    _exit_error({"status": "error",
                 "error": ("must pass exactly one of --devices "
                           "(local subprocess) or --worker-url "
                           "(HTTP worker).")})
    return [], []  # unreachable; _exit_error always raises SystemExit


def _read_seed_files(args) -> tuple:
    if not args.op_name:
        _exit_error({"status": "error",
                     "error": "--op-name is required"})
    if not os.path.isfile(args.ref):
        _exit_error({"status": "error",
                     "error": f"Reference file not found: {args.ref}"})
    with open(args.ref, "r", encoding="utf-8") as f:
        ref_code = f.read()
    try:
        validate_ref(ref_code, args.ref)
    except ValueError as e:
        _exit_error({"status": "error", "error": str(e)})
    if not os.path.isfile(args.kernel):
        _exit_error({"status": "error",
                     "error": f"Kernel file not found: {args.kernel}"})
    with open(args.kernel, "r", encoding="utf-8") as f:
        kernel_code = f.read()
    return ref_code, kernel_code


def _baseline_hint(err_source) -> str:
    if err_source == "ref":
        return ("The file passed via --ref is broken (import / forward / "
                "device-only bug). Fix the SOURCE file and re-run "
                "/autoresearch from scratch. The task directory is left "
                "for inspection but MUST NOT be activated — reference.py "
                "is not editable.")
    return ("INFRA_FAIL: no per-shape data — the seed kernel wasn't "
            "meaningfully exercised. Fix env (timeout / worker / device / "
            "OOM) and re-run `/autoresearch --resume <task_dir>`. Phase "
            "stays at BASELINE.")


def _run_baseline_step(task_dir: str, args) -> None:
    """Invoke baseline.py and translate its exit code into the scaffold
    JSON protocol. baseline exit codes are binary
    (workflow.baseline._EXIT_FOR): 0 = task activatable (OK or
    KERNEL_FAIL — hook routes to PLAN); 4 = NOT activatable (INFRA_FAIL —
    operator must intervene)."""
    print("[scaffold] Running baseline eval...", file=sys.stderr)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    baseline_cmd = [sys.executable,
                    os.path.join(script_dir, "engine", "baseline.py"),
                    task_dir]
    if args.worker_url:
        baseline_cmd.extend(["--worker-url", args.worker_url])
    rc = subprocess.run(baseline_cmd, check=False).returncode
    if rc == 0:
        return
    if rc == 4:
        try:
            with open(os.path.join(task_dir, ".ar_state",
                                   "progress.json"),
                      encoding="utf-8") as _f:
                err_source = json.load(_f).get("baseline_error_source")
        except (OSError, ValueError):
            err_source = None
        _exit_error({
            "status": "error",
            "task_dir": task_dir,
            "error": ("eval pipeline broken during baseline — see "
                      "[baseline]/[eval] stderr above"),
            "hint": _baseline_hint(err_source),
        }, rc=4)
    _exit_error({
        "status": "error",
        "task_dir": task_dir,
        "error": (f"baseline crashed unexpectedly (exit {rc}); "
                  "see [baseline]/[eval] stderr above"),
        "hint": ("This is not a classified outcome. Inspect the "
                 "baseline / eval stderr above and file a bug if the "
                 "exit code isn't in _EXIT_FOR."),
    }, rc=rc)


def _read_baseline_outcome(task_dir: str):
    try:
        with open(os.path.join(task_dir, ".ar_state", "progress.json"),
                  encoding="utf-8") as _f:
            return json.load(_f).get("baseline_outcome")
    except (OSError, ValueError):
        return None


def main():
    args = _make_arg_parser().parse_args()

    _validate_dsl_and_framework(args)
    devices_list, worker_urls = _resolve_hardware(args)
    ref_code, kernel_code = _read_seed_files(args)

    print(f"[scaffold] Creating task directory for {args.op_name}...",
          file=sys.stderr)
    task_dir = scaffold_task_dir(
        ref_code=ref_code,
        kernel_code=kernel_code,
        op_name=args.op_name,
        dsl=args.dsl,
        framework=args.framework,
        backend=args.backend,
        arch=args.arch,
        devices=devices_list,
        worker_urls=worker_urls,
        max_rounds=args.max_rounds,
        eval_timeout=args.eval_timeout,
        output_dir=args.output_dir,
        code_checker_enabled=not args.no_code_checker,
    )
    print(f"[scaffold] Task directory created: {task_dir}",
          file=sys.stderr)
    print("[scaffold] Files:", file=sys.stderr)
    for f in sorted(os.listdir(task_dir)):
        print(f"  {f}", file=sys.stderr)

    if args.run_baseline:
        _run_baseline_step(task_dir, args)

    # baseline_outcome distinguishes OK from KERNEL_FAIL without forcing
    # callers to re-read progress.json. Both are activatable (status=ok,
    # rc=0); the difference is whether the seed kernel produced valid
    # timings or PLAN has to rewrite it.
    print(json.dumps({"task_dir": task_dir, "status": "ok",
                      "baseline_outcome": _read_baseline_outcome(task_dir)}))


if __name__ == "__main__":
    main()
