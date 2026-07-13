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
    validated by --run-baseline whose verify routine tags error_source.)
  - kernel.py (editable seed; from --kernel file, or sibling kernel.py when
    --kernel is a multi-file DSL project directory)
  - <dsl project>/ (multi-file DSLs only, when --kernel points at that folder)
  - .ar_state/ (progress tracking)
  - .git/ (baseline commit)

Usage:
    # NOTE: --devices values below are placeholders; pass the actual free
    # device id at invocation time.

    # Local eval (arch auto-derived from device — npu-smi for ascend,
    # nvidia-smi for cuda; dispatch via config.yaml defaults.backend):
    python scripts/scaffold.py --ref reference.py --kernel kernel.py --op-name my_op --devices <DEV>

    # Custom output directory:
    python scripts/scaffold.py --ref reference.py --kernel kernel.py --op-name my_op --devices <DEV> --output-dir /tmp/tasks

Output (last line of stdout):
    {"task_dir": "/absolute/path/to/task_dir", "status": "ok"}
"""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Reference validation — delegated to the standalone library module so
# phase_machine.validators can call the same rule without importing this
# CLI script. The local re-export keeps callers that imported
# `scaffold.validate_ref` working.
# ---------------------------------------------------------------------------
from utils.ref_ast import validate_ref  # noqa: E402, F401  (re-export)
from utils.git_utils import commit_in_task  # noqa: E402
from utils.hw_detect import derive_arch, probe_hint  # noqa: E402
from utils.settings import (  # noqa: E402
    default_max_rounds, default_eval_timeout, default_metric,
    default_code_checker_enabled, target_backend, target_dsl,
)
from akg_agents.op.utils.task_layout import REF_FILE_DEFAULT  # noqa: E402
from phase_machine import task_summary  # noqa: E402
from task_handle import open_task, Role  # noqa: E402


# ---------------------------------------------------------------------------
# DSL-aware scaffold dispatch: every per-DSL knob (does --kernel take a
# directory, what files beyond kernel.py are editable, what extra source
# tree gets copied into task_dir) is owned by the DSL adapter. Scaffold
# stays DSL-name-agnostic.
# ---------------------------------------------------------------------------

def _scaffold_dsl_adapter():
    from akg_agents.op.verifier.adapters.factory import get_dsl_adapter
    return get_dsl_adapter(target_dsl())


def _run_initial_baseline(task_dir: str) -> int:
    """Activate a newly scaffolded task, then run its baseline.

    PostToolUse cannot own this transition: ``--run-baseline`` executes
    baseline.py inside the scaffold command, before the hook can observe the
    new task.  Keep activation next to that synchronous call so every caller
    (Claude, OpenCode, batch, or a plain CLI) sees the same INIT -> BASELINE
    ordering.
    """
    with open_task(task_dir, role=Role.SUPERVISOR) as task:
        task.activate(fresh=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return subprocess.run([
        sys.executable,
        os.path.join(script_dir, "engine", "baseline.py"),
        task_dir,
    ]).returncode


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------

def scaffold_task_dir(
    *,
    ref_code: str,
    kernel_code: str,
    op_name: str,
    desc: str = "",
    arch: str = "",
    devices: list | None = None,
    max_rounds: int | None = None,
    eval_timeout: int | None = None,
    output_dir: str | None = None,
    editable_filename: str = "kernel.py",
    editable_files: list | None = None,
    kernel_project_src: str | None = None,
    code_checker_enabled: bool | None = None,
    ref_source_path: str | None = None,
    worker_url: str = "",
) -> str:
    """Create task directory with all files. Returns absolute path."""
    if max_rounds is None:
        max_rounds = default_max_rounds()
    if eval_timeout is None:
        eval_timeout = default_eval_timeout()
    if code_checker_enabled is None:
        code_checker_enabled = default_code_checker_enabled()
    # Determine base directory
    if output_dir:
        base_dir = output_dir
    else:
        base_dir = os.path.join(os.getcwd(), "ar_tasks")

    dir_name = f"{op_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    task_dir = os.path.join(base_dir, dir_name)
    os.makedirs(task_dir)

    # Write reference.py and the seed kernel.py from the user's files.
    _write(task_dir, REF_FILE_DEFAULT, ref_code)
    _write(task_dir, editable_filename, kernel_code)
    # Per-DSL hook: copy extra source trees (e.g. catlass_op/) +
    # any one-shot patches the adapter wants done at scaffold time.
    _scaffold_dsl_adapter().materialize_project_tree(task_dir, kernel_project_src)

    # NPUKernelBench-style refs read shape lists from a sibling JSON via
    # `os.path.join(os.path.dirname(__file__), "<basename>.json")`;
    # sglang-style refs torch.load() a sibling `.pt` cache the same way.
    # Copy these next to ref.py inside task_dir (preserving names so
    # `dirname(__file__)` resolution still hits them) and remember the
    # basenames — they're written into task.yaml `data_files:` below so
    # the remote package builder ships them too. Without this, local
    # eval works (the file is on disk next to ref.py) but remote eval
    # FileNotFoundErrors on the worker.
    discovered_data_files: list[str] = []
    if ref_source_path:
        try:
            import shutil as _shutil
            ref_dir_src = os.path.dirname(os.path.abspath(ref_source_path))
            ref_stem = os.path.splitext(os.path.basename(ref_source_path))[0]
            # Match only the current ref's siblings — NPUKernelBench's
            # level0/ has dozens of ops, each with its own `<name>.json`
            # + `<name>_all_case.json`. Without this filter, scaffold
            # would copy every neighbouring op's case file.
            for fname in sorted(os.listdir(ref_dir_src)):
                if fname.startswith("."):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext not in (".json", ".pt", ".npz"):
                    continue
                stem, ext = os.path.splitext(fname)
                if stem != ref_stem and not stem.startswith(ref_stem + "_"):
                    continue
                src = os.path.join(ref_dir_src, fname)
                if not os.path.isfile(src):
                    continue
                # If the sidecar's stem EXACTLY matches the original ref
                # stem, the ref likely derives its JSON path from `__file__`
                # (the convention-compliant case). Since scaffold renames
                # the ref to REF_FILE_DEFAULT (`reference.py`), the
                # `os.path.dirname(__file__) + basename(stem) + '.json'`
                # path resolves to `<REF_STEM>.json` in task_dir at runtime
                # — so we must copy the sidecar under that renamed name
                # too, else baseline hits FileNotFoundError. Other refs
                # (those that hardcode their sidecar filename) keep the
                # original filename.
                if stem == ref_stem:
                    dest_name = os.path.splitext(REF_FILE_DEFAULT)[0] + ext
                else:
                    dest_name = fname
                _shutil.copy(src, os.path.join(task_dir, dest_name))
                discovered_data_files.append(dest_name)
        except Exception as _e:
            print(f"[scaffold] WARNING: sidecar data file copy failed: {_e}")

    # Generate task.yaml — only fields that vary per-task. dsl /
    # framework / backend are pinned per repo in config.yaml's
    # ``defaults`` block (utils.settings.target_*); not written here.
    # Probe the ref's case count once, here at scaffold time (cwd has the
    # ref + data_files already written above). Pin it into task.yaml
    # `eval.num_cases` so later rounds — including a first remote baseline
    # on a dev host that can't import the ref — scale the eval timeout and
    # sticky fingerprint correctly instead of falling back to 1. Probe
    # failure (no torch/CANN here) just omits the field; eval client then
    # falls back to its own probe / fingerprint reuse as before.
    eval_block = {"timeout": eval_timeout}
    num_cases = _probe_num_cases(task_dir, REF_FILE_DEFAULT)
    if num_cases and num_cases >= 1:
        eval_block["num_cases"] = num_cases

    task_yaml = {
        "name": op_name,
        "description": desc or f"Optimize {op_name}",
        "arch": arch or None,
        "editable_files": editable_files or [editable_filename],
        "eval": eval_block,
        "metric": {
            "primary": default_metric()["primary"],
            "lower_is_better": default_metric()["lower_is_better"],
            # Pin the threshold too: loader falls back to the live global
            # config when it's absent, so an unpinned task would silently
            # change KEEP/DISCARD behaviour if the global default is retuned.
            "improvement_threshold": default_metric()["improvement_threshold"],
        },
        "agent": {
            "ref_file": REF_FILE_DEFAULT,
            "max_rounds": max_rounds,
        },
    }
    if devices:
        task_yaml["devices"] = list(devices)
    if worker_url:
        worker_urls = [u.strip() for u in worker_url.split(",") if u.strip()]
        task_yaml["worker"] = {"urls": worker_urls}
    if discovered_data_files:
        task_yaml["data_files"] = discovered_data_files

    # Always pin the task-level value (code_checker_enabled is already
    # resolved to a concrete bool above). loader falls back to the live
    # global config when this field is absent, so an unpinned task would
    # silently flip behaviour if the global default is retuned — same
    # pinning rationale as eval.timeout / metric.improvement_threshold.
    # quick_check.py honors this field.
    task_yaml["code_checker"] = {"enabled": bool(code_checker_enabled)}

    yaml_content = yaml.dump(task_yaml, default_flow_style=False, allow_unicode=True)
    _write(task_dir, "task.yaml", yaml_content)

    # Create .ar_state directory
    os.makedirs(os.path.join(task_dir, ".ar_state"), exist_ok=True)

    # Git init + baseline commit
    _git_init(task_dir)

    return os.path.abspath(task_dir)


def _probe_num_cases(task_dir: str, ref_file: str):
    """Best-effort case count for task.yaml ``eval.num_cases``. Loads the
    just-written reference module. Generated multi-shape refs expose a
    literal ``CASES`` table, so count that first instead of calling
    ``get_input_groups()`` and constructing large tensors at scaffold time.
    For generic refs, delegate to ``utils.input_groups.num_cases`` so
    single / dyn_list / input_groups refs still resolve consistently.
    Returns None when the ref can't be imported here (e.g. no torch on the
    dev host); caller omits the field and eval_timeout scaling falls back
    to a runtime re-probe."""
    import importlib.util
    ref_path = os.path.join(task_dir, ref_file)
    if not os.path.isfile(ref_path):
        return None
    try:
        old_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            spec = importlib.util.spec_from_file_location("_ref_probe", ref_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = old_dont_write_bytecode
        cases = getattr(mod, "CASES", None)
        if isinstance(cases, (list, tuple)):
            return len(cases)
        from utils.input_groups import num_cases
        return num_cases(mod)
    except Exception:
        return None


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
                        help="Seed kernel .py file, or multi-file DSL project "
                             "directory with sibling kernel.py")
    parser.add_argument("--op-name", default=None,
                        help="Operator name (required)")
    # backend / framework / dsl are pinned per repo in config.yaml's
    # ``defaults`` block. arch is derived from the picked --devices via
    # the backend-appropriate probe (npu-smi / nvidia-smi).
    parser.add_argument("--devices", default=None,
                        help="Comma-separated device IDs for local eval "
                             "(e.g. '5' or '0,1,2,3'). Required.")
    parser.add_argument("--max-rounds", type=int, default=default_max_rounds())
    parser.add_argument("--eval-timeout", type=int, default=default_eval_timeout())
    parser.add_argument("--output-dir", default=None,
                        help="Parent directory for the task (default: ./ar_tasks/)")
    parser.add_argument("--run-baseline", action="store_true",
                        help="Also run baseline eval after scaffolding")
    # Single flag, store_const so the absence of --no-code-checker yields
    # None (lets defaults.code_checker_enabled in config.yaml decide) and
    # presence yields False (pinned into task.yaml as enabled: false).
    parser.add_argument("--no-code-checker", dest="code_checker",
                        action="store_const", const=False, default=None,
                        help=("Disable the static Triton regression check "
                              "(validate_triton_impl) for this task. "
                              "Useful when the regression rules are too "
                              "strict for the chosen kernel style. Writes "
                              "`code_checker: {enabled: false}` into "
                              "task.yaml; flip the field to re-enable later."))
    parser.add_argument("--worker-url", default="",
                        help="Remote worker URL(s) (host:port, comma-separated). "
                             "Routes eval through the remote HTTP worker "
                             "instead of probing a local device.")
    return parser


def main():
    parser = _make_arg_parser()
    args = parser.parse_args()

    # Hardware resolution: --devices is required unless --worker-url routes
    # eval to a remote machine. dsl / framework / backend are pinned by
    # this workspace's config.yaml (single-target-per-repo); arch varies
    # per machine and is auto-probed from the picked --devices via the
    # backend-appropriate tool (npu-smi for ascend, nvidia-smi for cuda).
    backend = target_backend()
    devices_list: list = []
    args.arch = None

    has_remote = bool(args.worker_url and args.worker_url.strip())

    if not args.devices and not has_remote:
        print(json.dumps({"status": "error",
                          "error": "--devices (local eval) or "
                                   "--worker-url (remote worker) is required."}))
        sys.exit(1)

    if args.devices:
        devices_list = [int(d.strip()) for d in args.devices.split(",")
                        if d.strip()]

    if not has_remote:
        args.arch = derive_arch(devices_list[0], backend=backend)
        if not args.arch:
            print(json.dumps({"status": "error",
                              "error": (f"could not derive arch from "
                                        f"device {devices_list[0]} for "
                                        f"backend={backend!r} "
                                        f"({probe_hint(backend)})")}))
            sys.exit(1)

    if not args.op_name:
        print(json.dumps({"status": "error",
                          "error": "--op-name is required"}))
        sys.exit(1)

    if not os.path.isfile(args.ref):
        print(json.dumps({"status": "error",
                          "error": f"Reference file not found: {args.ref}"}))
        sys.exit(1)
    with open(args.ref, "r", encoding="utf-8") as f:
        ref_code = f.read()
    try:
        validate_ref(ref_code, args.ref)
    except ValueError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)

    # Ask the configured DSL's adapter how to interpret --kernel. The
    # default reads a .py file; catlass overrides to accept a directory
    # and resolve the sibling kernel.py. Errors raised here render as
    # JSON-on-stdout for parse_args to forward.
    kernel_path = os.path.abspath(args.kernel)
    if not os.path.exists(kernel_path):
        print(json.dumps({"status": "error",
                          "error": f"--kernel path not found: {args.kernel}"}))
        sys.exit(1)
    adapter = _scaffold_dsl_adapter()
    try:
        kernel_code, kernel_project_src = adapter.read_kernel_source(
            kernel_path, op_name=args.op_name)
    except FileNotFoundError as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        sys.exit(1)
    entry_filename = adapter.entry_filename_template.format(op_name=args.op_name)
    editable_files = [entry_filename] + list(
        adapter.list_kernel_project_files(
            kernel_project_src, op_name=args.op_name)
    )

    # devices_list was resolved above.
    print(f"[scaffold] Creating task directory for {args.op_name}...")

    task_dir = scaffold_task_dir(
        ref_code=ref_code,
        kernel_code=kernel_code,
        op_name=args.op_name,
        devices=devices_list,
        arch=args.arch,
        max_rounds=args.max_rounds,
        eval_timeout=args.eval_timeout,
        output_dir=args.output_dir,
        code_checker_enabled=args.code_checker,  # None -> config default
        ref_source_path=args.ref,
        worker_url=args.worker_url,
        kernel_project_src=kernel_project_src,
        editable_filename=entry_filename,
        editable_files=editable_files,
    )

    print(f"[scaffold] Task directory created: {task_dir}")
    print("[scaffold] Files:")
    for f in sorted(os.listdir(task_dir)):
        print(f"  {f}")

    # Bind directly into this batch's manifest. A repo-global per-op pointer
    # races when two batches optimize the same op, so batch identity stays in
    # AR_BATCH_DIR/AR_BATCH_OP instead.
    batch_dir = os.environ.get("AR_BATCH_DIR")
    batch_op = os.environ.get("AR_BATCH_OP")
    if batch_dir and batch_op == args.op_name:
        try:
            sys.path.insert(
                0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch"))
            import manifest as batch_manifest  # noqa: E402
            batch_manifest.update_case(
                Path(batch_dir), args.op_name,
                task_dir=os.path.abspath(task_dir))
        except Exception as e:
            print(f"[scaffold] warning: failed to update batch task_dir: {e}")

    # Reference validation is now a single path through baseline.py: the
    # generated verify routine splits ref-side and kernel-side try/excepts
    # and tags error_source on failure. Scaffold reads the resulting
    # baseline exit code and decides:
    #   - exit 4 (INFRA_FAIL) → reject task (operator must fix --ref or env)
    #   - any other non-zero → kernel-side failure, task activates and
    #     hook routes to PLAN
    # AST symbol presence was already checked earlier (validate_ref on
    # the source --ref file before copying), so import errors / missing
    # symbols never reach this point.
    if args.run_baseline:
        print("[scaffold] Running baseline eval...")
        rc = _run_initial_baseline(task_dir)
        # baseline exit codes are binary now (workflow.baseline._EXIT_FOR):
        #   0 = task activatable (OK or KERNEL_FAIL — hook routes to PLAN)
        #   4 = task NOT activatable (INFRA_FAIL — operator must intervene)
        # Anything else here is an unexpected baseline crash.
        if rc == 4:
            # baseline_error_source lives in
            # state.json via the task_summary facade. summary is None
            # only when baseline never wrote state at all (older crashes
            # before the first save_state), in which case err_source
            # stays None and we fall through to the generic INFRA_FAIL
            # hint below.
            summary = task_summary(task_dir) or {}
            err_source = summary.get("baseline_error_source")
            if err_source == "ref":
                hint = ("The file passed via --ref is broken (import / "
                        "forward / device-only bug). Fix the SOURCE file "
                        "and re-run /autoresearch from scratch. The task "
                        "directory is left for inspection but MUST NOT be "
                        "activated — reference.py is not editable.")
            else:
                hint = ("INFRA_FAIL: no per-shape data — the seed kernel "
                        "wasn't meaningfully exercised. Fix env (device / "
                        "eval.timeout / worker / OOM) and re-run "
                        "`/autoresearch --resume <task_dir>`. Phase stays "
                        "at BASELINE.")
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": ("eval pipeline broken during baseline — see "
                          "[baseline]/[eval] stderr above"),
                "hint": hint,
            }))
            sys.exit(4)
        if rc != 0:
            # Unexpected baseline crash (not the 0/4 we know about).
            print(json.dumps({
                "status": "error",
                "task_dir": task_dir,
                "error": (f"baseline crashed unexpectedly (exit {rc}); "
                          f"see [baseline]/[eval] stderr above"),
                "hint": ("This is not a classified outcome. Inspect the "
                         "baseline / eval stderr above and file a bug if "
                         "the exit code isn't in _EXIT_FOR."),
            }))
            sys.exit(rc)

    # Output
    # Surface baseline_outcome so callers can distinguish OK from
    # KERNEL_FAIL without rereading state. Both are activatable
    # (status=ok, rc=0); the difference is whether the seed kernel
    # produced valid timings or the first PLAN cycle has to rewrite it.
    # When --run-baseline wasn't passed, summary is None (no state.json
    # yet) → outcome stays None and the caller knows it's an
    # un-baselined task.
    summary = task_summary(task_dir) or {}
    outcome = summary.get("baseline_outcome")
    print(json.dumps({"task_dir": task_dir, "status": "ok",
                      "baseline_outcome": outcome}))


if __name__ == "__main__":
    main()
