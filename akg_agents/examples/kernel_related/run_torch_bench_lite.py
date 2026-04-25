#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Benchmark Lite runner with multi-backend support.

Supports three modes:
  - correctness: Pass@N correctness evaluation only (default, backward-compatible)
  - performance: Correctness + performance evaluation + scoring (no leaderboard)
  - full: Correctness + performance + scoring + leaderboard

All modes run correctness first, then extract submissions from Agent results
for performance measurement (performance and full modes).

Note: ``--submission-dir`` is an *output* directory where extracted Agent
submissions are written, not an input for pre-existing submissions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

os.environ["AKG_AGENTS_STREAM_OUTPUT"] = "on"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_lite_common import (  # noqa: E402
    aggregate_correctness_results,
    build_environment_info,
    build_full_output_payload,
    build_task_specs,
    combine_backend_payloads,
    compute_performance_summary,
    compute_summary,
    discover_bench_lite_cases,
    extract_submissions_from_results,
    get_bench_lite_dir,
    print_backend_summary,
    print_full_summary,
    print_performance_summary,
    resolve_backend_config,
    resolve_devices,
    run_performance_evaluation,
    write_leaderboard,
    DEFAULT_PERF_WARMUP,
    DEFAULT_PERF_ITERATIONS,
    DEFAULT_PERF_NUM_TRIALS,
    DEFAULT_PERF_RTOL,
    DEFAULT_PERF_ATOL,
    DEFAULT_PERF_TIMEOUT,
    DEFAULT_TEAM_NAME,
    validate_team_name,
)
from akg_agents.core.async_pool.task_pool import TaskPool  # noqa: E402
from akg_agents.core.worker.manager import register_local_worker  # noqa: E402
from akg_agents.op.config.config_validator import load_config  # noqa: E402
from akg_agents.op.langgraph_op.task import LangGraphTask  # noqa: E402
from akg_agents.utils.environment_check import check_env_for_task  # noqa: E402


class RunnerConfigPayload(TypedDict, total=False):
    mode: str
    backend: str
    arch: Optional[str]
    dsl: Optional[str]
    backend_name: Optional[str]
    devices: List[int]
    pass_n: int
    max_concurrent: int
    tiers: Optional[List[str]]
    cases: Optional[List[str]]
    filter: Optional[str]
    team_name: str
    workflow: str
    warmup: int
    iterations: int
    num_trials: int
    rtol: float
    atol: float
    timeout: int


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    return argparse.ArgumentParser(
        description="Run Benchmark Lite evaluation (correctness, performance, or full pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Correctness only (default, backward-compatible)
  python run_torch_bench_lite.py --backend gpu

  # Full pipeline: correctness + performance + scoring
  python run_torch_bench_lite.py --backend gpu --mode full

  # Full pipeline with custom performance parameters
  python run_torch_bench_lite.py --backend gpu --mode full --warmup 20 --iterations 200

  # Save full results and leaderboard
  python run_torch_bench_lite.py --backend gpu --mode full --output results.json

  # Override arch
  python run_torch_bench_lite.py --backend gpu --arch a100

  # Custom Pass@N
  python run_torch_bench_lite.py --backend gpu --pass-n 5

  # Run specific tiers or cases
  python run_torch_bench_lite.py --backend gpu --tiers t1 t2
  python run_torch_bench_lite.py --backend gpu --cases gelu softmax
  python run_torch_bench_lite.py --backend gpu --filter matmul
        """,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse and validate command line arguments.

    Args:
        argv: Optional list of command line arguments.

    Returns:
        Parsed and validated arguments namespace.

    Raises:
        SystemExit: If validation fails.
    """
    parser = create_parser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["correctness", "performance", "full"],
        default="correctness",
        help="Evaluation mode: correctness (default), performance (correctness + perf), or full (+ leaderboard)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["cpu", "gpu", "npu", "all"],
        default="gpu",
        help="Backend to run on: cpu, gpu, npu, or all (default: gpu)",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Override backend default arch (for example: x86_64, rtx3090, ascend910b4)",
    )
    parser.add_argument(
        "--dsl",
        type=str,
        default=None,
        help="Override backend default DSL (for example: cpp, triton_cuda, triton_ascend, pytorch)",
    )
    parser.add_argument(
        "--backend-name",
        type=str,
        default=None,
        help="Override backend mapping name passed to worker/config (for example: cpu, cuda, ascend)",
    )
    parser.add_argument(
        "--pass-n",
        type=int,
        default=3,
        help="Number of attempts per case (Pass@N), default: 3",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=None,
        help="Optional device IDs. Defaults are resolved per backend at runtime.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum concurrent tasks, default: 4",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        nargs="+",
        default=None,
        help="Specific tiers to run (for example: t1 t2). Default: all tiers.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        nargs="+",
        default=None,
        help="Specific cases to run (for example: gelu softmax). Default: all cases.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter cases by keyword (for example: matmul, norm).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for CI artifacts.",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default=None,
        help="Directory to save extracted Agent submissions (default: auto-generated in log_dir).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_PERF_WARMUP,
        help=f"Performance measurement warmup runs (default: {DEFAULT_PERF_WARMUP})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_PERF_ITERATIONS,
        help=f"Performance measurement iterations per trial (default: {DEFAULT_PERF_ITERATIONS})",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_PERF_NUM_TRIALS,
        help=f"Performance measurement trial count (default: {DEFAULT_PERF_NUM_TRIALS})",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_PERF_RTOL,
        help=f"Relative tolerance for post-verification (default: {DEFAULT_PERF_RTOL})",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_PERF_ATOL,
        help=f"Absolute tolerance for post-verification (default: {DEFAULT_PERF_ATOL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_PERF_TIMEOUT,
        help=f"Per-case timeout in seconds (default: {DEFAULT_PERF_TIMEOUT})",
    )
    parser.add_argument(
        "--team-name",
        type=str,
        default=DEFAULT_TEAM_NAME,
        help=f"Team/agent identity for submissions and leaderboard (default: {DEFAULT_TEAM_NAME})",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default="coder_only_workflow",
        help="LangGraph workflow name (default: coder_only_workflow)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=None,
        help="Backend execution order for --backend all (default: cpu gpu npu)",
    )
    args = parser.parse_args(argv)

    # Validate pass-n
    if args.pass_n < 1:
        parser.error("--pass-n must be at least 1")

    # Validate devices if provided
    if args.devices:
        if any(d < 0 for d in args.devices):
            parser.error("--devices must contain non-negative integers only")
        if len(args.devices) != len(set(args.devices)):
            parser.error("--devices must contain unique device IDs")

    # --backends only makes sense with --backend all
    if args.backends and args.backend != "all":
        parser.error("--backends can only be used with --backend all")

    # Validate backends if provided
    if args.backends:
        valid_backends = {"cpu", "gpu", "npu"}
        invalid = set(args.backends) - valid_backends
        if invalid:
            parser.error(f"--backends contains invalid values: {invalid}. Valid: {valid_backends}")
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for b in args.backends:
            if b not in seen:
                seen.add(b)
                deduped.append(b)
        args.backends = deduped

    # --arch/--dsl/--backend-name are per-backend overrides; broadcasting them
    # to all backends in --backend all mode is semantically invalid.
    if args.backend == "all":
        if args.arch is not None:
            parser.error("--arch cannot be used with --backend all (each backend has its own default arch)")
        if args.dsl is not None:
            parser.error("--dsl cannot be used with --backend all (each backend has its own default dsl)")
        if args.backend_name is not None:
            parser.error("--backend-name cannot be used with --backend all (each backend has its own mapping name)")

    # Validate team-name (path traversal prevention)
    try:
        validate_team_name(args.team_name)
    except ValueError as exc:
        parser.error(str(exc))

    # Validate performance parameters
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.num_trials < 1:
        parser.error("--num-trials must be at least 1")
    if args.rtol < 0:
        parser.error("--rtol must be non-negative")
    if args.atol < 0:
        parser.error("--atol must be non-negative")
    if args.timeout < 1:
        parser.error("--timeout must be at least 1")

    return args


def build_runner_config(
    args: argparse.Namespace,
    backend: str,
    devices: List[int],
    arch: Optional[str],
    dsl: Optional[str] = None,
) -> RunnerConfigPayload:
    config: RunnerConfigPayload = {
        "mode": args.mode,
        "backend": backend,
        "arch": arch,
        "dsl": dsl,
        "backend_name": args.backend_name,
        "devices": devices,
        "pass_n": args.pass_n,
        "max_concurrent": args.max_concurrent,
        "tiers": args.tiers,
        "cases": args.cases,
        "filter": args.filter,
        "team_name": args.team_name,
        "workflow": args.workflow,
    }
    if args.mode in ("performance", "full"):
        config["warmup"] = args.warmup
        config["iterations"] = args.iterations
        config["num_trials"] = args.num_trials
        config["rtol"] = args.rtol
        config["timeout"] = args.timeout
        config["atol"] = args.atol
    return config


async def run_single_backend(args: argparse.Namespace, backend: str) -> Tuple[Dict[str, object], int]:
    backend_config = resolve_backend_config(
        backend, args.arch,
        dsl_override=args.dsl,
        backend_override=args.backend_name,
    )

    perf_results = None
    perf_summary = None
    perf_config = None
    discovery = None
    config_payload = None
    environment = None

    try:
        devices = resolve_devices(backend, args.devices)
        bench_lite_dir = get_bench_lite_dir(Path(__file__))

        mode_label = args.mode.upper()
        print("=" * 80)
        print(f"Benchmark Lite Runner [{mode_label}] - {backend_config['name']}")
        print("=" * 80)
        print(f"Bench Lite Directory: {bench_lite_dir}")
        print(f"Mode: {args.mode}")
        print(f"Pass@N: {args.pass_n}")
        print(f"Arch: {backend_config['arch']}")
        print(f"Devices: {devices}")
        print(f"Max Concurrent: {args.max_concurrent}")
        if args.mode in ("performance", "full"):
            print(f"Perf Warmup: {args.warmup}  Iterations: {args.iterations}  Trials: {args.num_trials}")
            print(f"Perf Tolerance: rtol={args.rtol}  atol={args.atol}")
        print()

        discovery = discover_bench_lite_cases(
            bench_lite_dir=bench_lite_dir,
            tiers=args.tiers,
            cases=args.cases,
            filter_pattern=args.filter,
            skip_npu=bool(backend_config["skip_npu"]),
        )

        environment = build_environment_info(backend, backend_config, devices)
        config_payload = build_runner_config(args, backend, devices, str(backend_config["arch"]), str(backend_config["dsl"]))

        print("Registering local worker...")
        await register_local_worker(devices, backend=str(backend_config["backend"]), arch=str(backend_config["arch"]))

        print("Loading backend configuration...")
        config_obj = load_config(str(backend_config["dsl"]), backend=str(backend_config["backend"]))
        check_env_for_task("torch", str(backend_config["backend"]), str(backend_config["dsl"]), config_obj)

        if not discovery.cases:
            summary = compute_summary(
                results=[],
                total_elapsed=0.0,
                skipped_cases=discovery.skipped_cases,
                environment_error="No runnable cases found for the current selection",
            )
            # In performance/full mode, include empty perf fields for schema consistency
            empty_perf = [] if args.mode in ("performance", "full") else None
            empty_perf_summary = compute_performance_summary([]) if empty_perf is not None else None
            empty_perf_config = {
                "warmup": args.warmup, "iterations": args.iterations,
                "num_trials": args.num_trials, "rtol": args.rtol,
                "atol": args.atol, "timeout": args.timeout,
            } if empty_perf is not None else None
            payload = build_full_output_payload(
                backend=backend,
                mode=args.mode,
                config_payload=config_payload,
                environment=environment,
                summary=summary,
                results=[],
                perf_results=empty_perf,
                perf_summary=empty_perf_summary,
                perf_config=empty_perf_config,
            )
            print_backend_summary(str(backend_config["name"]), summary, [])
            return payload, 1

        # === Phase 1: Correctness (Pass@N) ===
        task_pool = TaskPool(max_concurrency=args.max_concurrent)
        start_time = time.time()
        for spec in build_task_specs(discovery.cases, backend, backend_config, args.pass_n):
            task = LangGraphTask(
                op_name=str(spec["op_name"]),
                task_desc=str(spec["task_desc"]),
                task_id=str(spec["task_id"]),
                dsl=str(spec["dsl"]),
                backend=str(spec["backend"]),
                arch=str(spec["arch"]),
                config=config_obj,
                framework="torch",
                workflow=args.workflow,
            )
            task_pool.create_task(task.run, task_name=str(spec["task_id"]))

        raw_results = await task_pool.wait_all()
        total_elapsed = time.time() - start_time
        results = aggregate_correctness_results(raw_results, discovery.cases, args.pass_n, backend)
        summary = compute_summary(results, total_elapsed, discovery.skipped_cases)

        print_backend_summary(str(backend_config["name"]), summary, results)

        # === Phase 2 & 3: Submission extraction + Performance evaluation ===
        if args.mode in ("performance", "full"):
            submission_base = Path(args.submission_dir) if args.submission_dir else (
                Path(config_obj.get("log_dir", "")).expanduser() / "bench_lite_submissions"
                if config_obj.get("log_dir") else
                bench_lite_dir / "submissions"
            )
            submission_base.mkdir(parents=True, exist_ok=True)

            print(f"\nExtracting Agent submissions to: {submission_base}")
            team_dir = extract_submissions_from_results(
                raw_results=raw_results,
                cases=discovery.cases,
                bench_lite_dir=bench_lite_dir,
                submission_dir=submission_base,
                backend=backend,
                team_name=args.team_name,
            )

            extracted_count = sum(
                1 for _ in team_dir.rglob("*.py") if _.name != "__init__.py" and _.name != "meta.json"
            )
            print(f"Extracted {extracted_count} submission files.\n")

            perf_config = {
                "warmup": args.warmup,
                "iterations": args.iterations,
                "num_trials": args.num_trials,
                "rtol": args.rtol,
                "atol": args.atol,
                "timeout": args.timeout,
            }
            if extracted_count > 0:
                print("Running performance evaluation...")
                perf_results = run_performance_evaluation(
                    team_dir=team_dir,
                    bench_lite_dir=bench_lite_dir,
                    warmup_runs=args.warmup,
                    iterations=args.iterations,
                    num_trials=args.num_trials,
                    rtol=args.rtol,
                    atol=args.atol,
                    timeout=args.timeout,
                    backend=backend,
                )
                perf_summary = compute_performance_summary(perf_results)
                print_performance_summary(str(backend_config["name"]), perf_results, perf_summary)
            else:
                print("[WARN] No successful submissions extracted - performance evaluation skipped.")
                perf_results = []
                perf_summary = compute_performance_summary(perf_results)

        # === Phase 4: Build payload ===
        if args.mode != "correctness":
            print_full_summary(
                str(backend_config["name"]), summary, results, perf_results, perf_summary,
                mode=args.mode,
            )
        payload = build_full_output_payload(
            backend=backend,
            mode=args.mode,
            config_payload=config_payload,
            environment=environment,
            summary=summary,
            results=results,
            perf_results=perf_results,
            perf_summary=perf_summary,
            perf_config=perf_config,
            submission_dir=str(submission_base) if args.mode in ("performance", "full") else None,
        )

        # Exit code: non-zero if any correctness failure OR performance-phase failure
        # In performance/full mode, 0 extracted submissions is also a failure
        has_correctness_failure = summary["failed_cases"] > 0
        has_perf_failure = (
            perf_summary is not None
            and perf_summary.get("failed_cases", 0) > 0
        )
        has_no_perf_data = (
            args.mode in ("performance", "full")
            and perf_summary is not None
            and perf_summary.get("total_cases", 0) == 0
            and summary.get("passed_cases", 0) > 0
        )
        return payload, 1 if (has_correctness_failure or has_perf_failure or has_no_perf_data) else 0
    except Exception as exc:
        skipped = discovery.skipped_cases if discovery is not None else []
        fallback_config = config_payload if config_payload is not None else build_runner_config(
            args, backend, [], str(backend_config["arch"]), str(backend_config["dsl"]))
        fallback_env = environment if environment is not None else {
            "framework": "torch",
            "dsl": str(backend_config["dsl"]),
            "backend": backend,
            "visible_devices": [],
        }
        summary = compute_summary(
            results=[],
            total_elapsed=0.0,
            skipped_cases=skipped,
            environment_error=str(exc),
        )
        # In performance/full mode, include empty perf fields for schema consistency
        fail_perf_results = [] if args.mode in ("performance", "full") else None
        fail_perf_summary = compute_performance_summary([]) if fail_perf_results is not None else None
        fail_perf_config = {
            "warmup": args.warmup, "iterations": args.iterations,
            "num_trials": args.num_trials, "rtol": args.rtol,
            "atol": args.atol, "timeout": args.timeout,
        } if fail_perf_results is not None else None
        payload = build_full_output_payload(
            backend=backend,
            mode=args.mode,
            config_payload=fallback_config,
            environment=fallback_env,
            summary=summary,
            results=[],
            perf_results=fail_perf_results,
            perf_summary=fail_perf_summary,
            perf_config=fail_perf_config,
        )
        print_backend_summary(str(backend_config["name"]), summary, [])
        return payload, 1


def write_output(output_path: str, payload: Dict[str, object]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Results saved to: {path}")


async def run_bench_lite_tests(args: argparse.Namespace) -> int:
    if args.backend != "all":
        payload, exit_code = await run_single_backend(args, args.backend)
        if args.output:
            write_output(args.output, payload)
            if args.mode == "full" and payload.get("performance_results"):
                lb_path = Path(args.output).with_name("leaderboard.json")
                write_leaderboard(
                    payload["performance_results"],
                    payload["performance_summary"],
                    lb_path,
                    args.backend,
                    team_name=args.team_name,
                )
        return exit_code

    backend_payloads: Dict[str, Dict[str, object]] = {}
    exit_code = 0
    all_backends = args.backends or ["cpu", "gpu", "npu"]
    for backend in all_backends:
        payload, backend_exit_code = await run_single_backend(args, backend)
        backend_payloads[backend] = payload
        exit_code = max(exit_code, backend_exit_code)

    combined_summary = combine_backend_payloads(backend_payloads)
    combined_results = combined_summary.pop("results", [])

    # Aggregate performance results from all backends (if any ran perf evaluation)
    combined_perf_results: Optional[List[Dict[str, object]]] = None
    combined_perf_summary: Optional[Dict[str, object]] = None
    combined_perf_config: Optional[Dict[str, object]] = None
    if args.mode in ("performance", "full"):
        all_perf_results: List[Dict[str, object]] = []
        for bk, bp in backend_payloads.items():
            for r in bp.get("performance_results", []):
                entry = dict(r)
                entry["backend"] = bk
                all_perf_results.append(entry)
        # Always produce perf fields for schema consistency (even if empty)
        combined_perf_results = all_perf_results
        combined_perf_summary = compute_performance_summary(all_perf_results)
        combined_perf_config = {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "num_trials": args.num_trials,
            "rtol": args.rtol,
            "atol": args.atol,
            "timeout": args.timeout,
        }

    # Aggregate actual config/environment from each backend's payload
    per_backend_env = {}
    per_backend_arch = {}
    for bk, bp in backend_payloads.items():
        bk_env = bp.get("environment", {})
        per_backend_env[bk] = {
            "visible_devices": bk_env.get("visible_devices", []),
            "dsl": bk_env.get("dsl", "unknown"),
        }
        bk_cfg = bp.get("config", {})
        per_backend_arch[bk] = bk_cfg.get("arch", args.arch)

    combined_payload = build_full_output_payload(
        backend="all",
        mode=args.mode,
        config_payload={
            "mode": args.mode,
            "backend": "all",
            "backends": all_backends,
            "arch": per_backend_arch,
            "devices": args.devices,
            "pass_n": args.pass_n,
            "max_concurrent": args.max_concurrent,
            "tiers": args.tiers,
            "cases": args.cases,
            "filter": args.filter,
            "team_name": args.team_name,
            "workflow": args.workflow,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "num_trials": args.num_trials,
            "rtol": args.rtol,
            "atol": args.atol,
            "timeout": args.timeout,
        },
        environment={
            "framework": "torch",
            "dsl": "multiple",
            "backend": "all",
            "per_backend": per_backend_env,
        },
        summary=combined_summary,
        results=combined_results,
        perf_results=combined_perf_results,
        perf_summary=combined_perf_summary,
        perf_config=combined_perf_config,
        backend_results=backend_payloads,
    )

    print("\n" + "=" * 80)
    print(f"Benchmark Lite Summary [{args.mode.upper()}] - ALL BACKENDS")
    print("=" * 80)
    print(
        f"Passed Cases: {combined_summary['passed_cases']}/{combined_summary['total_cases']} "
        f"({combined_summary['case_pass_rate']:.1%})"
    )
    print(
        f"Successful Attempts: {combined_summary['successful_attempts']}/{combined_summary['total_attempts']} "
        f"({combined_summary['attempt_pass_rate']:.1%})"
    )
    if combined_summary.get("environment_error"):
        print(f"Environment Errors: {combined_summary['environment_error']}")

    if args.output:
        write_output(args.output, combined_payload)
        if args.mode == "full" and combined_perf_results:
            lb_path = Path(args.output).with_name("leaderboard_all.json")
            write_leaderboard(combined_perf_results, combined_perf_summary, lb_path, "all",
                              team_name=args.team_name)
    return exit_code


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    sys.exit(asyncio.run(run_bench_lite_tests(args)))


if __name__ == "__main__":
    main()
