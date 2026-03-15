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

"""
Batch testing script for AKG Kernels Bench Lite with multi-backend support.

This script automatically discovers and tests all kernels in the bench_lite directory,
supporting CPU, GPU (CUDA), and NPU (Ascend) backends.

Features:
- Multi-backend: CPU (cpp), GPU (triton_cuda), NPU (triton_ascend)
- Auto-discover all cases in t1/t2/t3 directories
- Pass@N: Generate N implementations per case, select the best
- Concurrent execution: LLM code gen + device run in parallel (TaskPool)
- Device pool: Parallel execution across multiple devices
- Detailed results and statistics

Usage:
  # Run on CPU (cpp)
  python run_torch_bench_lite.py --backend cpu

  # Run on GPU (triton_cuda)
  python run_torch_bench_lite.py --backend gpu

  # Run on NPU (triton_ascend)
  python run_torch_bench_lite.py --backend npu

  # Custom Pass@N
  python run_torch_bench_lite.py --backend gpu --pass-n 5

  # Custom devices (GPU/NPU)
  python run_torch_bench_lite.py --backend gpu --devices 0 1 2 3

  # Run specific tiers
  python run_torch_bench_lite.py --backend cpu --tiers t1 t2

  # Run specific cases
  python run_torch_bench_lite.py --backend gpu --cases gelu softmax

  # Filter cases by keyword
  python run_torch_bench_lite.py --backend gpu --filter matmul
"""

from akg_agents.op.config.config_validator import load_config
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.environment_check import check_env_for_task
import asyncio
import argparse
import os
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import time

os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'

# Backend configurations
BACKEND_CONFIGS = {
    'cpu': {
        'name': 'CPU',
        'backend': 'cpu',
        'dsl': 'cpp',
        'arch': 'x86_64',
        'devices': [0],
        'skip_npu': True
    },
    'gpu': {
        'name': 'GPU (CUDA)',
        'backend': 'cuda',
        'dsl': 'triton_cuda',
        'arch': 'rtx3090',
        'devices': [0, 1, 2, 3, 4, 5, 6, 7],
        'skip_npu': True
    },
    'npu': {
        'name': 'NPU (Ascend)',
        'backend': 'ascend',
        'dsl': 'triton_ascend',
        'arch': 'ascend910b4',
        'devices': [0],
        'skip_npu': False
    }
}


def discover_bench_lite_cases(
    bench_lite_dir: Path,
    tiers: Optional[List[str]] = None,
    cases: Optional[List[str]] = None,
    filter_pattern: Optional[str] = None,
    skip_npu: bool = True
) -> List[Tuple[str, str, Path]]:
    """
    Discover all kernel files in bench_lite directory.

    Args:
        bench_lite_dir: Path to bench_lite directory
        tiers: Optional list of tiers to include (e.g., ['t1', 't2'])
        cases: Optional list of case names to include (e.g., ['gelu', 'softmax'])
        filter_pattern: Optional keyword to filter case names (e.g., 'matmul', 'norm')
        skip_npu: Skip cases that contain 'torch_npu' (default: True)

    Returns:
        List of tuples (tier, case_name, file_path)
    """
    if not bench_lite_dir.exists():
        print(f"Warning: Bench lite directory not found: {bench_lite_dir}")
        return []

    all_cases = []
    skipped_cases = []

    # Default to all tiers if not specified
    if tiers is None:
        tiers = ['t1', 't2', 't3']

    for tier in tiers:
        tier_dir = bench_lite_dir / tier
        if not tier_dir.exists():
            continue

        for py_file in sorted(tier_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue

            case_name = py_file.stem

            # Filter by case names if specified
            if cases is not None and case_name not in cases:
                continue

            # Filter by pattern if specified
            if filter_pattern is not None and filter_pattern.lower() not in case_name.lower():
                continue

            # Skip NPU cases if requested
            if skip_npu:
                content = py_file.read_text()
                if 'torch_npu' in content:
                    skipped_cases.append((tier, case_name))
                    continue

            all_cases.append((tier, case_name, py_file))

    if skipped_cases and skip_npu:
        print(f"Skipped {len(skipped_cases)} NPU-related cases:")
        for tier, case_name in skipped_cases:
            print(f"  - {tier}/{case_name}")
        print()

    return all_cases


def read_kernel_file(file_path: Path, backend: str) -> str:
    """
    Read the entire kernel file as task description.

    Args:
        file_path: Path to the kernel file
        backend: Target backend ('cpu', 'gpu', 'npu')

    Returns:
        File contents as string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # For GPU backend, convert CPU-specific code to CUDA-compatible code
    if backend == 'gpu':
        content = re.sub(r"device='cpu'", "device='cuda'", content)
        content = re.sub(r'device="cpu"', 'device="cuda"', content)
        content = re.sub(r"\.cpu\(\)", ".cuda()", content)

    return content


def _parse_op_name(op_name: str) -> Optional[Tuple[str, str, str, int]]:
    """Parse op_name like 'cpu_t1_gelu_attempt2' -> (backend, tier, case_name, attempt_id)."""
    match = re.match(r"^(.+?)_(.+?)_(.+?)_attempt(\d+)$", op_name)
    if match:
        return match.group(1), match.group(2), match.group(3), int(match.group(4))
    return None


def _aggregate_results(
    raw_results: list,
    cases: List[Tuple[str, str, Path]],
    backend: str,
    pass_n: int
) -> List[Dict]:
    """
    Aggregate raw task results by case.

    raw_results: from task_pool.wait_all(), each item is (op_name, success, task_info)
    """
    # Group by (tier, case_name)
    case_results = {}  # (tier, case_name) -> list of (attempt_id, success)
    for item in raw_results:
        if len(item) >= 2:
            op_name, success = item[0], item[1]
            parsed = _parse_op_name(op_name)
            if parsed:
                _, tier, case_name, attempt_id = parsed
                key = (tier, case_name)
                if key not in case_results:
                    case_results[key] = []
                case_results[key].append({'attempt_id': attempt_id, 'success': success})

    # Build results in original case order
    results = []
    for tier, case_name, file_path in cases:
        key = (tier, case_name)
        attempt_list = case_results.get(key, [])
        attempt_list.sort(key=lambda x: x['attempt_id'])

        attempts = []
        for a in attempt_list:
            attempts.append({
                'attempt_id': a['attempt_id'],
                'success': a['success'],
                'elapsed_time': None,
                'info': {'task_id': f"{backend}_{tier}_{case_name}_attempt{a['attempt_id']}", 'success': a['success']}
            })

        successful_attempts = [a for a in attempts if a['success']]
        success_count = len(successful_attempts)
        pass_rate = success_count / pass_n if pass_n > 0 else 0.0

        best_attempt = None
        if successful_attempts:
            best_attempt = successful_attempts[0]

        results.append({
            'tier': tier,
            'case_name': case_name,
            'pass_n': pass_n,
            'attempts': attempts,
            'success_count': success_count,
            'pass_rate': pass_rate,
            'best_attempt': best_attempt,
            'total_time': None,
            'overall_success': success_count > 0
        })

    return results


async def run_backend_tests(args, backend: str):
    """
    Run tests for a single backend.

    Args:
        args: Command line arguments
        backend: Backend name ('cpu', 'gpu', 'npu')

    Returns:
        Tuple of (results, total_elapsed)
    """
    config = BACKEND_CONFIGS[backend]

    print("=" * 80)
    print(f"AKG Kernels Bench Lite - {config['name']} with Pass@{args.pass_n}")
    print("=" * 80)
    print()

    # Locate bench_lite directory
    bench_lite_dir = Path(__file__).parent.parent.parent / "benchmark" / "akg_kernels_bench_lite"

    print(f"Bench Lite Directory: {bench_lite_dir}")
    print(f"Pass@N: {args.pass_n}")
    print(f"Devices: {args.devices if args.devices else config['devices']}")
    print(f"Max Concurrent: {args.max_concurrent}")
    print()

    # Discover cases
    print("Discovering cases...")
    if args.filter:
        print(f"Filter: '{args.filter}'")
    cases = discover_bench_lite_cases(
        bench_lite_dir=bench_lite_dir,
        tiers=args.tiers,
        cases=args.cases,
        filter_pattern=args.filter,
        skip_npu=config['skip_npu']
    )

    if not cases:
        print("No cases found!")
        return [], 0.0

    # Group by tier
    tier_counts = {}
    for tier, _, _ in cases:
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    print(f"Found {len(cases)} cases:")
    for tier in sorted(tier_counts.keys()):
        print(f"  {tier}: {tier_counts[tier]} cases")
    print()

    # Register worker
    devices = args.devices if args.devices else config['devices']
    print(f"Registering local worker ({config['name']}, arch={config['arch']})...")
    await register_local_worker(devices, backend=config['backend'], arch=config['arch'])
    print("✓ Worker registered")
    print()

    # Load configuration
    print("Loading configuration...")
    config_obj = load_config(config['dsl'], backend=config['backend'])
    check_env_for_task("torch", config['backend'], config['dsl'], config_obj)
    print("✓ Configuration loaded")
    print()

    # Run tests
    print("=" * 80)
    print("Running Tests (concurrent)")
    print("=" * 80)
    total_tasks = len(cases) * args.pass_n
    print(f"Submitting {total_tasks} tasks ({len(cases)} cases × Pass@{args.pass_n})...")
    print()

    task_pool = TaskPool(max_concurrency=args.max_concurrent)
    start_time = time.time()

    # Pre-read all task descriptions
    case_task_descs = {}
    for tier, case_name, file_path in cases:
        case_task_descs[(tier, case_name)] = read_kernel_file(file_path, backend)

    # Create all tasks (case × attempt)
    for tier, case_name, file_path in cases:
        task_desc = case_task_descs[(tier, case_name)]
        for attempt_id in range(1, args.pass_n + 1):
            task_id = f"{backend}_{tier}_{case_name}_attempt{attempt_id}"
            op_name = f"{backend}_{tier}_{case_name}_attempt{attempt_id}"

            task = LangGraphTask(
                op_name=op_name,
                task_desc=task_desc,
                task_id=task_id,
                dsl=config['dsl'],
                backend=config['backend'],
                arch=config['arch'],
                config=config_obj,
                framework="torch",
                workflow="coder_only_workflow"
            )
            task_pool.create_task(task.run, task_name=task_id)

    raw_results = await task_pool.wait_all()
    total_elapsed = time.time() - start_time

    # Aggregate results by case
    results = _aggregate_results(raw_results, cases, backend, args.pass_n)

    return results, total_elapsed


def print_summary(backend: str, results: List[Dict], total_elapsed: float):
    """Print test summary for a backend."""
    config = BACKEND_CONFIGS[backend]

    print("\n" + "=" * 80)
    print(f"Test Summary - {config['name']}")
    print("=" * 80)
    print()

    # Overall statistics
    total_cases = len(results)
    passed_cases = sum(1 for r in results if r['overall_success'])
    failed_cases = total_cases - passed_cases

    total_attempts = sum(r['pass_n'] for r in results)
    total_successful_attempts = sum(r['success_count'] for r in results)
    overall_pass_rate = total_successful_attempts / total_attempts if total_attempts > 0 else 0.0

    print(f"Total Cases: {total_cases}")
    print(f"Passed Cases: {passed_cases} ({passed_cases/total_cases:.1%})")
    print(f"Failed Cases: {failed_cases} ({failed_cases/total_cases:.1%})")
    print()
    print(f"Total Attempts: {total_attempts}")
    print(f"Successful Attempts: {total_successful_attempts} ({overall_pass_rate:.1%})")
    print(f"Total Time: {total_elapsed:.2f}s")
    print()

    # Per-tier statistics
    print("Per-Tier Statistics:")
    print("-" * 80)
    tier_stats = {}
    for result in results:
        tier = result['tier']
        if tier not in tier_stats:
            tier_stats[tier] = {
                'total': 0,
                'passed': 0,
                'attempts': 0,
                'successful_attempts': 0
            }
        tier_stats[tier]['total'] += 1
        if result['overall_success']:
            tier_stats[tier]['passed'] += 1
        tier_stats[tier]['attempts'] += result['pass_n']
        tier_stats[tier]['successful_attempts'] += result['success_count']

    for tier in sorted(tier_stats.keys()):
        stats = tier_stats[tier]
        case_pass_rate = stats['passed'] / stats['total'] if stats['total'] > 0 else 0.0
        attempt_pass_rate = stats['successful_attempts'] / stats['attempts'] if stats['attempts'] > 0 else 0.0
        print(f"{tier}: {stats['passed']}/{stats['total']} cases ({case_pass_rate:.1%}), "
              f"{stats['successful_attempts']}/{stats['attempts']} attempts ({attempt_pass_rate:.1%})")
    print()

    # Detailed results
    print("Detailed Results:")
    print("-" * 80)
    print(f"{'No.':<5} {'Tier':<6} {'Case':<30} {'Pass@N':<8} {'Success':<10} {'Best Time':<12}")
    print("-" * 80)

    for i, result in enumerate(results, 1):
        tier = result['tier']
        case_name = result['case_name']
        pass_n_str = f"{result['success_count']}/{result['pass_n']}"
        success_str = "✓ PASS" if result['overall_success'] else "✗ FAIL"

        if result['best_attempt'] and result['best_attempt'].get('elapsed_time') is not None:
            best_time_str = f"{result['best_attempt']['elapsed_time']:.2f}s"
        else:
            best_time_str = "N/A"

        print(f"{i:<5} {tier:<6} {case_name:<30} {pass_n_str:<8} {success_str:<10} {best_time_str:<12}")

    print("-" * 80)
    print()

    # Failed cases
    if failed_cases > 0:
        print("Failed Cases (0 successful attempts):")
        for result in results:
            if not result['overall_success']:
                print(f"  ✗ {result['tier']}/{result['case_name']}")
        print()

    print("=" * 80)
    print(f"Batch testing completed! ({passed_cases}/{total_cases} cases passed)")
    print("=" * 80)
    print()

    return {
        'total_cases': total_cases,
        'passed_cases': passed_cases,
        'failed_cases': failed_cases,
        'total_attempts': total_attempts,
        'successful_attempts': total_successful_attempts,
        'overall_pass_rate': overall_pass_rate,
        'total_time': total_elapsed,
        'tier_stats': tier_stats,
        'results': results
    }


async def run_bench_lite_tests(args):
    """
    Main function to run bench lite tests with multi-backend support.
    """
    backend = args.backend

    if backend not in BACKEND_CONFIGS:
        print(f"Error: Unknown backend '{backend}'. Available: {list(BACKEND_CONFIGS.keys())}")
        return

    results, total_elapsed = await run_backend_tests(args, backend)
    if not results:
        return
    summary = print_summary(backend, results, total_elapsed)

    # Save results to JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'backend': backend,
                'pass_n': args.pass_n,
                'devices': args.devices if args.devices else None,
                'max_concurrent': args.max_concurrent,
                'tiers': args.tiers,
                'cases': args.cases,
                'filter': args.filter
            },
            'results': summary
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {output_path}")
        print()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run AKG Kernels Bench Lite with multi-backend support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on CPU
  python run_torch_bench_lite.py --backend cpu

  # Run on GPU (CUDA)
  python run_torch_bench_lite.py --backend gpu

  # Run on NPU (Ascend)
  python run_torch_bench_lite.py --backend npu

  # Run on all backends
  python run_torch_bench_lite.py --backend all

  # Custom Pass@N
  python run_torch_bench_lite.py --backend gpu --pass-n 5

  # Custom devices (GPU/NPU)
  python run_torch_bench_lite.py --backend gpu --devices 0 1 2 3

  # Run specific tiers
  python run_torch_bench_lite.py --backend cpu --tiers t1 t2

  # Run specific cases
  python run_torch_bench_lite.py --backend gpu --cases gelu softmax

  # Filter cases by keyword
  python run_torch_bench_lite.py --backend gpu --filter matmul

  # Save results to file
  python run_torch_bench_lite.py --backend gpu --output results.json
        """
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=['cpu', 'gpu', 'npu'],
        default='gpu',
        help="Backend to run on: cpu, gpu, or npu (default: gpu)"
    )

    parser.add_argument(
        "--pass-n",
        type=int,
        default=3,
        help="Number of attempts per case (Pass@N), default: 3"
    )

    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=None,
        help="Device IDs (default: CPU: [0], GPU: [0-7], NPU: [0])"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum concurrent tasks, default: 4"
    )

    parser.add_argument(
        "--tiers",
        type=str,
        nargs="+",
        default=None,
        help="Specific tiers to run (e.g., t1 t2), default: all tiers"
    )

    parser.add_argument(
        "--cases",
        type=str,
        nargs="+",
        default=None,
        help="Specific cases to run (e.g., gelu softmax), default: all cases"
    )

    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter cases by keyword (e.g., 'matmul', 'norm'), default: no filter"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path for results, default: no output file"
    )

    args = parser.parse_args()

    asyncio.run(run_bench_lite_tests(args))


if __name__ == "__main__":
    main()
