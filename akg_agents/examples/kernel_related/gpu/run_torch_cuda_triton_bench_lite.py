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
Batch testing script for AKG Kernels Bench Lite on GPU with Pass@N support.

This script automatically discovers and tests all kernels in the bench_lite directory,
generating optimized Triton CUDA code with multiple attempts (Pass@N).

Features:
- Auto-discover all cases in t1/t2/t3 directories
- Pass@N: Generate N implementations per case, select the best
- Device pool: Parallel execution across multiple GPUs
- Detailed results and statistics

Usage:
  # Run with default settings (Pass@3, 4 GPUs)
  python run_torch_cuda_triton_bench_lite.py
  
  # Custom Pass@N
  python run_torch_cuda_triton_bench_lite.py --pass-n 5
  
  # Custom devices
  python run_torch_cuda_triton_bench_lite.py --devices 0 1 2 3
  
  # Run specific tiers
  python run_torch_cuda_triton_bench_lite.py --tiers t1 t2
  
  # Run specific cases
  python run_torch_cuda_triton_bench_lite.py --cases gelu softmax
  
  # Filter cases by keyword (for quick testing)
  python run_torch_cuda_triton_bench_lite.py --filter matmul
  python run_torch_cuda_triton_bench_lite.py --filter norm --pass-n 1
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


def discover_bench_lite_cases(
    bench_lite_dir: Path,
    tiers: Optional[List[str]] = None,
    cases: Optional[List[str]] = None,
    filter_pattern: Optional[str] = None
) -> List[Tuple[str, str, Path]]:
    """
    Discover all kernel files in bench_lite directory.
    
    Args:
        bench_lite_dir: Path to bench_lite directory
        tiers: Optional list of tiers to include (e.g., ['t1', 't2'])
        cases: Optional list of case names to include (e.g., ['gelu', 'softmax'])
        filter_pattern: Optional keyword to filter case names (e.g., 'matmul', 'norm')
        
    Returns:
        List of tuples (tier, case_name, file_path)
    """
    if not bench_lite_dir.exists():
        print(f"Warning: Bench lite directory not found: {bench_lite_dir}")
        return []
    
    all_cases = []
    
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
            
            all_cases.append((tier, case_name, py_file))
    
    return all_cases


def convert_cpu_to_cuda(content: str) -> str:
    """
    Convert CPU-specific code to CUDA-compatible code.
    
    Args:
        content: Original file content
        
    Returns:
        Converted content for CUDA
    """
    content = re.sub(r"device='cpu'", "device='cuda'", content)
    content = re.sub(r'device="cpu"', 'device="cuda"', content)
    content = re.sub(r"\.cpu\(\)", ".cuda()", content)
    
    return content


def read_kernel_file(file_path: Path) -> str:
    """
    Read the entire kernel file as task description and convert for CUDA.
    
    Args:
        file_path: Path to the kernel file
        
    Returns:
        File contents as string (converted for CUDA)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return convert_cpu_to_cuda(content)


async def run_single_attempt(
    tier: str,
    case_name: str,
    task_desc: str,
    attempt_id: int,
    total_attempts: int,
    config: dict
) -> Tuple[str, str, int, bool, float, Optional[Dict]]:
    """
    Run a single attempt for a case.
    
    Args:
        tier: Tier name (t1, t2, t3)
        case_name: Case name
        task_desc: Task description (kernel code)
        attempt_id: Current attempt number (1-indexed)
        total_attempts: Total number of attempts
        config: Configuration dictionary
        
    Returns:
        Tuple of (tier, case_name, attempt_id, success, elapsed_time, result_info)
    """
    start_time = time.time()
    
    try:
        task_pool = TaskPool()
        
        # Create unique task ID for this attempt
        task_id = f"{tier}_{case_name}_attempt{attempt_id}"
        op_name = f"{tier}_{case_name}"
        
        task = LangGraphTask(
            op_name=op_name,
            task_desc=task_desc,
            task_id=task_id,
            dsl="triton_cuda",
            backend="cuda",
            arch="rtx3090",
            config=config,
            framework="torch",
            workflow="coder_only_workflow"
        )
        
        task_pool.create_task(task.run)
        results = await task_pool.wait_all()
        
        for op_name_result, result, _ in results:
            elapsed_time = time.time() - start_time
            
            result_info = {
                'task_id': task_id,
                'elapsed_time': elapsed_time,
                'success': result
            }
            
            return (tier, case_name, attempt_id, result, elapsed_time, result_info)
        
        elapsed_time = time.time() - start_time
        return (tier, case_name, attempt_id, False, elapsed_time, None)
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"  ✗ Attempt {attempt_id}/{total_attempts} ERROR: {str(e)}")
        return (tier, case_name, attempt_id, False, elapsed_time, None)


async def run_case_with_pass_n(
    tier: str,
    case_name: str,
    file_path: Path,
    pass_n: int,
    config: dict,
    case_index: int,
    total_cases: int
) -> Dict:
    """
    Run a single case with Pass@N attempts.
    
    Args:
        tier: Tier name
        case_name: Case name
        file_path: Path to the kernel file
        pass_n: Number of attempts
        config: Configuration dictionary
        case_index: Current case index (1-indexed)
        total_cases: Total number of cases
        
    Returns:
        Dictionary with case results
    """
    print(f"\n[{case_index}/{total_cases}] {tier}/{case_name}")
    print("=" * 80)
    
    # Read task description
    task_desc = read_kernel_file(file_path)
    
    # Run all attempts
    print(f"Running Pass@{pass_n} attempts...")
    
    attempts = []
    for attempt_id in range(1, pass_n + 1):
        print(f"  Attempt {attempt_id}/{pass_n}...", end=" ", flush=True)
        
        tier_result, case_result, attempt_num, success, elapsed, info = await run_single_attempt(
            tier=tier,
            case_name=case_name,
            task_desc=task_desc,
            attempt_id=attempt_id,
            total_attempts=pass_n,
            config=config
        )
        
        attempts.append({
            'attempt_id': attempt_id,
            'success': success,
            'elapsed_time': elapsed,
            'info': info
        })
        
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status} ({elapsed:.2f}s)")
    
    # Calculate statistics
    successful_attempts = [a for a in attempts if a['success']]
    success_count = len(successful_attempts)
    pass_rate = success_count / pass_n if pass_n > 0 else 0.0
    
    best_attempt = None
    if successful_attempts:
        # Select the fastest successful attempt
        best_attempt = min(successful_attempts, key=lambda x: x['elapsed_time'])
    
    total_time = sum(a['elapsed_time'] for a in attempts)
    
    # Print summary
    print(f"\nResults: {success_count}/{pass_n} passed ({pass_rate:.1%})")
    if best_attempt:
        print(f"Best attempt: #{best_attempt['attempt_id']} ({best_attempt['elapsed_time']:.2f}s)")
    print(f"Total time: {total_time:.2f}s")
    
    return {
        'tier': tier,
        'case_name': case_name,
        'pass_n': pass_n,
        'attempts': attempts,
        'success_count': success_count,
        'pass_rate': pass_rate,
        'best_attempt': best_attempt,
        'total_time': total_time,
        'overall_success': success_count > 0
    }


async def run_bench_lite_tests(args):
    """
    Main function to run bench lite tests with Pass@N.
    """
    print("=" * 80)
    print("AKG Kernels Bench Lite - GPU (Triton CUDA) with Pass@N")
    print("=" * 80)
    print()
    
    # Locate bench_lite directory
    bench_lite_dir = Path(__file__).parent.parent.parent.parent / "benchmark" / "akg_kernels_bench_lite"
    
    print(f"Bench Lite Directory: {bench_lite_dir}")
    print(f"Pass@N: {args.pass_n}")
    print(f"Devices: {args.devices}")
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
        filter_pattern=args.filter
    )
    
    if not cases:
        print("No cases found!")
        return
    
    # Group by tier
    tier_counts = {}
    for tier, _, _ in cases:
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    
    print(f"Found {len(cases)} cases:")
    for tier in sorted(tier_counts.keys()):
        print(f"  {tier}: {tier_counts[tier]} cases")
    print()
    
    # Register worker
    print("Registering local worker (CUDA RTX 3090)...")
    await register_local_worker(args.devices, backend="cuda", arch="rtx3090")
    print("✓ Worker registered")
    print()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config("triton_cuda", backend="cuda")
    check_env_for_task("torch", "cuda", "triton_cuda", config)
    print("✓ Configuration loaded")
    print()
    
    # Run tests
    print("=" * 80)
    print("Running Tests")
    print("=" * 80)
    
    start_time = time.time()
    results = []
    
    for i, (tier, case_name, file_path) in enumerate(cases, 1):
        result = await run_case_with_pass_n(
            tier=tier,
            case_name=case_name,
            file_path=file_path,
            pass_n=args.pass_n,
            config=config,
            case_index=i,
            total_cases=len(cases)
        )
        results.append(result)
    
    total_elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
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
        
        if result['best_attempt']:
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
    
    # Save results to JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'pass_n': args.pass_n,
                'devices': args.devices,
                'max_concurrent': args.max_concurrent,
                'tiers': args.tiers,
                'cases': args.cases,
                'filter': args.filter
            },
            'summary': {
                'total_cases': total_cases,
                'passed_cases': passed_cases,
                'failed_cases': failed_cases,
                'total_attempts': total_attempts,
                'successful_attempts': total_successful_attempts,
                'overall_pass_rate': overall_pass_rate,
                'total_time': total_elapsed
            },
            'tier_stats': tier_stats,
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        print()
    
    print("=" * 80)
    print(f"Batch testing completed! ({passed_cases}/{total_cases} cases passed)")
    print("=" * 80)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run AKG Kernels Bench Lite on GPU with Pass@N",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (Pass@3, 4 GPUs)
  python run_torch_cuda_triton_bench_lite.py
  
  # Custom Pass@N
  python run_torch_cuda_triton_bench_lite.py --pass-n 5
  
  # Custom devices
  python run_torch_cuda_triton_bench_lite.py --devices 0 1 2 3
  
  # Run specific tiers
  python run_torch_cuda_triton_bench_lite.py --tiers t1 t2
  
  # Run specific cases
  python run_torch_cuda_triton_bench_lite.py --cases gelu softmax
  
  # Filter cases by keyword (quick testing)
  python run_torch_cuda_triton_bench_lite.py --filter matmul --pass-n 1
  python run_torch_cuda_triton_bench_lite.py --filter norm --devices 0 1
  
  # Save results to file
  python run_torch_cuda_triton_bench_lite.py --output results.json
        """
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
        default=[4, 5, 6, 7],
        help="GPU device IDs, default: [4, 5, 6, 7]"
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
