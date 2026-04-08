import os
import asyncio
import time
from pathlib import Path
from typing import List, Tuple
import argparse

from akg_agents.op.config.config_validator import load_config
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.environment_check import check_env_for_task

os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'

def discover_sol_cases(level="L1") -> List[Tuple[str, Path]]:
    """
    Discover all SOL cases in the specified level directory.
    """
    benchmark_dir = Path(__file__).parent.parent.parent.parent / "thirdparty" / "sol-execbench" / "data" / "benchmark" / level
    
    if not benchmark_dir.exists():
        print(f"Warning: SOL Benchmark directory not found: {benchmark_dir}")
        print("Please make sure you have run 'bash download.sh --with_sol_execbench' first.")
        return []
    
    sol_cases = []
    for case_dir in sorted(benchmark_dir.iterdir()):
        if case_dir.is_dir():
            sol_cases.append((case_dir.name, case_dir))
            
    return sol_cases

async def run_sol_kernel(
    kernel_name: str,
    case_dir: Path,
    task_id: str,
    config: dict,
    arch: str
) -> Tuple[str, bool, float]:
    start_time = time.time()
    
    try:
        # Read definition and reference to construct task_desc
        with open(case_dir / "definition.json", "r") as f:
            def_json = f.read()
        with open(case_dir / "reference.py", "r") as f:
            ref_py = f.read()
            
        task_desc = f"请实现一个 Triton CUDA 算子。\n\n## definition.json\n```json\n{def_json}\n```\n\n## reference.py\n```python\n{ref_py}\n```\n\n注意：请使用 Triton 编写 kernel，并将其封装在 ModelNew 类的 forward 方法中。"
        
        # Update config with the specific case dir
        case_config = config.copy()
        case_config["sol_problem_dir"] = str(case_dir)
        
        task_pool = TaskPool()
        
        task = LangGraphTask(
            op_name=kernel_name,
            task_desc=task_desc,
            task_id=task_id,
            dsl="triton_cuda",
            backend="cuda",
            arch=arch,
            config=case_config,
            framework="torch",
            workflow="coder_only_workflow",
            bench_type="sol"
        )
        
        task_pool.create_task(task.run)
        results = await task_pool.wait_all()
        
        for op_name, result, _ in results:
            elapsed_time = time.time() - start_time
            return (kernel_name, result, elapsed_time)
        
        elapsed_time = time.time() - start_time
        return (kernel_name, False, elapsed_time)
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error running {kernel_name}: {str(e)}")
        return (kernel_name, False, elapsed_time)

async def run_batch_sol_tests(args):
    print("=" * 80)
    print(f"Batch SOL-ExecBench Kernel Testing on GPU (Triton CUDA) - Level: {args.level}")
    print("=" * 80)
    print()
    
    print(f"Discovering SOL cases in {args.level}...")
    sol_cases = discover_sol_cases(args.level)
    
    if not sol_cases:
        return
        
    if args.limit > 0 and len(sol_cases) > args.limit:
        print(f"Found {len(sol_cases)} cases, limiting to first {args.limit} as requested.")
        sol_cases = sol_cases[:args.limit]
    else:
        print(f"Found {len(sol_cases)} cases.")
    
    print(f"Registering local worker (CUDA {args.arch})...")
    await register_local_worker([args.device_id], backend="cuda", arch=args.arch)
    print("✓ Worker registered")
    print()
    
    print("Loading configuration...")
    config = load_config(config_path="./python/akg_agents/op/config/triton_cuda_coderonly_config.yaml")
    config["verify_timeout"] = args.timeout
    check_env_for_task("torch", "cuda", "triton_cuda", config)
    print("✓ Configuration loaded")
    print()
    
    print("=" * 80)
    print("Running Tests")
    print("=" * 80)
    print()
    
    results = []
    for i, (kernel_name, case_dir) in enumerate(sol_cases, 1):
        print(f"[{i}/{len(sol_cases)}] Testing: {kernel_name}")
        print("-" * 80)
        
        try:
            task_id = f"sol_batch_{args.level}_{i}"
            _, success, elapsed_time = await run_sol_kernel(
                kernel_name=kernel_name,
                case_dir=case_dir,
                task_id=task_id,
                config=config,
                arch=args.arch
            )
            
            results.append((kernel_name, success, elapsed_time))
            
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"Status: {status} (took {elapsed_time:.2f}s)")
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            results.append((kernel_name, False, 0.0))
        
        print()
        
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print()
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    total_time = sum(elapsed for _, _, elapsed in results)
    
    print(f"Total:  {len(results)} cases")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print(f"Total time: {total_time:.2f}s")
    print()
    
    print("Detailed Results:")
    print("-" * 80)
    print(f"{'No.':<5} {'Kernel Name':<40} {'Status':<10} {'Time (s)':<10}")
    print("-" * 80)
    
    for i, (kernel_name, success, elapsed_time) in enumerate(results, 1):
        status = "PASSED" if success else "FAILED"
        print(f"{i:<5} {kernel_name[:38]:<40} {status:<10} {elapsed_time:>8.2f}")
    
    print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch run SOL-ExecBench cases with Triton CUDA generation")
    parser.add_argument("--level", type=str, default="L1", choices=["L1", "L2", "Quant", "FlashInfer-Bench"], help="SOL Bench level")
    parser.add_argument("--limit", type=int, default=3, help="Limit the number of cases to run (0 for all)")
    parser.add_argument("--arch", type=str, default="a100", help="Architecture type")
    parser.add_argument("--device-id", type=int, default=0, help="Device ID to use")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per case in seconds")
    
    args = parser.parse_args()
    asyncio.run(run_batch_sol_tests(args))
