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
Batch testing script for all attention kernels on CPU.

This script automatically discovers and tests all attention implementations
in the benchmark directory, generating optimized C++ code for each variant.
"""

from akg_agents.op.config.config_validator import load_config
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.environment_check import check_env_for_task
import asyncio
import os
import importlib.util
from pathlib import Path
from typing import List, Tuple
import time

# Enable stream output
os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'


def discover_attention_kernels() -> List[Tuple[str, Path]]:
    """
    Discover all attention kernel files in the benchmark directory.
    
    Returns:
        List of tuples (kernel_name, file_path)
    """
    benchmark_dir = Path(__file__).parent.parent.parent.parent / "benchmark" / "akg_kernels_bench" / "static_shape" / "attention"
    
    if not benchmark_dir.exists():
        print(f"Warning: Benchmark directory not found: {benchmark_dir}")
        return []
    
    attention_files = []
    for py_file in sorted(benchmark_dir.glob("attention_*.py")):
        kernel_name = py_file.stem  # Remove .py extension
        attention_files.append((kernel_name, py_file))
    
    return attention_files


def load_kernel_module(file_path: Path):
    """
    Dynamically load a Python module from file path.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Loaded module object
    """
    spec = importlib.util.spec_from_file_location("kernel_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_kernel_file(file_path: Path) -> str:
    """
    Read the entire kernel file as task description.
    
    Args:
        file_path: Path to the kernel file
        
    Returns:
        File contents as string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


async def run_attention_kernel(
    kernel_name: str,
    task_desc: str,
    task_id: str,
    config: dict
) -> Tuple[str, bool, float]:
    """
    Run a single attention kernel test.
    
    Args:
        kernel_name: Name of the kernel
        task_desc: Task description (kernel code)
        task_id: Unique task ID
        config: Configuration dictionary
        
    Returns:
        Tuple of (kernel_name, success, elapsed_time)
    """
    start_time = time.time()
    
    try:
        task_pool = TaskPool()
        
        task = LangGraphTask(
            op_name=kernel_name,
            task_desc=task_desc,
            task_id=task_id,
            dsl="cpp",
            backend="cpu",
            arch="x86_64",
            config=config,
            framework="torch",
            workflow="coder_only_workflow"
        )
        
        task_pool.create_task(task.run)
        results = await task_pool.wait_all()
        
        # Extract result
        for op_name, result, _ in results:
            elapsed_time = time.time() - start_time
            return (kernel_name, result, elapsed_time)
        
        # No results returned
        elapsed_time = time.time() - start_time
        return (kernel_name, False, elapsed_time)
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"Error running {kernel_name}: {str(e)}")
        return (kernel_name, False, elapsed_time)


async def run_batch_attention_tests():
    """
    Main function to run batch attention kernel tests.
    """
    print("=" * 80)
    print("Batch Attention Kernel Testing on CPU")
    print("=" * 80)
    print()
    
    # Discover all attention kernels
    print("Discovering attention kernels...")
    attention_kernels = discover_attention_kernels()
    
    if not attention_kernels:
        print("No attention kernels found!")
        return
    
    print(f"Found {len(attention_kernels)} attention kernels:")
    for i, (name, path) in enumerate(attention_kernels, 1):
        print(f"  {i:2d}. {name}")
    print()
    
    # Register LocalWorker (CPU x86_64)
    print("Registering local worker (CPU x86_64)...")
    await register_local_worker([0], backend="cpu", arch="x86_64")
    print("✓ Worker registered")
    print()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config("cpp")
    check_env_for_task("torch", "cpu", "cpp", config)
    print("✓ Configuration loaded")
    print()
    
    # Run tests
    print("=" * 80)
    print("Running Tests")
    print("=" * 80)
    print()
    
    results = []
    for i, (kernel_name, file_path) in enumerate(attention_kernels, 1):
        print(f"[{i}/{len(attention_kernels)}] Testing: {kernel_name}")
        print("-" * 80)
        
        # Read kernel file as task description
        try:
            task_desc = read_kernel_file(file_path)
            
            # Run the test
            kernel_name_result, success, elapsed_time = await run_attention_kernel(
                kernel_name=kernel_name,
                task_desc=task_desc,
                task_id=str(i),
                config=config
            )
            
            results.append((kernel_name, success, elapsed_time))
            
            status = "✓ PASSED" if success else "✗ FAILED"
            print(f"Status: {status} (took {elapsed_time:.2f}s)")
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            results.append((kernel_name, False, 0.0))
        
        print()
    
    # Print summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print()
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    total_time = sum(elapsed for _, _, elapsed in results)
    
    print(f"Total:  {len(results)} kernels")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print(f"Total time: {total_time:.2f}s")
    print()
    
    # Detailed results
    print("Detailed Results:")
    print("-" * 80)
    print(f"{'No.':<5} {'Kernel Name':<40} {'Status':<10} {'Time (s)':<10}")
    print("-" * 80)
    
    for i, (kernel_name, success, elapsed_time) in enumerate(results, 1):
        status = "PASSED" if success else "FAILED"
        print(f"{i:<5} {kernel_name:<40} {status:<10} {elapsed_time:>8.2f}")
    
    print("-" * 80)
    print()
    
    # List failed tests
    if failed > 0:
        print("Failed Tests:")
        for kernel_name, success, _ in results:
            if not success:
                print(f"  ✗ {kernel_name}")
        print()
    
    print("=" * 80)
    print(f"Batch testing completed! ({passed}/{len(results)} passed)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_batch_attention_tests())
