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

from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.utils.environment_check import check_env_for_task
import asyncio
import os
import argparse
from pathlib import Path

os.environ['AIKG_STREAM_OUTPUT'] = 'on'


def check_file_has_required_components(file_path: str) -> bool:
    """
    检查文件是否包含必需的组件：
    - class Model(nn.Module)
    - get_inputs
    - get_init_inputs
    
    Args:
        file_path: 文件路径
        
    Returns:
        如果包含所有必需组件返回 True，否则返回 False
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查 class Model(nn.Module)，支持多种格式
        has_model_class = (
            'class Model(nn.Module)' in content or 
            'class Model(nn.Module):' in content or
            'class Model(nn.Module) :' in content
        )
        # 检查 get_inputs 函数
        has_get_inputs = 'def get_inputs' in content
        # 检查 get_init_inputs 函数
        has_get_init_inputs = 'def get_init_inputs' in content
        
        return has_model_class and has_get_inputs and has_get_init_inputs
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False


def get_py_files_from_dir(dir_path: str) -> list:
    """
    从目录中获取所有 .py 文件
    
    Args:
        dir_path: 目录路径
        
    Returns:
        .py 文件路径列表
    """
    py_files = []
    dir_path_obj = Path(dir_path)
    
    if not dir_path_obj.exists():
        print(f"Error: Directory {dir_path} does not exist")
        return py_files
    
    if not dir_path_obj.is_dir():
        print(f"Error: {dir_path} is not a directory")
        return py_files
    
    for file_path in dir_path_obj.rglob('*.py'):
        py_files.append(str(file_path))
    
    return sorted(py_files)


async def run_single_task(op_name: str, task_desc: str):
    """
    运行单个任务
    
    Args:
        op_name: 算子名称
        task_desc: 任务描述
    """
    task_pool = TaskPool()
    device_pool = DevicePool([0])
    config = load_config("triton_ascend", backend="ascend")
    # config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")

    check_env_for_task("torch", "ascend", "triton_ascend", config)

    task = Task(
        op_name=op_name,
        task_desc=task_desc,
        task_id="0",
        dsl="triton_ascend",
        backend="ascend",
        arch="ascend910b4",
        config=config,
        device_pool=device_pool,
        framework="torch",
        workflow="coder_only_workflow"
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
    for op_name, result, _ in results:
        if result:
            print(f"Task {op_name} passed")
        else:
            print(f"Task {op_name} failed")
        return result
    # 如果没有结果，返回 False
    return False


async def batch_run_torch_npu_triton(dir_path: str):
    """
    批量运行目录中的 py 文件
    
    Args:
        dir_path: 包含 py 文件的目录路径
    """
    # 获取所有 py 文件
    py_files = get_py_files_from_dir(dir_path)
    
    if not py_files:
        print(f"No Python files found in directory: {dir_path}")
        return
    
    # 过滤出包含必需组件的文件
    valid_files = []
    for file_path in py_files:
        if check_file_has_required_components(file_path):
            valid_files.append(file_path)
            print(f"✓ Found valid file: {file_path}")
        else:
            print(f"✗ Skipping file (missing required components): {file_path}")
    
    if not valid_files:
        print("No valid files found with required components (class Model(nn.Module), get_inputs, get_init_inputs)")
        return
    
    print(f"\nFound {len(valid_files)} valid file(s) to run\n")
    
    # 逐个运行任务并收集结果
    results_summary = []
    for idx, file_path in enumerate(valid_files, 1):
        print(f"\n{'='*80}")
        print(f"Running task {idx}/{len(valid_files)}: {file_path}")
        print(f"{'='*80}")
        
        # 读取文件内容作为 task_desc
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                task_desc = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            results_summary.append((file_path, False, str(e)))
            continue
        
        # 从文件名提取 op_name（去掉路径和 .py 扩展名）
        op_name = Path(file_path).stem
        
        # 运行任务
        try:
            result = await run_single_task(op_name, task_desc)
            results_summary.append((file_path, result, None))
        except Exception as e:
            print(f"Error running task: {e}")
            results_summary.append((file_path, False, str(e)))
    
    # 输出总结
    print(f"\n{'='*80}")
    print("Batch Run Summary")
    print(f"{'='*80}")
    
    passed_cases = [file_path for file_path, result, _ in results_summary if result]
    failed_cases = [file_path for file_path, result, _ in results_summary if not result]
    
    print(f"\nTotal: {len(results_summary)}")
    print(f"Passed: {len(passed_cases)}")
    print(f"Failed: {len(failed_cases)}")
    
    if passed_cases:
        print(f"\n✓ Passed cases ({len(passed_cases)}):")
        for case in passed_cases:
            print(f"  - {case}")
    
    if failed_cases:
        print(f"\n✗ Failed cases ({len(failed_cases)}):")
        for case in failed_cases:
            error = next((error for file_path, _, error in results_summary if file_path == case), None)
            print(f"  - {case}")
            if error:
                print(f"    Error: {error}")
    
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='批量运行目录中的 torch npu triton 任务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python batch_run_torch_npu_triton.py /path/to/directory
        """
    )
    parser.add_argument(
        'dir_path',
        type=str,
        help='包含 py 文件的目录路径'
    )
    
    args = parser.parse_args()
    
    asyncio.run(batch_run_torch_npu_triton(args.dir_path))


if __name__ == "__main__":
    main()

