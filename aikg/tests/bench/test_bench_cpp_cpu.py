import os
import pytest
from pathlib import Path
from collections import defaultdict
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import (
    get_kernelbench_op_name, get_multikernelbench_op_name,
    get_kernelbench_task_desc, get_multikernelbench_task_desc, add_op_prefix,
    generate_beautiful_test_report
)
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task

os.environ['AIKG_DATA_COLLECT'] = 'on'


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.cpp
@pytest.mark.cpu
@pytest.mark.x86_64
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_torch_cpu_x86_64():
    """测试 KernelBench - PyTorch C++ CPU"""
    framework = "torch"
    dsl = "cpp"
    backend = "cpu"
    arch = "x86_64"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    device_pool = DevicePool([1])
    # or load_config("/your-path-to-config/xxx_config.yaml")
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_cpp_coderonly_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    # KernelBench: 按序号读取
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19, ], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{benchmark}' 不支持")

    for i in range(len(benchmark_name)):
        task_desc = get_kernelbench_task_desc(
            benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i], benchmark=benchmark)

        task = Task(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            device_pool=device_pool,
            framework=framework,
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.cpp
@pytest.mark.cpu
@pytest.mark.aarch64
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_torch_cpu_aarch64():
    """测试 KernelBench - PyTorch C++ CPU"""
    framework = "torch"
    dsl = "cpp"
    backend = "cpu"
    arch = "aarch64"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    device_pool = DevicePool([1])
    # or load_config("/your-path-to-config/xxx_config.yaml")
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_cpp_coderonly_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    # KernelBench: 按序号读取
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19, ], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{benchmark}' 不支持")

    for i in range(len(benchmark_name)):
        task_desc = get_kernelbench_task_desc(
            benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i], benchmark=benchmark)

        task = Task(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            device_pool=device_pool,
            framework=framework,
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )
