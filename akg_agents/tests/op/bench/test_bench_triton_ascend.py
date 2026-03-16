import os
import pytest
from pathlib import Path
from collections import defaultdict
from akg_agents.core.async_pool.task_pool import TaskPool

from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask
from akg_agents.core.worker.manager import register_local_worker
from ..utils import (
    get_kernelbench_op_name, get_multikernelbench_op_name,
    get_kernelbench_task_desc, get_multikernelbench_task_desc,
    get_akg_kernels_bench_op_name, get_akg_kernels_bench_task_desc,
    get_evokernel_mhc_op_name, get_evokernel_task_desc,
    add_op_prefix, generate_beautiful_test_report, get_device_id
)
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task

os.environ['AKG_AGENTS_DATA_COLLECT'] = 'on'
os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'
device_id = get_device_id()


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_torch_triton_ascend910b4():
    """测试 KernelBench - PyTorch Triton Ascend910B4"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    # device_pool = DevicePool([device_id])  # 旧写法
    # or load_config("/your-path-to-config/xxx_config.yaml")
    config = load_config(dsl=dsl, backend=backend)

    check_env_for_task(framework, backend, dsl, config)

    # 新写法：注册 LocalWorker
    await register_local_worker([device_id], backend=backend, arch=arch)

    # KernelBench: 按序号读取
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19, ], framework=framework)

    if benchmark_name is None:
        raise RuntimeError("在 KernelBench 中未找到指定序号的任务文件，请检查 task_index_list 参数是否正确")

    for i in range(len(benchmark_name)):
        task_desc = get_kernelbench_task_desc(
            benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i], benchmark=benchmark)

        task = AIKGTask(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
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
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_akg_kernels_bench_torch_triton_ascend910b4():
    """测试 AIKGBench - PyTorch Triton Ascend910B4"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    benchmark = "AIKGBench"
    category = "dynamic"
    subcategory = "elemwise"

    task_pool = TaskPool()
    # device_pool = DevicePool([device_id])  # 旧写法
    # or load_config("/your-path-to-config/xxx_config.yaml")
    config = load_config(dsl=dsl, backend=backend)

    check_env_for_task(framework, backend, dsl, config)

    # 新写法：注册 LocalWorker
    await register_local_worker([device_id], backend=backend, arch=arch)

    # AIKGBench: 按分类读取，可以指定具体的 op_name 来获取单个 case
    benchmark_name = get_akg_kernels_bench_op_name(
        op_name="elemwise_add_001_var",
        category=category,
        subcategory=subcategory,
        framework=framework
    )
    # # 按类别获取一个类别所有的op
    # benchmark_name = get_akg_kernels_bench_op_name(
    #     category=category,
    #     subcategory=subcategory,
    #     framework=framework
    # )

    if benchmark_name is None:
        raise RuntimeError("在 KernelBench 中未找到指定序号的任务文件，请检查 task_index_list 参数是否正确")

    for i in range(len(benchmark_name)):
        task_desc = get_akg_kernels_bench_task_desc(
            benchmark_name[i], category=category, framework=framework)
        op_name = add_op_prefix(benchmark_name[i], benchmark=benchmark)

        task = AIKGTask(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
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
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_evokernel_mhc_torch_triton_ascend910b4():
    """测试 EvoKernel MHC - PyTorch Triton Ascend910B4"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    benchmark = "EvoKernel"
    category = "MHC"

    task_pool = TaskPool()
    config = load_config(dsl=dsl, backend=backend)

    check_env_for_task(framework, backend, dsl, config)

    # 新写法：注册 LocalWorker
    await register_local_worker([device_id], backend=backend, arch=arch)

    # EvoKernel: 按名称读取 (使用简单的 StreamWrite 算子)
    benchmark_name = get_evokernel_mhc_op_name(op_name="05_StreamWrite")

    if benchmark_name is None:
        raise RuntimeError(f"在 EvoKernel {category} 中未找到指定的操作")

    for i in range(len(benchmark_name)):
        task_desc = get_evokernel_task_desc(
            benchmark_name[i], category=category)
        op_name = add_op_prefix(f"{category}_{benchmark_name[i]}", benchmark=benchmark)

        task = AIKGTask(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            framework=framework,
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )
