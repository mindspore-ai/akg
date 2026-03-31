import os
import pytest
from akg_agents.core.async_pool.task_pool import TaskPool

from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask
from akg_agents.core.worker.manager import register_local_worker, register_remote_worker
from ..utils import (
    get_kernelbench_op_name, get_kernelbench_task_desc, add_op_prefix,
    get_evokernel_attention_op_name, get_evokernel_task_desc,
    generate_beautiful_test_report, get_device_id
)
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task

os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'
device_id = get_device_id()


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_bench_triton_cuda():
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    level = "level1"  # 可以设置为 "level1", "level2", "level3" 等
    task_pool = TaskPool(1)
    # device_pool = DevicePool([device_id])  # 旧写法
    config = load_config(config_path="./python/akg_agents/op/config/triton_cuda_coderonly_config.yaml")

    # 根据环境变量判断使用哪种 worker
    worker_url = os.getenv("AKG_AGENTS_WORKER_URL")
    use_remote_worker = worker_url is not None

    # Remote 模式跳过硬件检查
    check_env_for_task(framework, backend, dsl, config, is_remote=use_remote_worker)

    # 根据环境变量注册对应的 Worker
    if use_remote_worker:
        await register_remote_worker(backend=backend, arch=arch, worker_url=worker_url)
    else:
        await register_local_worker([device_id], backend=backend, arch=arch)

    benchmark_name = get_kernelbench_op_name([19], framework=framework, level=level)

    if benchmark_name is None:
        raise RuntimeError("在 KernelBench 中未找到指定序号的任务文件，请检查 task_index_list 参数是否正确")

    for i in range(len(benchmark_name)):
        task_desc = get_kernelbench_task_desc(benchmark_name[i], framework=framework, level=level)
        op_name = add_op_prefix(benchmark_name[i], benchmark="KernelBench")

        task = AIKGTask(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            dsl=dsl,
            backend=backend,
            arch=arch,
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
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_bench_triton_cuda_level_2():
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    level = "level2"  # 可以设置为 "level1", "level2", "level3" 等
    task_pool = TaskPool(1)
    # device_pool = DevicePool([device_id])  # 旧写法
    config = load_config(config_path="./python/akg_agents/op/config/triton_cuda_coderonly_config.yaml")

    # 根据环境变量判断使用哪种 worker
    worker_url = os.getenv("AKG_AGENTS_WORKER_URL")
    use_remote_worker = worker_url is not None

    # Remote 模式跳过硬件检查
    check_env_for_task(framework, backend, dsl, config, is_remote=use_remote_worker)

    # 根据环境变量注册对应的 Worker
    if use_remote_worker:
        await register_remote_worker(backend=backend, arch=arch, worker_url=worker_url)
    else:
        await register_local_worker([device_id], backend=backend, arch=arch)

    benchmark_name = get_kernelbench_op_name([63], framework=framework, level=level)

    if benchmark_name is None:
        raise RuntimeError("在 KernelBench 中未找到指定序号的任务文件，请检查 task_index_list 参数是否正确")

    for i in range(len(benchmark_name)):
        task_desc = get_kernelbench_task_desc(benchmark_name[i], framework=framework, level=level)
        op_name = add_op_prefix(benchmark_name[i], benchmark="KernelBench")

        task = AIKGTask(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            dsl=dsl,
            backend=backend,
            arch=arch,
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
@pytest.mark.trtriton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_evokernel_attention_torch_triton_cuda_a100():
    """测试 EvoKernel Attention - PyTorch Triton CUDA"""
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    benchmark = "EvoKernel"
    category = "Attention"

    task_pool = TaskPool(1)
    config = load_config(config_path="./python/akg_agents/op/config/triton_cuda_coderonly_config.yaml")

    # 根据环境变量判断使用哪种 worker
    worker_url = os.getenv("AKG_AGENTS_WORKER_URL")
    use_remote_worker = worker_url is not None

    # Remote 模式跳过硬件检查
    check_env_for_task(framework, backend, dsl, config, is_remote=use_remote_worker)

    # 根据环境变量注册对应的 Worker
    if use_remote_worker:
        await register_remote_worker(backend=backend, arch=arch, worker_url=worker_url)
    else:
        await register_local_worker([device_id], backend=backend, arch=arch)

    # EvoKernel: 按名称读取 (使用简单的 ScaledDotProductAttention 算子)
    benchmark_name = get_evokernel_attention_op_name(op_name="1_ScaledDotProductAttention")

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
            dsl=dsl,
            backend=backend,
            arch=arch,
            config=config,
            framework=framework,
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )
