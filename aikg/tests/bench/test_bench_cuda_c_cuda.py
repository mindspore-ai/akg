import os
import pytest
from pathlib import Path
from collections import defaultdict
from ai_kernel_generator.core.async_pool.task_pool import TaskPool

# 自动选择 Task 实现：优先使用 LangGraphTask，否则使用原 Task
try:
    import langgraph
    from ai_kernel_generator.core.langgraph_task import LangGraphTask as AIKGTask
    _USE_LANGGRAPH = True
except ImportError:
    from ai_kernel_generator.core.task import Task as AIKGTask
    _USE_LANGGRAPH = False
from ai_kernel_generator.core.worker.manager import register_local_worker
from ..utils import (
    get_kernelbench_op_name, get_multikernelbench_op_name,
    get_kernelbench_task_desc, get_multikernelbench_task_desc, add_op_prefix,
    generate_beautiful_test_report, get_device_id
)
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task

os.environ['AIKG_DATA_COLLECT'] = 'on'
device_id = get_device_id()


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.cuda_c
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_torch_cuda_a100():
    """测试 KernelBench - PyTorch CUDA C"""
    framework = "torch"
    dsl = "cuda_c"
    backend = "cuda"
    arch = "a100"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    # device_pool = DevicePool([device_id])  # 旧写法
    # or load_config("/your-path-to-config/xxx_config.yaml")
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_cuda_c_coderonly_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    # 新写法：注册 LocalWorker
    await register_local_worker([device_id], backend=backend, arch=arch)

    # KernelBench: 按序号读取
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19, ], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{benchmark}' 不支持")

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
