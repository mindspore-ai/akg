import pytest
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.worker.manager import register_local_worker
from ..utils import (
    get_kernelbench_op_name, get_kernelbench_task_desc, add_op_prefix,
    process_task_results, get_device_id
)
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task

# 自动选择 Task 实现：优先使用 LangGraphTask，否则使用原 Task
try:
    import langgraph
    from ai_kernel_generator.core.langgraph_task import LangGraphTask as AIKGTask
    _USE_LANGGRAPH = True
except ImportError:
    from ai_kernel_generator.core.task import Task as AIKGTask
    _USE_LANGGRAPH = False


device_id = get_device_id()


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_parallel_task_triton_cuda():
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    task_pool = TaskPool(1)
    # device_pool = DevicePool([device_id])  # 旧写法
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_cuda_coderonly_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    # 新写法：注册 LocalWorker
    await register_local_worker([device_id], backend=backend, arch=arch)

    benchmark_name = get_kernelbench_op_name([19], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{framework}' 不支持")

    for i in range(len(benchmark_name)):
        task_desc = get_kernelbench_task_desc(benchmark_name[i], framework=framework)
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

    # 使用通用的结果处理函数
    success = process_task_results(results, print_summary=True)
    assert success, "存在测试case失败"
