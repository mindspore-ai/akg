import pytest
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import (
    get_kernelbench_op_name, get_kernelbench_task_desc, add_op_prefix, process_task_results
)
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.cuda
@pytest.mark.a100
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_parallel_task_triton_cuda():
    framework = "torch"
    dsl = "triton"
    backend = "cuda"
    arch = "a100"
    task_pool = TaskPool(1)
    device_pool = DevicePool([1])
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    benchmark_name = get_kernelbench_op_name([19], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{framework}' 不支持")

    for i in range(len(benchmark_name)):
        task_desc = get_kernelbench_task_desc(benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i], benchmark="KernelBench")

        task = Task(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            dsl=dsl,
            backend=backend,
            arch=arch,
            config=config,
            device_pool=device_pool,
            framework=framework,
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()

    # 使用通用的结果处理函数
    success = process_task_results(results, print_summary=True)
    assert success, "存在测试case失败"
