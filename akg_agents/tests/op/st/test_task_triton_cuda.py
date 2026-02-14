import pytest
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.core.worker.manager import register_local_worker
from ..utils import (
    get_kernelbench_op_name, get_kernelbench_task_desc, add_op_prefix,
    process_task_results, get_device_id
)
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task

from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask


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
    config = load_config(config_path="./python/akg_agents/op/config/triton_cuda_coderonly_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    # 新写法：注册 LocalWorker
    await register_local_worker([device_id], backend=backend, arch=arch)

    benchmark_name = get_kernelbench_op_name([19], framework=framework)

    if benchmark_name is None:
        raise RuntimeError("在 KernelBench 中未找到指定序号的任务文件，请检查 task_index_list 参数是否正确")

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
