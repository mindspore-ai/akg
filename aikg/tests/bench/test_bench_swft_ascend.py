import os
import pytest
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import (
    get_kernelbench_op_name, get_kernelbench_task_desc, add_op_prefix,
    generate_beautiful_test_report
)
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task

os.environ['AIKG_DATA_COLLECT'] = 'on'


@pytest.mark.level2
@pytest.mark.numpy
@pytest.mark.swft
@pytest.mark.ascend
@pytest.mark.ascend310p3
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_bench_swft_ascend():
    framework = "numpy"
    dsl = "swft"
    backend = "ascend"
    arch = "ascend310p3"
    task_pool = TaskPool()
    device_pool = DevicePool([1, 2])
    config = load_config(dsl)

    check_env_for_task(framework, backend, dsl, config)

    benchmark_name = get_kernelbench_op_name([19, 20], framework=framework)

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

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )
