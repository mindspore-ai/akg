import pytest
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import get_benchmark_name, get_benchmark_task, remove_alnum_from_benchmark_name
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.utils import ActionType, ParsedCode


@pytest.mark.asyncio
@pytest.mark.parametrize("framework,impl_type,backend,arch", [
    ("torch", "swft", "ascend", "ascend310p3")
])
async def test_parallel_task_from_coder(framework, impl_type, backend, arch):
    task_pool = TaskPool()
    device_pool = DevicePool([1, 2])
    config = load_config()
    benchmark_name = get_benchmark_name([19, 20, 21, 22], framework=framework)

    for i in range(len(benchmark_name)):
        task_desc = get_benchmark_task(benchmark_name[i], framework=framework)
        op_name = remove_alnum_from_benchmark_name(benchmark_name[i])

        # 读取实现代码
        aul_path = f"./database/{impl_type}/{arch}/{op_name[:-3]}/aigen/{op_name[:-3]}_aul.py"
        with open(aul_path, "r", encoding="utf-8") as f:
            aul_code = f.read()

        task = Task(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            impl_type=impl_type,
            backend=backend,
            arch=arch,
            config=config,
            device_pool=device_pool,
            framework=framework
        )

        parsed_code = ParsedCode()
        parsed_code.aul_code = aul_code
        task_pool.create_task(task.run, ActionType.DO_CODER, parsed_code)

    await task_pool.wait_all()