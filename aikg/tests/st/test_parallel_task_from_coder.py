import pytest
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import get_benchmark_name, get_benchmark_task, add_op_prefix
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
    benchmark_name = get_benchmark_name([19, ], framework=framework)

    for i in range(len(benchmark_name)):
        task_desc = get_benchmark_task(benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i])

        # 读取实现代码
        # Extract the core name from aikg_ prefixed op_name
        core_name = op_name[5:]  # Remove "aikg_" prefix
        aul_path = f"./database/{impl_type}/{arch}/{core_name}/aigen/{core_name}_aul.py"
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
