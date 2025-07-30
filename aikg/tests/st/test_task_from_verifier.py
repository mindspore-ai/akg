import pytest
from collections import defaultdict
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import get_benchmark_name, get_benchmark_task, add_op_prefix
from ai_kernel_generator.config.config_validator import load_config


@pytest.mark.asyncio
@pytest.mark.parametrize("framework,dsl,backend,arch", [
    ("numpy", "swft", "ascend", "ascend310p3")
])
async def test_parallel_task_from_verifier(framework, dsl, backend, arch):
    task_pool = TaskPool()
    device_pool = DevicePool([4, 5])
    config = load_config(dsl)
    benchmark_name = get_benchmark_name([23], framework=framework)

    for i in range(len(benchmark_name)):
        task_desc = get_benchmark_task(benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i])

        # 读取实现代码
        # Extract the core name from aikg_ prefixed op_name
        core_name = op_name[5:]  # Remove "aikg_" prefix
        kernel_path = f"./database/{dsl}/{arch}/{core_name}/aigen/{core_name}_{dsl}.py"
        with open(kernel_path, "r", encoding="utf-8") as f:
            kernel_code = f.read()

        aul_path = f"./database/{dsl}/{arch}/{core_name}/aigen/{core_name}_aul.py"
        with open(aul_path, "r", encoding="utf-8") as f:
            designer_code = f.read()

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
            workflow="verifier_only_workflow"
        )

        # Add initial codes to task_info
        init_task_info = {
            "designer_code": designer_code,
            "coder_code": kernel_code
        }

        task_pool.create_task(task.run, init_task_info)

    await task_pool.wait_all()
