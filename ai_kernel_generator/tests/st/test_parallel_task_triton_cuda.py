import pytest
from collections import defaultdict
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import get_benchmark_name, get_benchmark_task, remove_alnum_from_benchmark_name
from ai_kernel_generator.config.config_validator import load_config


@pytest.mark.asyncio
@pytest.mark.parametrize("framework,impl_type,backend,arch", [
    ("torch", "triton", "cuda", "a100")
])
async def test_parallel_task_triton_a100(framework, impl_type, backend, arch):
    task_pool = TaskPool()
    device_pool = DevicePool([1, 2])
    config = load_config()  # or load_config("/your-path-to-config/xxx_config.yaml")
    benchmark_name = get_benchmark_name([19], framework=framework)

    result_dict = defaultdict(int)

    for i in range(len(benchmark_name)):
        task_desc = get_benchmark_task(benchmark_name[i], framework=framework)
        op_name = remove_alnum_from_benchmark_name(benchmark_name[i])

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
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()
    for op_name, result in results:
        result_dict[op_name] += result

    pass_num = 0
    for result in result_dict.values():
        if result:
            pass_num += 1
    pass_rate = pass_num / len(result_dict)

    print('-' * 60)
    print(result_dict)
    print("PASS RATE: ", pass_rate)
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"结果字典: {dict(result_dict)}\n")
        f.write(f"通过率: {pass_rate}\n")
    print("结果已保存到 test_results.txt")
    print('-' * 60)