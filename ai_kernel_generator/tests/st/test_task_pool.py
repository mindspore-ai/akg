import pytest
from collections import defaultdict
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import get_benchmark_name, get_benchmark_task, remove_alnum_from_benchmark_name
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.core.utils import ActionType, ParsedCode


@pytest.mark.asyncio
@pytest.mark.parametrize("framework,impl_type,backend,arch", [
    ("mindspore", "triton", "ascend", "ascend910b4"),
])
async def test_parallel_task_triton_ascend910b4(framework, impl_type, backend, arch):
    task_pool = TaskPool()
    device_pool = DevicePool(["1", "2"])
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
    print(result_dict)

    pass_num = 0
    for result in result_dict.values():
        if result:
            pass_num += 1
    pass_rate = pass_num / len(result_dict)
    print('***************************************')
    print(result_dict)
    print("PASS RATE: ", pass_rate)
    print('***************************************')

    # 保存结果到文件
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"结果字典: {dict(result_dict)}\n")
        f.write(f"通过率: {pass_rate}\n")

    print("结果已保存到 test_results.txt")


@pytest.mark.asyncio
@pytest.mark.parametrize("framework,impl_type,backend,arch", [
    ("numpy", "swft", "ascend", "ascend310p3"),
])
async def test_parallel_task_swft_ascend310p3(framework, impl_type, backend, arch):
    task_pool = TaskPool()
    device_pool = DevicePool(["1", "2"])
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
    print(result_dict)

    pass_num = 0
    for result in result_dict.values():
        if result:
            pass_num += 1
    pass_rate = pass_num / len(result_dict)
    print('***************************************')
    print(result_dict)
    print("PASS RATE: ", pass_rate)
    print('***************************************')

    # 保存结果到文件
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"结果字典: {dict(result_dict)}\n")
        f.write(f"通过率: {pass_rate}\n")

    print("结果已保存到 test_results.txt")


@pytest.mark.asyncio
@pytest.mark.parametrize("framework,impl_type,backend,arch", [
    ("torch", "triton", "cuda", "a100")
])
async def test_parallel_task_triton_a100(framework, impl_type, backend, arch):
    task_pool = TaskPool()
    device_pool = DevicePool(["1", "2"])
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
    print(result_dict)

    pass_num = 0
    for result in result_dict.values():
        if result:
            pass_num += 1
    pass_rate = pass_num / len(result_dict)
    print('***************************************')
    print(result_dict)
    print("PASS RATE: ", pass_rate)
    print('***************************************')

    # 保存结果到文件
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"结果字典: {dict(result_dict)}\n")
        f.write(f"通过率: {pass_rate}\n")

    print("结果已保存到 test_results.txt")


@pytest.mark.asyncio
@pytest.mark.parametrize("framework,impl_type,backend,arch", [
    ("torch", "swft", "ascend", "ascend310p3")
])
async def test_parallel_task_from_coder(framework, impl_type, backend, arch):
    task_pool = TaskPool()
    device_pool = DevicePool(["1", "2"])
    config = load_config()
    benchmark_name = get_benchmark_name([19, 20, 21, 22], framework=framework)

    for i in range(len(benchmark_name)):
        task_desc = get_benchmark_task(benchmark_name[i], framework=framework)
        op_name = remove_alnum_from_benchmark_name(benchmark_name[i])

        # 读取实现代码
        aul_path = f"./database/benchmark_to_aul/{benchmark_name[i]}.py"
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


@pytest.mark.asyncio
@pytest.mark.parametrize("framework,impl_type,backend,arch", [
    ("torch", "swft", "ascend", "ascend310p3"),
    # ("torch", "triton", "cuda", "a100")
])
async def test_parallel_task_from_verifier(framework, impl_type, backend, arch):
    task_pool = TaskPool()
    device_pool = DevicePool(["4", "5"])
    config = load_config()
    benchmark_name = get_benchmark_name([19], framework=framework) * 2

    for i in range(len(benchmark_name)):
        task_desc = get_benchmark_task(benchmark_name[i], framework=framework)
        op_name = remove_alnum_from_benchmark_name(benchmark_name[i])

        # 读取实现代码
        kernel_path = f"./tests/resources/triton/{op_name}_op/{op_name}_{impl_type}.py"
        with open(kernel_path, "r", encoding="utf-8") as f:
            kernel_code = f.read()

        aul_path = f"./database/benchmark_to_aul/{benchmark_name[i]}.py"
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
        if impl_type == "swft":
            parsed_code.swft_code = kernel_code
        elif impl_type == "triton":
            parsed_code.triton_code = kernel_code
        else:
            raise ValueError(f"Invalid implementation type: {impl_type}")

        task_pool.create_task(task.run, ActionType.DO_TESTER, parsed_code)

    await task_pool.wait_all()
