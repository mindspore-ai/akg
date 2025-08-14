import os
import pytest
from collections import defaultdict
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import (
    get_kernelbench_op_name, get_multikernelbench_op_name,
    get_kernelbench_task_desc, get_multikernelbench_task_desc, add_op_prefix
)
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task

os.environ['AIKG_DATA_COLLECT'] = 'on'


@pytest.mark.level2
@pytest.mark.mindspore
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_mindspore_triton_ascend910b4():
    """测试 KernelBench - MindSpore Triton Ascend910B4"""
    framework = "mindspore"
    dsl = "triton"
    backend = "ascend"
    arch = "ascend910b4"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    device_pool = DevicePool([1])
    # or load_config("/your-path-to-config/xxx_config.yaml")
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    # KernelBench: 按序号读取
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19, ], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{benchmark}' 不支持")

    result_dict = defaultdict(int)

    for i in range(len(benchmark_name)):
        task_desc = get_kernelbench_task_desc(
            benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i], benchmark=benchmark)

        task = Task(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            device_pool=device_pool,
            framework=framework,
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()
    for op_name, result, _ in results:
        result_dict[op_name] += result

    pass_num = 0
    for result in result_dict.values():
        if result:
            pass_num += 1
    pass_rate = pass_num / len(result_dict)

    print('-' * 60)
    print(f"Benchmark: {benchmark}")
    print(f"Category: all")
    print(result_dict)
    print("PASS RATE: ", pass_rate)
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Benchmark: {benchmark}\n")
        f.write(f"Category: all\n")
        f.write(f"结果字典: {dict(result_dict)}\n")
        f.write(f"通过率: {pass_rate}\n")
    print("结果已保存到 test_results.txt")
    print('-' * 60)


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_torch_triton_ascend910b4():
    """测试 KernelBench - PyTorch Triton Ascend910B4"""
    framework = "torch"
    dsl = "triton"
    backend = "ascend"
    arch = "ascend910b4"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    device_pool = DevicePool([1])
    # or load_config("/your-path-to-config/xxx_config.yaml")
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    # KernelBench: 按序号读取
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19, ], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{benchmark}' 不支持")

    result_dict = defaultdict(int)

    for i in range(len(benchmark_name)):
        task_desc = get_kernelbench_task_desc(
            benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i], benchmark=benchmark)

        task = Task(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            device_pool=device_pool,
            framework=framework,
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()
    for op_name, result, _ in results:
        result_dict[op_name] += result

    pass_num = 0
    for result in result_dict.values():
        if result:
            pass_num += 1
    pass_rate = pass_num / len(result_dict)

    print('-' * 60)
    print(f"Benchmark: {benchmark}")
    print(f"Category: all")
    print(result_dict)
    print("PASS RATE: ", pass_rate)
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Benchmark: {benchmark}\n")
        f.write(f"Category: all\n")
        f.write(f"结果字典: {dict(result_dict)}\n")
        f.write(f"通过率: {pass_rate}\n")
    print("结果已保存到 test_results.txt")
    print('-' * 60)


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_multikernelbench_activation_torch_triton_ascend910b4():
    """测试 MultiKernelBench - PyTorch Triton Ascend910B4 (激活函数分类)"""
    framework = "torch"
    dsl = "triton"
    backend = "ascend"
    arch = "ascend910b4"
    benchmark = "MultiKernelBench"
    category = "activation"

    task_pool = TaskPool()
    device_pool = DevicePool([1])
    # or load_config("/your-path-to-config/xxx_config.yaml")
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    # MultiKernelBench: 按分类读取，可以指定具体的 op_name 来获取单个 case
    benchmark_name = get_multikernelbench_op_name(
        op_name="relu", category=category, framework=framework)
    # 按类别获取一个类别所有的op
    # benchmark_name = get_multikernelbench_op_name(
    #     category=category, framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{benchmark}' 不支持")

    result_dict = defaultdict(int)

    for i in range(len(benchmark_name)):
        task_desc = get_multikernelbench_task_desc(
            benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i], benchmark=benchmark)

        task = Task(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            device_pool=device_pool,
            framework=framework,
            workflow="coder_only_workflow"
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()
    for op_name, result, _ in results:
        result_dict[op_name] += result

    pass_num = 0
    for result in result_dict.values():
        if result:
            pass_num += 1
    pass_rate = pass_num / len(result_dict)

    print('-' * 60)
    print(f"Benchmark: {benchmark}")
    print(f"Category: {category}")
    print(result_dict)
    print("PASS RATE: ", pass_rate)
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Benchmark: {benchmark}\n")
        f.write(f"Category: {category}\n")
        f.write(f"结果字典: {dict(result_dict)}\n")
        f.write(f"通过率: {pass_rate}\n")
    print("结果已保存到 test_results.txt")
    print('-' * 60)
