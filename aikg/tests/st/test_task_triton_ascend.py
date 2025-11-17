import pytest
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import (
    get_kernelbench_op_name, get_multikernelbench_op_name,
    get_kernelbench_task_desc, get_multikernelbench_task_desc, add_op_prefix,
    process_task_results, get_device_id
)
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task


device_id = get_device_id()


@pytest.mark.level0
@pytest.mark.mindspore
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_mindspore_triton_ascend910b4():
    """测试 KernelBench - MindSpore Triton Ascend910B4"""
    framework = "mindspore"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    device_pool = DevicePool([device_id])
    # or load_config("/your-path-to-config/xxx_config.yaml")
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    # KernelBench: 按序号读取
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19, ], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{benchmark}' 不支持")

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

    # 使用通用的结果处理函数
    success = process_task_results(results, print_summary=True)
    assert success, "存在测试case失败"


@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_torch_triton_ascend910b4():
    """测试 KernelBench - PyTorch Triton Ascend910B4"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    device_pool = DevicePool([device_id])
    # or load_config("/your-path-to-config/xxx_config.yaml")
    config = load_config(config_path="./python/ai_kernel_generator/config/vllm_triton_coderonly_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    # KernelBench: 按序号读取
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19, ], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{benchmark}' 不支持")

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

    # 使用通用的结果处理函数
    success = process_task_results(results, print_summary=True)
    assert success, "存在测试case失败"


@pytest.mark.level1
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_multikernelbench_activation_torch_triton_ascend910b4():
    """测试 MultiKernelBench - PyTorch Triton Ascend910B4 (激活函数分类)"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    benchmark = "MultiKernelBench"
    category = "activation"

    task_pool = TaskPool()
    device_pool = DevicePool([device_id])
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

    # 使用通用的结果处理函数
    success = process_task_results(results, print_summary=True)
    assert success, "存在测试case失败"
