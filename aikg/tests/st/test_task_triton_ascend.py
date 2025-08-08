import pytest
from collections import defaultdict
from ai_kernel_generator.core.task import Task
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import get_benchmark_name, get_benchmark_task, add_op_prefix
from ai_kernel_generator.config.config_validator import load_config


@pytest.mark.asyncio
@pytest.mark.parametrize("framework,dsl,backend,arch,benchmark,category", [
    # ("mindspore", "triton", "ascend", "ascend910b4", "KernelBench", "all"),
    ("torch", "triton", "ascend", "ascend910b4", "KernelBench", "all"),
    # ("torch", "triton", "ascend", "ascend910b4", "multiKernelBench", "all"),
    # ("torch", "triton", "ascend", "ascend910b4", "multiKernelBench", "activation"),
])
async def test_parallel_task_triton_ascend910b4(framework, dsl, backend, arch, benchmark, category):
    task_pool = TaskPool()
    device_pool = DevicePool([1, 2])
    # or load_config("/your-path-to-config/xxx_config.yaml")
    config = load_config(dsl)

    # 根据 benchmark 类型选择调用方式
    if benchmark == "multiKernelBench":
        # MultiKernelBench: 按分类读取，支持指定 op_name 获取单个 case
        if category != "all":
            # 可以指定具体的 op_name 来获取单个 case
            benchmark_name = get_benchmark_name(
                op_name="relu", category=category, framework=framework, benchmark=benchmark)
            # 按类别获取一个类别所有的op
            # benchmark_name = get_benchmark_name(
            #     category=category, framework=framework, benchmark=benchmark)
        else:
            benchmark_name = get_benchmark_name(
                category=category, framework=framework, benchmark=benchmark)
    elif benchmark == "KernelBench":
        # KernelBench: 按序号读取
        benchmark_name = get_benchmark_name(
            task_index_list=[19, ], framework=framework, benchmark=benchmark)
    else:
        # 不支持的 benchmark 类型
        print(f"警告: 不支持的 benchmark 类型 '{benchmark}'")
        print(f"当前支持的 benchmark 类型: KernelBench, multiKernelBench")
        benchmark_name = None

    if benchmark_name is None:
        print(f"跳过测试: benchmark '{benchmark}' 不支持")
        return

    result_dict = defaultdict(int)

    for i in range(len(benchmark_name)):
        task_desc = get_benchmark_task(
            benchmark_name[i], framework=framework, benchmark=benchmark)
        op_name = add_op_prefix(benchmark_name[i])

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
