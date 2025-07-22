import os
from pathlib import Path
import pytest
from collections import defaultdict
from ai_kernel_generator.core.coder_only_task import CoderOnlyTask
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ..utils import get_benchmark_name, get_benchmark_task, add_op_prefix, get_folder_names, get_task_content
from ai_kernel_generator.config.config_validator import load_config
from functools import partial
from ai_kernel_generator.core.action_type import ActionType


@pytest.mark.asyncio
@pytest.mark.parametrize("framework,impl_type,backend,arch", [
    ("mindspore", "triton", "ascend", "ascend910b4"),
    ("torch", "triton-russia", "ascend", "ascend910b4"),
])
async def test_parallel_coder_only_task_triton_ascend910b4(framework, impl_type, backend, arch):
    task_pool = TaskPool(1)
    device_pool = DevicePool([1, 2])
    config = load_config()  # or load_config("/your-path-to-config/xxx_config.yaml")
    task_names = get_benchmark_name([19, ], framework=framework)
    # task_names = get_folder_names("/your-path-to-torch-files")

    result_dict = defaultdict(int)

    for i in range(len(task_names)):
        task_desc = get_benchmark_task(task_names[i], framework=framework)
        # task_desc = get_task_content("/your-path-to-torch-files", task_names[i])
        op_name = add_op_prefix(task_names[i])

        task = CoderOnlyTask(
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
        task_pool.create_task(partial(task.run, init_action_type=ActionType.DO_CODER_DIRECT))

    results = await task_pool.wait_all()
    for op_name, result in results:
        result_dict[op_name] += result

    pass_num = 0
    for result in result_dict.values():
        if result:
            pass_num += 1
    pass_rate = pass_num / len(result_dict)

    print('-' * 60)
    print("result_dict: ", result_dict)
    print("PASS RATE: ", pass_rate)
    result_path = os.path.join(os.path.expanduser(config["log_dir"]), "test_coder_only_results.txt")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"结果字典: {dict(result_dict)}\n")
        f.write(f"通过率: {pass_rate}\n")
    print("结果已保存到 test_coder_only_results.txt")
    print('-' * 60)
