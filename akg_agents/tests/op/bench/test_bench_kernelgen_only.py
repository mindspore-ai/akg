# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
KernelGen-Only Workflow 测试
基于 Skill 系统的内核代码生成工作流
"""

import os
import pytest
from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task
from ..utils import (
    get_kernelbench_op_name,
    get_kernelbench_task_desc,
    add_op_prefix,
    generate_beautiful_test_report,
    get_device_id
)

# 设置数据收集环境变量（与其他测试保持一致）
os.environ['AKG_AGENTS_DATA_COLLECT'] = 'on'

# 指定设备 ID
device_id = get_device_id()


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_torch_triton_ascend910b4():
    """测试 KernelBench - PyTorch Triton Ascend910B4 (KernelGen-Only Workflow)"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    
    # 加载 KernelGen-only 配置
    config = load_config(config_path="./python/akg_agents/op/config/vllm_triton_ascend_kernelgen_config.yaml")

    check_env_for_task(framework, backend, dsl, config)

    # 注册 LocalWorker
    await register_local_worker([device_id], backend=backend, arch=arch)

    # KernelBench: 按序号读取（可以指定多个任务进行测试）
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19, ], framework=framework)

    if benchmark_name is None:
        raise RuntimeError("在 KernelBench 中未找到指定序号的任务文件，请检查 task_index_list 参数是否正确")

    for i in range(len(benchmark_name)):
        task_desc = get_kernelbench_task_desc(
            benchmark_name[i], framework=framework)
        op_name = add_op_prefix(benchmark_name[i], benchmark=benchmark)

        task = AIKGTask(
            op_name=op_name,
            task_desc=task_desc,
            task_id=str(i),
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            framework=framework,
            workflow="kernelgen_only_workflow"  # 使用 kernelgen_only_workflow
        )
        task_pool.create_task(task.run)

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )
